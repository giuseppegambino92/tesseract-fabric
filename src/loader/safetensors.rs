//! SafeTensors: parser over a zero-copy mapping, and writer.
//!
//! Layout: `u64` little-endian header length `N`, an `N`-byte JSON header
//! `{"name": {"dtype": "F32", "shape": [..], "data_offsets": [begin, end]},
//! "__metadata__": {..}}` (offsets relative to the byte after the header),
//! then the raw little-endian tensor bytes.

use super::mmap::MmapFile;
use crate::server::json::Json;
use crate::tensor::quant::{ByteSource, QuantView, WeightFormat};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    F64,
    F32,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
}

impl Dtype {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "F64" => Dtype::F64,
            "F32" => Dtype::F32,
            "F16" => Dtype::F16,
            "BF16" => Dtype::BF16,
            "I64" => Dtype::I64,
            "I32" => Dtype::I32,
            "I16" => Dtype::I16,
            "I8" => Dtype::I8,
            "U8" => Dtype::U8,
            "BOOL" => Dtype::Bool,
            _ => return None,
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            Dtype::F64 => "F64",
            Dtype::F32 => "F32",
            Dtype::F16 => "F16",
            Dtype::BF16 => "BF16",
            Dtype::I64 => "I64",
            Dtype::I32 => "I32",
            Dtype::I16 => "I16",
            Dtype::I8 => "I8",
            Dtype::U8 => "U8",
            Dtype::Bool => "BOOL",
        }
    }
    pub fn size(self) -> usize {
        match self {
            Dtype::F64 | Dtype::I64 => 8,
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::F16 | Dtype::BF16 | Dtype::I16 => 2,
            Dtype::I8 | Dtype::U8 | Dtype::Bool => 1,
        }
    }
    /// The matching weight format for matrix views.
    pub fn weight_format(self) -> Option<WeightFormat> {
        Some(match self {
            Dtype::F32 => WeightFormat::F32,
            Dtype::F16 => WeightFormat::F16,
            Dtype::BF16 => WeightFormat::BF16,
            _ => return None,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct StTensorInfo {
    pub name: String,
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// Absolute byte range in the file.
    pub start: usize,
    pub end: usize,
}

#[derive(Debug)]
pub struct StError(pub String);
impl std::fmt::Display for StError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SafeTensors: {}", self.0)
    }
}
impl std::error::Error for StError {}

fn err<T>(m: impl Into<String>) -> Result<T, StError> {
    Err(StError(m.into()))
}

pub struct SafeTensorsFile {
    pub map: Arc<MmapFile>,
    pub tensors: Vec<StTensorInfo>,
    pub metadata: Vec<(String, String)>,
}

/// Largest header accepted (the format's own 100 MB guard).
const MAX_HEADER: usize = 100 << 20;

impl SafeTensorsFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, StError> {
        let map = MmapFile::open(path.as_ref())
            .map_err(|e| StError(format!("{}: {e}", path.as_ref().display())))?;
        Self::parse(Arc::new(map))
    }

    pub fn parse(map: Arc<MmapFile>) -> Result<Self, StError> {
        let b = map.as_slice();
        if b.len() < 8 {
            return err("file shorter than its header length");
        }
        let n = u64::from_le_bytes(b[..8].try_into().unwrap()) as usize;
        if n > MAX_HEADER || 8 + n > b.len() {
            return err("invalid header length");
        }
        let text =
            std::str::from_utf8(&b[8..8 + n]).map_err(|_| StError("header is not UTF-8".into()))?;
        let j = Json::parse(text.trim_end()).map_err(|e| StError(e.to_string()))?;
        let obj = j
            .as_object()
            .ok_or_else(|| StError("header is not an object".into()))?;
        let base = 8 + n;
        let mut tensors = Vec::new();
        let mut metadata = Vec::new();
        for (k, v) in obj {
            if k == "__metadata__" {
                for (mk, mv) in v.as_object().unwrap_or(&[]) {
                    metadata.push((mk.clone(), mv.as_str().unwrap_or_default().to_string()));
                }
                continue;
            }
            let dtype = v
                .get("dtype")
                .and_then(|d| d.as_str())
                .and_then(Dtype::parse)
                .ok_or_else(|| StError(format!("{k}: missing/unknown dtype")))?;
            let shape: Vec<usize> = v
                .get("shape")
                .and_then(|s| s.as_array())
                .ok_or_else(|| StError(format!("{k}: missing shape")))?
                .iter()
                .map(|d| d.as_u64().map(|x| x as usize))
                .collect::<Option<_>>()
                .ok_or_else(|| StError(format!("{k}: bad shape")))?;
            let off = v
                .get("data_offsets")
                .and_then(|o| o.as_array())
                .filter(|o| o.len() == 2)
                .ok_or_else(|| StError(format!("{k}: missing data_offsets")))?;
            let (s, e) = match (off[0].as_u64(), off[1].as_u64()) {
                (Some(s), Some(e)) if s <= e => (s as usize, e as usize),
                _ => return err(format!("{k}: bad data_offsets")),
            };
            let numel: usize = shape.iter().product();
            if e - s != numel * dtype.size() {
                return err(format!("{k}: data_offsets do not match shape × dtype"));
            }
            if base + e > b.len() {
                return err(format!("{k}: data past end of file"));
            }
            tensors.push(StTensorInfo {
                name: k.clone(),
                dtype,
                shape,
                start: base + s,
                end: base + e,
            });
        }
        Ok(SafeTensorsFile {
            map,
            tensors,
            metadata,
        })
    }

    pub fn tensor(&self, name: &str) -> Option<&StTensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Raw bytes of a tensor (zero copy).
    pub fn bytes(&self, name: &str) -> Result<&[u8], StError> {
        let t = self
            .tensor(name)
            .ok_or_else(|| StError(format!("missing tensor {name}")))?;
        Ok(&self.map.as_slice()[t.start..t.end])
    }

    /// Zero-copy matrix view (`[rows, cols]`, row-major) of an F32/F16/BF16
    /// tensor; 1-D tensors are one row. `slab = Some(e)` views `[e]` of a
    /// 3-D `[n, rows, cols]` tensor.
    pub fn view(&self, name: &str, slab: Option<usize>) -> Result<QuantView, StError> {
        let t = self
            .tensor(name)
            .ok_or_else(|| StError(format!("missing tensor {name}")))?;
        let fmt = t.dtype.weight_format().ok_or_else(|| {
            StError(format!(
                "{name}: dtype {} is not a float weight",
                t.dtype.name()
            ))
        })?;
        let (rows, cols, off) = match (t.shape.len(), slab) {
            (1, None) => (1, t.shape[0], t.start),
            (2, None) => (t.shape[0], t.shape[1], t.start),
            (3, Some(e)) if e < t.shape[0] => {
                let (r, c) = (t.shape[1], t.shape[2]);
                (r, c, t.start + e * r * c * t.dtype.size())
            }
            _ => return err(format!("{name}: unexpected shape {:?}", t.shape)),
        };
        let src: Arc<dyn ByteSource> = self.map.clone();
        Ok(QuantView::new(src, off, fmt, rows, cols))
    }

    /// Decode any float tensor to f32 (copy).
    pub fn to_f32(&self, name: &str) -> Result<Vec<f32>, StError> {
        let v = self.view(name, None)?;
        Ok(v.dequantize().data)
    }
}

/// Builds a SafeTensors file.
#[derive(Default)]
pub struct SafeTensorsWriter {
    tensors: Vec<(String, Dtype, Vec<usize>, Vec<u8>)>,
    metadata: Vec<(String, String)>,
}

impl SafeTensorsWriter {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add_metadata(&mut self, k: &str, v: &str) -> &mut Self {
        self.metadata.push((k.into(), v.into()));
        self
    }
    pub fn add(&mut self, name: &str, dtype: Dtype, shape: &[usize], bytes: Vec<u8>) -> &mut Self {
        assert_eq!(
            shape.iter().product::<usize>() * dtype.size(),
            bytes.len(),
            "{name}: payload size"
        );
        self.tensors
            .push((name.into(), dtype, shape.to_vec(), bytes));
        self
    }
    /// Add f32 data encoded as F32 / F16 / BF16.
    pub fn add_f32(
        &mut self,
        name: &str,
        dtype: Dtype,
        shape: &[usize],
        data: &[f32],
    ) -> &mut Self {
        let fmt = dtype.weight_format().expect("float dtype");
        let cols = *shape.last().unwrap_or(&1);
        let bytes = crate::tensor::quant::encode_matrix(data, cols.max(1), fmt);
        self.add(name, dtype, shape, bytes)
    }

    pub fn write(&self, path: impl AsRef<Path>) -> std::io::Result<u64> {
        let mut entries = Vec::new();
        if !self.metadata.is_empty() {
            entries.push((
                "__metadata__".to_string(),
                Json::Obj(
                    self.metadata
                        .iter()
                        .map(|(k, v)| (k.clone(), Json::str(v.clone())))
                        .collect(),
                ),
            ));
        }
        let mut off = 0usize;
        for (name, dtype, shape, bytes) in &self.tensors {
            entries.push((
                name.clone(),
                Json::obj(vec![
                    ("dtype", Json::str(dtype.name())),
                    (
                        "shape",
                        Json::Arr(shape.iter().map(|&d| Json::num(d as f64)).collect()),
                    ),
                    (
                        "data_offsets",
                        Json::Arr(vec![
                            Json::num(off as f64),
                            Json::num((off + bytes.len()) as f64),
                        ]),
                    ),
                ]),
            ));
            off += bytes.len();
        }
        let mut header = Json::Obj(entries).dump().into_bytes();
        // pad with spaces so the data section starts 8-byte aligned
        while !(8 + header.len()).is_multiple_of(8) {
            header.push(b' ');
        }
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(&(header.len() as u64).to_le_bytes())?;
        f.write_all(&header)?;
        for (_, _, _, bytes) in &self.tensors {
            f.write_all(bytes)?;
        }
        f.flush()?;
        Ok((8 + header.len() + off) as u64)
    }
}
