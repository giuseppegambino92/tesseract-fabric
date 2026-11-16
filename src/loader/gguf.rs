//! GGUF (v2 / v3) binary format: parser over a zero-copy mapping, and writer.
//!
//! Layout (little endian): magic `GGUF` (0x46554747), `u32` version,
//! `u64` tensor count, `u64` metadata count, metadata key/values, tensor
//! descriptors (name, `u32` n_dims, `u64` dims — innermost first —, `u32`
//! GGML type, `u64` offset into the data section), padding to
//! `general.alignment` (default 32), then the tensor data, each tensor
//! aligned. Strings are `u64` length + UTF-8 bytes.

use super::mmap::MmapFile;
use crate::tensor::quant::{ByteSource, QuantView, WeightFormat};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;

pub const GGUF_MAGIC: u32 = 0x4655_4747;
pub const DEFAULT_ALIGNMENT: u64 = 32;

#[derive(Clone, Debug, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    /// Element type id + items.
    Arr(u32, Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufValue {
    pub fn type_id(&self) -> u32 {
        match self {
            GgufValue::U8(_) => 0,
            GgufValue::I8(_) => 1,
            GgufValue::U16(_) => 2,
            GgufValue::I16(_) => 3,
            GgufValue::U32(_) => 4,
            GgufValue::I32(_) => 5,
            GgufValue::F32(_) => 6,
            GgufValue::Bool(_) => 7,
            GgufValue::Str(_) => 8,
            GgufValue::Arr(..) => 9,
            GgufValue::U64(_) => 10,
            GgufValue::I64(_) => 11,
            GgufValue::F64(_) => 12,
        }
    }
    /// Any integer value as u64 (negative → None).
    pub fn as_u64(&self) -> Option<u64> {
        Some(match *self {
            GgufValue::U8(v) => v as u64,
            GgufValue::U16(v) => v as u64,
            GgufValue::U32(v) => v as u64,
            GgufValue::U64(v) => v,
            GgufValue::I8(v) if v >= 0 => v as u64,
            GgufValue::I16(v) if v >= 0 => v as u64,
            GgufValue::I32(v) if v >= 0 => v as u64,
            GgufValue::I64(v) if v >= 0 => v as u64,
            _ => return None,
        })
    }
    pub fn as_f64(&self) -> Option<f64> {
        match *self {
            GgufValue::F32(v) => Some(v as f64),
            GgufValue::F64(v) => Some(v),
            _ => self.as_u64().map(|v| v as f64),
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::Str(s) => Some(s),
            _ => None,
        }
    }
    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            GgufValue::Arr(_, v) => Some(v),
            _ => None,
        }
    }
    pub fn str_array(v: impl IntoIterator<Item = String>) -> Self {
        GgufValue::Arr(8, v.into_iter().map(GgufValue::Str).collect())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GgufTensorInfo {
    pub name: String,
    /// Dimensions, innermost (contiguous) first.
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    /// Offset relative to the data section.
    pub offset: u64,
}

impl GgufTensorInfo {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }
    pub fn format(&self) -> Option<WeightFormat> {
        WeightFormat::from_ggml(self.ggml_type)
    }
    /// Stored byte size (None for types this engine does not implement).
    pub fn byte_size(&self) -> Option<u64> {
        let f = self.format()?;
        let n = self.n_elements();
        if !n.is_multiple_of(f.block_len() as u64) {
            return None;
        }
        Some(f.row_bytes(n as usize) as u64)
    }
}

#[derive(Debug)]
pub struct GgufError(pub String);

impl std::fmt::Display for GgufError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GGUF: {}", self.0)
    }
}
impl std::error::Error for GgufError {}

fn err<T>(m: impl Into<String>) -> Result<T, GgufError> {
    Err(GgufError(m.into()))
}

struct Cursor<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], GgufError> {
        if self.i.checked_add(n).is_none_or(|e| e > self.b.len()) {
            return err(format!("truncated file at byte {}", self.i));
        }
        let s = &self.b[self.i..self.i + n];
        self.i += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, GgufError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, GgufError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String, GgufError> {
        let n = self.u64()? as usize;
        if n > self.b.len() {
            return err("string length exceeds file");
        }
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| GgufError("non-UTF-8 string".into()))
    }
    fn value(&mut self, ty: u32, depth: usize) -> Result<GgufValue, GgufError> {
        let v = match ty {
            0 => GgufValue::U8(self.take(1)?[0]),
            1 => GgufValue::I8(self.take(1)?[0] as i8),
            2 => GgufValue::U16(u16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            3 => GgufValue::I16(i16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            4 => GgufValue::U32(self.u32()?),
            5 => GgufValue::I32(self.u32()? as i32),
            6 => GgufValue::F32(f32::from_bits(self.u32()?)),
            7 => GgufValue::Bool(self.take(1)?[0] != 0),
            8 => GgufValue::Str(self.string()?),
            9 => {
                if depth > 4 {
                    return err("arrays nested too deeply");
                }
                let et = self.u32()?;
                let n = self.u64()? as usize;
                if n > self.b.len() {
                    return err("array length exceeds file");
                }
                let mut items = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    items.push(self.value(et, depth + 1)?);
                }
                GgufValue::Arr(et, items)
            }
            10 => GgufValue::U64(self.u64()?),
            11 => GgufValue::I64(self.u64()? as i64),
            12 => GgufValue::F64(f64::from_bits(self.u64()?)),
            t => return err(format!("unknown metadata type {t}")),
        };
        Ok(v)
    }
}

/// A parsed GGUF file whose tensors are zero-copy views into its mapping.
pub struct GgufFile {
    pub map: Arc<MmapFile>,
    pub version: u32,
    pub metadata: Vec<(String, GgufValue)>,
    pub tensors: Vec<GgufTensorInfo>,
    pub alignment: u64,
    /// Absolute byte offset of the data section.
    pub data_offset: u64,
}

impl GgufFile {
    pub fn open(path: impl AsRef<Path>) -> Result<Self, GgufError> {
        let map = MmapFile::open(path.as_ref())
            .map_err(|e| GgufError(format!("{}: {e}", path.as_ref().display())))?;
        Self::parse(Arc::new(map))
    }

    pub fn parse(map: Arc<MmapFile>) -> Result<Self, GgufError> {
        let b = map.as_slice();
        let mut c = Cursor { b, i: 0 };
        if c.u32()? != GGUF_MAGIC {
            return err("bad magic (not a GGUF file)");
        }
        let version = c.u32()?;
        if !(2..=3).contains(&version) {
            return err(format!("unsupported GGUF version {version} (need 2 or 3)"));
        }
        let n_tensors = c.u64()? as usize;
        let n_kv = c.u64()? as usize;
        if n_tensors > b.len() || n_kv > b.len() {
            return err("implausible header counts");
        }
        let mut metadata = Vec::with_capacity(n_kv);
        for _ in 0..n_kv {
            let k = c.string()?;
            let t = c.u32()?;
            metadata.push((k, c.value(t, 0)?));
        }
        let mut tensors = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = c.string()?;
            let nd = c.u32()? as usize;
            if nd == 0 || nd > 4 {
                return err(format!("tensor {name}: {nd} dims"));
            }
            let dims = (0..nd).map(|_| c.u64()).collect::<Result<Vec<_>, _>>()?;
            let ggml_type = c.u32()?;
            let offset = c.u64()?;
            tensors.push(GgufTensorInfo {
                name,
                dims,
                ggml_type,
                offset,
            });
        }
        let alignment = metadata
            .iter()
            .find(|(k, _)| k == "general.alignment")
            .and_then(|(_, v)| v.as_u64())
            .unwrap_or(DEFAULT_ALIGNMENT);
        if alignment == 0 || !alignment.is_power_of_two() {
            return err("general.alignment must be a power of two");
        }
        let data_offset = (c.i as u64).div_ceil(alignment) * alignment;
        for t in &tensors {
            if t.offset % alignment != 0 {
                return err(format!("tensor {} is not aligned", t.name));
            }
            if let Some(sz) = t.byte_size() {
                if data_offset + t.offset + sz > b.len() as u64 {
                    return err(format!("tensor {} extends past end of file", t.name));
                }
            }
        }
        Ok(GgufFile {
            map,
            version,
            metadata,
            tensors,
            alignment,
            data_offset,
        })
    }

    pub fn get(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }
    pub fn get_u64(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(|v| v.as_u64())
    }
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(|v| v.as_f64())
    }
    pub fn get_str(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(|v| v.as_str())
    }
    pub fn architecture(&self) -> Option<&str> {
        self.get_str("general.architecture")
    }
    /// `{arch}.{key}` hyper-parameter.
    pub fn arch_u64(&self, key: &str) -> Option<u64> {
        self.get_u64(&format!("{}.{key}", self.architecture()?))
    }
    pub fn arch_f64(&self, key: &str) -> Option<f64> {
        self.get_f64(&format!("{}.{key}", self.architecture()?))
    }

    pub fn tensor(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Absolute byte range of a tensor in the file.
    pub fn tensor_range(&self, t: &GgufTensorInfo) -> Result<(usize, usize), GgufError> {
        let sz = t.byte_size().ok_or_else(|| {
            GgufError(format!(
                "tensor {}: GGML type {} is not supported",
                t.name, t.ggml_type
            ))
        })?;
        let s = (self.data_offset + t.offset) as usize;
        Ok((s, s + sz as usize))
    }

    /// Raw bytes of a tensor (zero copy).
    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8], GgufError> {
        let t = self
            .tensor(name)
            .ok_or_else(|| GgufError(format!("missing tensor {name}")))?;
        let (s, e) = self.tensor_range(t)?;
        Ok(&self.map.as_slice()[s..e])
    }

    /// Zero-copy matrix view of a 2-D tensor (rows = dims[1], cols = dims[0])
    /// or, with `slab = Some(e)`, of slab `e` of a 3-D tensor
    /// (`dims = [cols, rows, n]`, e.g. the stacked expert weights).
    pub fn view(&self, name: &str, slab: Option<usize>) -> Result<QuantView, GgufError> {
        let t = self
            .tensor(name)
            .ok_or_else(|| GgufError(format!("missing tensor {name}")))?;
        let fmt = t.format().ok_or_else(|| {
            GgufError(format!(
                "tensor {name}: GGML type {} is not supported (use F32/F16/BF16/Q8_0/Q4_0)",
                t.ggml_type
            ))
        })?;
        let (s, _) = self.tensor_range(t)?;
        let cols = t.dims[0] as usize;
        let rows = *t.dims.get(1).unwrap_or(&1) as usize;
        let (off, rows) = match slab {
            None if t.dims.len() <= 2 => (s, rows),
            Some(e) if t.dims.len() == 3 && (e as u64) < t.dims[2] => {
                (s + e * rows * fmt.row_bytes(cols), rows)
            }
            _ => return err(format!("tensor {name}: unexpected shape {:?}", t.dims)),
        };
        let src: Arc<dyn ByteSource> = self.map.clone();
        Ok(QuantView::new(src, off, fmt, rows, cols))
    }

    /// A 1-D tensor (norm weights) decoded to f32.
    pub fn vector_f32(&self, name: &str) -> Result<Vec<f32>, GgufError> {
        let v = self.view(name, None)?;
        let mut out = vec![0.0f32; v.cols * v.rows];
        for r in 0..v.rows {
            v.decode_row_into(r, &mut out[r * v.cols..(r + 1) * v.cols]);
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// Builds a GGUF v3 file. Tensors are given as their final byte payloads in
/// a GGML type (see `tensor::quant::encode_matrix`).
pub struct GgufWriter {
    pub alignment: u64,
    kv: Vec<(String, GgufValue)>,
    tensors: Vec<(String, Vec<u64>, u32, Vec<u8>)>,
}

impl Default for GgufWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl GgufWriter {
    pub fn new() -> Self {
        GgufWriter {
            alignment: DEFAULT_ALIGNMENT,
            kv: Vec::new(),
            tensors: Vec::new(),
        }
    }
    pub fn add_kv(&mut self, key: &str, v: GgufValue) -> &mut Self {
        self.kv.retain(|(k, _)| k != key);
        self.kv.push((key.to_string(), v));
        self
    }
    /// Add a tensor (`dims` innermost first) with its encoded bytes.
    pub fn add_tensor(
        &mut self,
        name: &str,
        dims: &[u64],
        fmt: WeightFormat,
        bytes: Vec<u8>,
    ) -> &mut Self {
        let n: u64 = dims.iter().product();
        assert_eq!(
            fmt.row_bytes(n as usize),
            bytes.len(),
            "tensor {name}: payload size"
        );
        self.tensors
            .push((name.to_string(), dims.to_vec(), fmt.ggml_id(), bytes));
        self
    }

    fn str_bytes(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }

    fn value_bytes(out: &mut Vec<u8>, v: &GgufValue) {
        match v {
            GgufValue::U8(x) => out.push(*x),
            GgufValue::I8(x) => out.push(*x as u8),
            GgufValue::U16(x) => out.extend_from_slice(&x.to_le_bytes()),
            GgufValue::I16(x) => out.extend_from_slice(&x.to_le_bytes()),
            GgufValue::U32(x) => out.extend_from_slice(&x.to_le_bytes()),
            GgufValue::I32(x) => out.extend_from_slice(&x.to_le_bytes()),
            GgufValue::F32(x) => out.extend_from_slice(&x.to_le_bytes()),
            GgufValue::Bool(x) => out.push(*x as u8),
            GgufValue::Str(s) => Self::str_bytes(out, s),
            GgufValue::Arr(t, items) => {
                out.extend_from_slice(&t.to_le_bytes());
                out.extend_from_slice(&(items.len() as u64).to_le_bytes());
                for it in items {
                    assert_eq!(it.type_id(), *t, "heterogeneous GGUF array");
                    Self::value_bytes(out, it);
                }
            }
            GgufValue::U64(x) => out.extend_from_slice(&x.to_le_bytes()),
            GgufValue::I64(x) => out.extend_from_slice(&x.to_le_bytes()),
            GgufValue::F64(x) => out.extend_from_slice(&x.to_le_bytes()),
        }
    }

    /// Serialize to `path`; returns the file size in bytes.
    pub fn write(&self, path: impl AsRef<Path>) -> std::io::Result<u64> {
        let a = self.alignment;
        let mut kv = self.kv.clone();
        if !kv.iter().any(|(k, _)| k == "general.alignment") {
            kv.insert(0, ("general.alignment".into(), GgufValue::U32(a as u32)));
        }
        let mut head = Vec::new();
        head.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        head.extend_from_slice(&3u32.to_le_bytes());
        head.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        head.extend_from_slice(&(kv.len() as u64).to_le_bytes());
        for (k, v) in &kv {
            Self::str_bytes(&mut head, k);
            head.extend_from_slice(&v.type_id().to_le_bytes());
            Self::value_bytes(&mut head, v);
        }
        let mut off = 0u64;
        for (name, dims, ty, bytes) in &self.tensors {
            Self::str_bytes(&mut head, name);
            head.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in dims {
                head.extend_from_slice(&d.to_le_bytes());
            }
            head.extend_from_slice(&ty.to_le_bytes());
            head.extend_from_slice(&off.to_le_bytes());
            off = (off + bytes.len() as u64).div_ceil(a) * a;
        }
        let pad = |n: u64| (n.div_ceil(a) * a - n) as usize;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(&head)?;
        f.write_all(&vec![0u8; pad(head.len() as u64)])?;
        let mut total = head.len() as u64 + pad(head.len() as u64) as u64;
        for (_, _, _, bytes) in &self.tensors {
            f.write_all(bytes)?;
            let p = pad(bytes.len() as u64);
            f.write_all(&vec![0u8; p])?;
            total += (bytes.len() + p) as u64;
        }
        f.flush()?;
        Ok(total)
    }
}
