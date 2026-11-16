//! Print a GGUF file's metadata (arrays summarised) and tensor-type mix:
//! `cargo run --release --example gguf_inspect -- model.gguf`.
use std::collections::BTreeMap;
use tesseract_fabric::loader::{GgufFile, GgufValue};
fn main() {
    let path = std::env::args().nth(1).expect("usage: gguf_inspect FILE");
    let g = GgufFile::open(&path).unwrap_or_else(|e| panic!("{e}"));
    println!(
        "GGUF v{} · {} tensors · {} metadata keys · alignment {}",
        g.version,
        g.tensors.len(),
        g.metadata.len(),
        g.alignment
    );
    for (k, v) in &g.metadata {
        let s = match v {
            GgufValue::Arr(_, a) => format!(
                "[{} items] {:?}",
                a.len(),
                a.iter().take(3).collect::<Vec<_>>()
            ),
            GgufValue::Str(s) if s.len() > 160 => format!(
                "{:?}… ({} chars)",
                s.chars().take(160).collect::<String>(),
                s.len()
            ),
            v => format!("{v:?}"),
        };
        println!("  {k} = {s}");
    }
    let mut mix: BTreeMap<String, (usize, u64)> = BTreeMap::new();
    for t in &g.tensors {
        let name = t
            .format()
            .map(|f| f.name().to_string())
            .unwrap_or(format!("ggml type {}", t.ggml_type));
        let e = mix.entry(name).or_default();
        e.0 += 1;
        e.1 += t.byte_size().unwrap_or(0);
    }
    println!("tensor types:");
    for (k, (n, b)) in mix {
        println!("  {k:10} {n:4} tensors  {:8.1} MB", b as f64 / 1e6);
    }
    for t in g
        .tensors
        .iter()
        .filter(|t| t.name.starts_with("blk.0.") || !t.name.starts_with("blk."))
    {
        println!(
            "  {:32} {:?} {}",
            t.name,
            t.dims,
            t.format().map(|f| f.name()).unwrap_or("?")
        );
    }
}
