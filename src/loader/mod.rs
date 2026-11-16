//! Real model files: zero-copy `mmap` loading of GGUF (v2/v3) and
//! SafeTensors weights, writers for both formats, and the shard-selective
//! loader that builds exactly the layers / expert slices a node owns.

pub mod gguf;
pub mod mmap;
pub mod safetensors;
pub mod sharded_loader;

pub use gguf::{GgufFile, GgufValue, GgufWriter};
pub use mmap::MmapFile;
pub use safetensors::{SafeTensorsFile, SafeTensorsWriter};
