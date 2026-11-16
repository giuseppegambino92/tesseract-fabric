//! Zero-copy read-only file mapping.
//!
//! On Unix the file is mapped with POSIX `mmap(PROT_READ, MAP_PRIVATE)` through
//! a thin `extern "C"` binding to libc (no crates), unmapped on drop, and
//! hinted with `madvise`. Only the pages a node actually reads are faulted in,
//! so a shard that owns 4 of 32 layers touches ≈ 1/8 of the file. Elsewhere
//! the file is read into an owned buffer (same API, not zero-copy).

use crate::tensor::quant::ByteSource;
use std::fs::File;
use std::path::Path;

#[cfg(unix)]
mod sys {
    use std::os::raw::{c_int, c_void};
    pub const PROT_READ: c_int = 1;
    pub const MAP_PRIVATE: c_int = 2;
    pub const MADV_SEQUENTIAL: c_int = 2;
    pub const MADV_WILLNEED: c_int = 3;
    extern "C" {
        pub fn mmap(
            addr: *mut c_void,
            len: usize,
            prot: c_int,
            flags: c_int,
            fd: c_int,
            off: i64,
        ) -> *mut c_void;
        pub fn munmap(addr: *mut c_void, len: usize) -> c_int;
        pub fn madvise(addr: *mut c_void, len: usize, advice: c_int) -> c_int;
        pub fn getpagesize() -> c_int;
    }
}

/// Access-pattern hints for [`MmapFile::advise`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Advice {
    /// Pages will be needed soon (start read-ahead).
    WillNeed,
    /// Pages will be read front to back.
    Sequential,
}

pub struct MmapFile {
    ptr: *const u8,
    len: usize,
    /// Fallback storage (non-Unix, or empty files).
    owned: Vec<u8>,
    mapped: bool,
}

// SAFETY: the mapping is read-only and lives until drop; `&[u8]` views of it
// are as shareable as any immutable buffer.
unsafe impl Send for MmapFile {}
unsafe impl Sync for MmapFile {}

impl MmapFile {
    /// Map `path` read-only.
    pub fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let f = File::open(path.as_ref())?;
        let len = f.metadata()?.len() as usize;
        if len == 0 {
            return Ok(MmapFile {
                ptr: std::ptr::null(),
                len: 0,
                owned: Vec::new(),
                mapped: false,
            });
        }
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            // SAFETY: valid fd, length from metadata, read-only private map.
            let p = unsafe {
                sys::mmap(
                    std::ptr::null_mut(),
                    len,
                    sys::PROT_READ,
                    sys::MAP_PRIVATE,
                    f.as_raw_fd(),
                    0,
                )
            };
            if p as isize == -1 {
                return Err(std::io::Error::last_os_error());
            }
            Ok(MmapFile {
                ptr: p as *const u8,
                len,
                owned: Vec::new(),
                mapped: true,
            })
        }
        #[cfg(not(unix))]
        {
            use std::io::Read;
            let mut owned = Vec::with_capacity(len);
            let mut f = f;
            f.read_to_end(&mut owned)?;
            Ok(MmapFile {
                ptr: owned.as_ptr(),
                len: owned.len(),
                owned,
                mapped: false,
            })
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// True when the bytes are an OS mapping (zero copy).
    pub fn is_mapped(&self) -> bool {
        self.mapped
    }

    pub fn as_slice(&self) -> &[u8] {
        if self.mapped {
            // SAFETY: [ptr, ptr+len) is a live read-only mapping.
            unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
        } else {
            &self.owned
        }
    }

    /// Hint the kernel about `[offset, offset + len)` (page-aligned outward).
    /// A no-op without a mapping.
    pub fn advise(&self, offset: usize, len: usize, advice: Advice) {
        #[cfg(unix)]
        if self.mapped && len > 0 && offset < self.len {
            // SAFETY: getpagesize has no preconditions.
            let page = unsafe { sys::getpagesize() } as usize;
            let start = offset / page * page;
            let end = (offset + len).min(self.len);
            let a = match advice {
                Advice::WillNeed => sys::MADV_WILLNEED,
                Advice::Sequential => sys::MADV_SEQUENTIAL,
            };
            // SAFETY: the range lies inside our mapping; madvise is advisory.
            unsafe {
                sys::madvise(self.ptr.add(start) as *mut _, end - start, a);
            }
        }
        #[cfg(not(unix))]
        let _ = (offset, len, advice);
    }
}

impl ByteSource for MmapFile {
    fn bytes(&self) -> &[u8] {
        self.as_slice()
    }
    fn will_need(&self, offset: usize, len: usize) {
        self.advise(offset, len, Advice::WillNeed);
    }
    fn is_mapped(&self) -> bool {
        self.mapped
    }
}

impl Drop for MmapFile {
    fn drop(&mut self) {
        #[cfg(unix)]
        if self.mapped {
            // SAFETY: unmapping exactly the region returned by mmap.
            unsafe {
                sys::munmap(self.ptr as *mut _, self.len);
            }
        }
    }
}
