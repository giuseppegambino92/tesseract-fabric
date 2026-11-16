//! `TesseractWire`: binary frame codec, multipath scheduler, and the live TCP
//! transport with physical link shaping.

pub mod auth;
pub mod multipath;
pub mod protocol;
pub mod transport;

pub use multipath::{LinkKind, LinkProfile, MultipathScheduler};
pub use protocol::{FrameHeader, FrameType, TesseractFrame};
pub use transport::{Endpoint, PeerLinks};
