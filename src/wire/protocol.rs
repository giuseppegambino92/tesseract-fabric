//! `TesseractWire` binary framing: a fixed 64-byte little-endian header
//! followed by `payload_len` bytes. CRC32C (Castagnoli) covers the header
//! (with the crc field zeroed) and the payload.
//!
//! ```text
//!  off size field
//!   0   4   magic  = 0x54455353 ("TESS")
//!   4   2   version
//!   6   1   frame_type
//!   7   1   flags
//!   8   8   sequence_id
//!  16   8   tree_epoch
//!  24   2   source_node_id
//!  26   2   target_node_id
//!  28   2   layer_id
//!  30   2   head_group_id
//!  32   4   payload_len
//!  36   4   crc32c
//!  40   8   canonical_scale_q16
//!  48   4   stripe_seq     (multipath: sub-frame index)    } reserved
//!  52   4   stripe_count   (multipath: total sub-frames)   } region
//!  56   8   send_ts_ns     (link telemetry)                } (16 bytes)
//! ```

use std::io::{self, Read, Write};

pub const MAGIC: u32 = 0x5445_5353;
pub const VERSION: u16 = 1;
pub const HEADER_LEN: usize = 64;
pub const MAX_PAYLOAD: usize = 256 << 20;

#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FrameType {
    ActivationDelta = 1,
    SpeculativeTreeBatch = 2,
    PreWakeSignal = 3,
    KvQueryBroadcast = 4,
    KvLsePartial = 5,
    TreeCommit = 6,
    HeartbeatTelemetry = 7,
    // control / baseline frames
    Hello = 16,
    Assign = 17,
    AssignAck = 18,
    RawActivation = 19,
    AllReducePartial = 20,
    TokenBroadcast = 21,
    KvAppend = 22,
    DraftRequest = 23,
    DraftTree = 24,
    RouteFeedback = 25,
    StripeAck = 26,
    Shutdown = 27,
    Stats = 28,
    Logits = 29,
    Reset = 30,
    Generate = 31,
    GenerateResult = 32,
    StatsRequest = 33,
    /// Head → driver: tokens accepted by one verification step (streaming).
    TokenStream = 34,
    /// Link probe: echo request / reply (discovery RTT + bandwidth).
    ProbePing = 35,
    ProbePong = 36,
    /// Head → ranks: one batched forward pass over several sessions.
    BatchStep = 37,
    /// Head → ranks: per-session accepted paths of a batched step.
    BatchCommit = 38,
}

impl FrameType {
    pub fn from_u8(v: u8) -> Option<Self> {
        use FrameType::*;
        Some(match v {
            1 => ActivationDelta,
            2 => SpeculativeTreeBatch,
            3 => PreWakeSignal,
            4 => KvQueryBroadcast,
            5 => KvLsePartial,
            6 => TreeCommit,
            7 => HeartbeatTelemetry,
            16 => Hello,
            17 => Assign,
            18 => AssignAck,
            19 => RawActivation,
            20 => AllReducePartial,
            21 => TokenBroadcast,
            22 => KvAppend,
            23 => DraftRequest,
            24 => DraftTree,
            25 => RouteFeedback,
            26 => StripeAck,
            27 => Shutdown,
            28 => Stats,
            29 => Logits,
            30 => Reset,
            31 => Generate,
            32 => GenerateResult,
            33 => StatsRequest,
            34 => TokenStream,
            35 => ProbePing,
            36 => ProbePong,
            37 => BatchStep,
            38 => BatchCommit,
            _ => return None,
        })
    }
}

pub mod flags {
    pub const COMPRESSED_HOLOGRAPHIC: u8 = 1 << 0;
    pub const FP16_PAYLOAD: u8 = 1 << 1;
    pub const BIG_ENDIAN: u8 = 1 << 2; // never set: we always emit LE
    pub const DETERMINISTIC_ANCHOR: u8 = 1 << 3;
    pub const STRIPED: u8 = 1 << 4;
    pub const OUT_OF_BAND: u8 = 1 << 5;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameHeader {
    pub version: u16,
    pub frame_type: FrameType,
    pub flags: u8,
    pub sequence_id: u64,
    pub tree_epoch: u64,
    pub source: u16,
    pub target: u16,
    pub layer: u16,
    pub head_group: u16,
    pub payload_len: u32,
    pub crc32c: u32,
    pub canonical_scale_q16: u64,
    pub stripe_seq: u32,
    pub stripe_count: u32,
    pub send_ts_ns: u64,
}

impl FrameHeader {
    pub fn new(frame_type: FrameType, source: u16, target: u16) -> Self {
        FrameHeader {
            version: VERSION,
            frame_type,
            flags: 0,
            sequence_id: 0,
            tree_epoch: 0,
            source,
            target,
            layer: 0,
            head_group: 0,
            payload_len: 0,
            crc32c: 0,
            canonical_scale_q16: 0,
            stripe_seq: 0,
            stripe_count: 1,
            send_ts_ns: 0,
        }
    }

    /// Serialise into exactly 64 bytes (no allocation).
    pub fn write_to(&self, b: &mut [u8; HEADER_LEN]) {
        b[0..4].copy_from_slice(&MAGIC.to_le_bytes());
        b[4..6].copy_from_slice(&self.version.to_le_bytes());
        b[6] = self.frame_type as u8;
        b[7] = self.flags;
        b[8..16].copy_from_slice(&self.sequence_id.to_le_bytes());
        b[16..24].copy_from_slice(&self.tree_epoch.to_le_bytes());
        b[24..26].copy_from_slice(&self.source.to_le_bytes());
        b[26..28].copy_from_slice(&self.target.to_le_bytes());
        b[28..30].copy_from_slice(&self.layer.to_le_bytes());
        b[30..32].copy_from_slice(&self.head_group.to_le_bytes());
        b[32..36].copy_from_slice(&self.payload_len.to_le_bytes());
        b[36..40].copy_from_slice(&self.crc32c.to_le_bytes());
        b[40..48].copy_from_slice(&self.canonical_scale_q16.to_le_bytes());
        b[48..52].copy_from_slice(&self.stripe_seq.to_le_bytes());
        b[52..56].copy_from_slice(&self.stripe_count.to_le_bytes());
        b[56..64].copy_from_slice(&self.send_ts_ns.to_le_bytes());
    }

    /// Parse a header view (zero-copy over the input slice).
    pub fn parse(b: &[u8; HEADER_LEN]) -> Result<Self, WireError> {
        let u16_at = |o: usize| u16::from_le_bytes([b[o], b[o + 1]]);
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        if u32_at(0) != MAGIC {
            return Err(WireError::BadMagic(u32_at(0)));
        }
        let version = u16_at(4);
        if version != VERSION {
            return Err(WireError::BadVersion(version));
        }
        let frame_type = FrameType::from_u8(b[6]).ok_or(WireError::BadType(b[6]))?;
        let payload_len = u32_at(32);
        if payload_len as usize > MAX_PAYLOAD {
            return Err(WireError::TooLarge(payload_len));
        }
        Ok(FrameHeader {
            version,
            frame_type,
            flags: b[7],
            sequence_id: u64_at(8),
            tree_epoch: u64_at(16),
            source: u16_at(24),
            target: u16_at(26),
            layer: u16_at(28),
            head_group: u16_at(30),
            payload_len,
            crc32c: u32_at(36),
            canonical_scale_q16: u64_at(40),
            stripe_seq: u32_at(48),
            stripe_count: u32_at(52),
            send_ts_ns: u64_at(56),
        })
    }
}

#[derive(Debug)]
pub enum WireError {
    BadMagic(u32),
    BadVersion(u16),
    BadType(u8),
    TooLarge(u32),
    CrcMismatch { expected: u32, actual: u32 },
    Io(io::Error),
}

impl std::fmt::Display for WireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for WireError {}
impl From<io::Error> for WireError {
    fn from(e: io::Error) -> Self {
        WireError::Io(e)
    }
}

/// A complete frame (header + owned payload).
#[derive(Clone, Debug, PartialEq)]
pub struct TesseractFrame {
    pub header: FrameHeader,
    pub payload: Vec<u8>,
}

impl TesseractFrame {
    pub fn new(frame_type: FrameType, source: u16, target: u16, payload: Vec<u8>) -> Self {
        let mut header = FrameHeader::new(frame_type, source, target);
        header.payload_len = payload.len() as u32;
        TesseractFrame { header, payload }
    }

    pub fn wire_len(&self) -> usize {
        HEADER_LEN + self.payload.len()
    }

    /// Compute CRC over header (crc field zeroed) + payload.
    pub fn compute_crc(header: &FrameHeader, payload: &[u8]) -> u32 {
        let mut h = *header;
        h.crc32c = 0;
        h.payload_len = payload.len() as u32;
        let mut hb = [0u8; HEADER_LEN];
        h.write_to(&mut hb);
        let c = crc32c_update(!0, &hb);
        !crc32c_update(c, payload)
    }

    /// Seal (fill payload_len + crc) and serialise into one contiguous buffer.
    pub fn encode(&self) -> Vec<u8> {
        let mut h = self.header;
        h.payload_len = self.payload.len() as u32;
        h.crc32c = Self::compute_crc(&h, &self.payload);
        let mut out = Vec::with_capacity(HEADER_LEN + self.payload.len());
        let mut hb = [0u8; HEADER_LEN];
        h.write_to(&mut hb);
        out.extend_from_slice(&hb);
        out.extend_from_slice(&self.payload);
        out
    }

    pub fn decode(buf: &[u8]) -> Result<Self, WireError> {
        if buf.len() < HEADER_LEN {
            return Err(WireError::Io(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "short frame",
            )));
        }
        let hb: &[u8; HEADER_LEN] = buf[..HEADER_LEN].try_into().unwrap();
        let header = FrameHeader::parse(hb)?;
        let end = HEADER_LEN + header.payload_len as usize;
        if buf.len() < end {
            return Err(WireError::Io(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "short payload",
            )));
        }
        let payload = buf[HEADER_LEN..end].to_vec();
        let actual = Self::compute_crc(&header, &payload);
        if actual != header.crc32c {
            return Err(WireError::CrcMismatch {
                expected: header.crc32c,
                actual,
            });
        }
        Ok(TesseractFrame { header, payload })
    }

    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<(), WireError> {
        w.write_all(&self.encode())?;
        Ok(())
    }

    pub fn read_from<R: Read>(r: &mut R) -> Result<Self, WireError> {
        let mut hb = [0u8; HEADER_LEN];
        r.read_exact(&mut hb)?;
        let header = FrameHeader::parse(&hb)?;
        let mut payload = vec![0u8; header.payload_len as usize];
        r.read_exact(&mut payload)?;
        let actual = Self::compute_crc(&header, &payload);
        if actual != header.crc32c {
            return Err(WireError::CrcMismatch {
                expected: header.crc32c,
                actual,
            });
        }
        Ok(TesseractFrame { header, payload })
    }
}

// ---------------------------------------------------------------------------
// CRC32C (Castagnoli, reflected poly 0x82F63B78), slicing-by-8.
// ---------------------------------------------------------------------------

const fn make_crc_tables() -> [[u32; 256]; 8] {
    let mut t = [[0u32; 256]; 8];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                (c >> 1) ^ 0x82F6_3B78
            } else {
                c >> 1
            };
            k += 1;
        }
        t[0][i] = c;
        i += 1;
    }
    let mut s = 1;
    while s < 8 {
        let mut i = 0;
        while i < 256 {
            let prev = t[s - 1][i];
            t[s][i] = (prev >> 8) ^ t[0][(prev & 0xff) as usize];
            i += 1;
        }
        s += 1;
    }
    t
}

static CRC_TABLES: [[u32; 256]; 8] = make_crc_tables();

/// Raw CRC32C update (caller handles init `!0` and final inversion).
pub fn crc32c_update(mut crc: u32, data: &[u8]) -> u32 {
    let t = &CRC_TABLES;
    let (chunks, rest) = data.as_chunks::<8>();
    for c in chunks {
        let lo = u32::from_le_bytes([c[0], c[1], c[2], c[3]]) ^ crc;
        let hi = u32::from_le_bytes([c[4], c[5], c[6], c[7]]);
        crc = t[7][(lo & 0xff) as usize]
            ^ t[6][((lo >> 8) & 0xff) as usize]
            ^ t[5][((lo >> 16) & 0xff) as usize]
            ^ t[4][(lo >> 24) as usize]
            ^ t[3][(hi & 0xff) as usize]
            ^ t[2][((hi >> 8) & 0xff) as usize]
            ^ t[1][((hi >> 16) & 0xff) as usize]
            ^ t[0][(hi >> 24) as usize];
    }
    for &b in rest {
        crc = (crc >> 8) ^ t[0][((crc ^ b as u32) & 0xff) as usize];
    }
    crc
}

pub fn crc32c(data: &[u8]) -> u32 {
    !crc32c_update(!0, data)
}

// ---------------------------------------------------------------------------
// Little-endian payload builders
// ---------------------------------------------------------------------------

#[derive(Default)]
pub struct ByteWriter {
    buf: Vec<u8>,
}

impl ByteWriter {
    pub fn with_capacity(n: usize) -> Self {
        ByteWriter {
            buf: Vec::with_capacity(n),
        }
    }
    pub fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    pub fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    pub fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    pub fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    pub fn f32(&mut self, v: f32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    pub fn f32_slice(&mut self, v: &[f32]) {
        self.buf.reserve(v.len() * 4);
        for x in v {
            self.buf.extend_from_slice(&x.to_le_bytes());
        }
    }
    pub fn bytes(&mut self, v: &[u8]) {
        self.buf.extend_from_slice(v);
    }
    pub fn len(&self) -> usize {
        self.buf.len()
    }
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }
    pub fn into_inner(self) -> Vec<u8> {
        self.buf
    }
}

pub struct ByteReader<'a> {
    b: &'a [u8],
    pub pos: usize,
}

impl<'a> ByteReader<'a> {
    pub fn new(b: &'a [u8]) -> Self {
        ByteReader { b, pos: 0 }
    }
    pub fn u8(&mut self) -> u8 {
        let v = self.b[self.pos];
        self.pos += 1;
        v
    }
    pub fn u16(&mut self) -> u16 {
        let v = u16::from_le_bytes([self.b[self.pos], self.b[self.pos + 1]]);
        self.pos += 2;
        v
    }
    pub fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.b[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        v
    }
    pub fn u64(&mut self) -> u64 {
        let v = u64::from_le_bytes(self.b[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        v
    }
    pub fn f32(&mut self) -> f32 {
        f32::from_bits(self.u32())
    }
    pub fn f32_vec(&mut self, n: usize) -> Vec<f32> {
        let mut v = Vec::with_capacity(n);
        for i in 0..n {
            let o = self.pos + 4 * i;
            v.push(f32::from_le_bytes(self.b[o..o + 4].try_into().unwrap()));
        }
        self.pos += 4 * n;
        v
    }
    pub fn bytes(&mut self, n: usize) -> &'a [u8] {
        let s = &self.b[self.pos..self.pos + n];
        self.pos += n;
        s
    }
    pub fn rest(&self) -> &'a [u8] {
        &self.b[self.pos..]
    }
    pub fn remaining(&self) -> usize {
        self.b.len() - self.pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32c_known_vector() {
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
        assert_eq!(crc32c(b""), 0);
    }

    #[test]
    fn header_is_64_bytes_and_roundtrips() {
        let mut f = TesseractFrame::new(FrameType::TreeCommit, 3, 7, vec![1, 2, 3, 4, 5]);
        f.header.sequence_id = 99;
        f.header.tree_epoch = 12;
        f.header.layer = 5;
        f.header.flags = flags::DETERMINISTIC_ANCHOR;
        let enc = f.encode();
        assert_eq!(enc.len(), 64 + 5);
        let d = TesseractFrame::decode(&enc).unwrap();
        assert_eq!(d.payload, f.payload);
        assert_eq!(d.header.sequence_id, 99);
        let mut bad = enc.clone();
        bad[70 - 2] ^= 1;
        assert!(matches!(
            TesseractFrame::decode(&bad),
            Err(WireError::CrcMismatch { .. })
        ));
    }
}
