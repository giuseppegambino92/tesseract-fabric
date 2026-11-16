//! Lookahead expert routing and hot-buffer pre-staging.
//!
//! * `ExpertStore` — a layer's experts kept *cold* as int8 block-quantised
//!   matrices (¼ the bytes) with a bounded *hot* buffer of dequantised f32
//!   experts (LRU). Using a cold expert requires staging (real dequantisation
//!   work); doing it on the critical path is a blocking miss.
//! * `LookaheadPredictor` — W_look ∈ ℝ^{E×d} at layer L predicting the experts
//!   the router at layer L+ΔL will pick. Warm-started from that router (the
//!   residual stream is persistent across layers) and calibrated online with a
//!   rank-1 softmax-cross-entropy step whenever it misses.
//! * `Stager` — background thread that executes `PreWakeSignal`s so experts
//!   are hot before the activation that needs them arrives.

use crate::model::Expert;
use crate::tensor::{rms_norm, softmax_inplace, Linear, QMatrix, Tensor2D};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

#[derive(Clone, Debug)]
pub struct QExpert {
    pub gate: QMatrix,
    pub up: QMatrix,
    pub down: QMatrix,
}

impl QExpert {
    pub fn quantize(e: &Expert) -> Self {
        QExpert {
            gate: QMatrix::quantize(&e.w_gate.dense()),
            up: QMatrix::quantize(&e.w_up.dense()),
            down: QMatrix::quantize(&e.w_down.dense()),
        }
    }
    /// Stage: dequantise into a hot f32 expert.
    pub fn stage(&self) -> Expert {
        Expert {
            w_gate: Linear::new(self.gate.dequantize()),
            w_up: Linear::new(self.up.dequantize()),
            w_down: Linear::new(self.down.dequantize()),
        }
    }
    pub fn bytes(&self) -> usize {
        self.gate.bytes() + self.up.bytes() + self.down.bytes()
    }
}

#[derive(Default, Debug)]
pub struct StagingStats {
    pub hits: AtomicU64,
    pub blocking_misses: AtomicU64,
    pub blocking_ns: AtomicU64,
    pub prestaged: AtomicU64,
    pub prestage_ns: AtomicU64,
    pub waited_inflight: AtomicU64,
}

impl StagingStats {
    pub fn snapshot(&self) -> [u64; 6] {
        [
            self.hits.load(Ordering::Relaxed),
            self.blocking_misses.load(Ordering::Relaxed),
            self.blocking_ns.load(Ordering::Relaxed),
            self.prestaged.load(Ordering::Relaxed),
            self.prestage_ns.load(Ordering::Relaxed),
            self.waited_inflight.load(Ordering::Relaxed),
        ]
    }
}

struct HotState {
    entries: HashMap<usize, (Arc<Expert>, u64)>,
    inflight: HashSet<usize>,
    tick: u64,
}

pub struct ExpertStore {
    pub layer: usize,
    pub capacity: usize,
    cold: Vec<QExpert>,
    hot: Mutex<HotState>,
    cv: Condvar,
    pub stats: StagingStats,
}

impl ExpertStore {
    pub fn new(layer: usize, experts: &[Expert], capacity: usize) -> Self {
        let cold: Vec<QExpert> = experts.iter().map(QExpert::quantize).collect();
        let mut entries = HashMap::new();
        for (e, q) in cold.iter().enumerate().take(capacity) {
            entries.insert(e, (Arc::new(q.stage()), e as u64));
        }
        ExpertStore {
            layer,
            capacity: capacity.max(1),
            cold,
            hot: Mutex::new(HotState {
                entries,
                inflight: HashSet::new(),
                tick: capacity as u64,
            }),
            cv: Condvar::new(),
            stats: StagingStats::default(),
        }
    }

    pub fn n_experts(&self) -> usize {
        self.cold.len()
    }

    /// Bytes of the int8 cold copies.
    pub fn cold_bytes(&self) -> usize {
        self.cold.iter().map(|q| q.bytes()).sum()
    }

    pub fn is_hot(&self, e: usize) -> bool {
        self.hot.lock().unwrap().entries.contains_key(&e)
    }

    pub fn hot_set(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self.hot.lock().unwrap().entries.keys().copied().collect();
        v.sort();
        v
    }

    fn insert(&self, st: &mut HotState, e: usize, x: Arc<Expert>, protect: &[usize]) {
        st.tick += 1;
        let tick = st.tick;
        st.entries.insert(e, (x, tick));
        while st.entries.len() > self.capacity {
            let victim = st
                .entries
                .iter()
                .filter(|(k, _)| **k != e && !protect.contains(k))
                .min_by_key(|(_, v)| v.1)
                .map(|(k, _)| *k);
            match victim {
                Some(v) => {
                    st.entries.remove(&v);
                }
                None => break,
            }
        }
    }

    /// Get an expert for compute. Cold ⇒ blocking stage on the caller.
    pub fn acquire(&self, e: usize) -> Arc<Expert> {
        let mut st = self.hot.lock().unwrap();
        loop {
            if st.entries.contains_key(&e) {
                st.tick += 1;
                let t = st.tick;
                let entry = st.entries.get_mut(&e).unwrap();
                entry.1 = t;
                self.stats.hits.fetch_add(1, Ordering::Relaxed);
                return entry.0.clone();
            }
            if st.inflight.contains(&e) {
                // Pre-staging already running: wait for it (partially hidden).
                self.stats.waited_inflight.fetch_add(1, Ordering::Relaxed);
                st = crate::tensor::pool::count_wall(|| self.cv.wait(st).unwrap());
                continue;
            }
            break;
        }
        st.inflight.insert(e);
        drop(st);
        let t0 = Instant::now();
        let x = Arc::new(self.cold[e].stage());
        self.stats
            .blocking_ns
            .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        self.stats.blocking_misses.fetch_add(1, Ordering::Relaxed);
        let mut st = self.hot.lock().unwrap();
        st.inflight.remove(&e);
        self.insert(&mut st, e, x.clone(), &[]);
        self.cv.notify_all();
        x
    }

    /// Stage experts ahead of need (called from the stager thread).
    pub fn prestage(&self, wanted: &[usize]) {
        for &e in wanted {
            {
                let mut st = self.hot.lock().unwrap();
                if st.entries.contains_key(&e) {
                    // refresh recency so it is not evicted before use
                    st.tick += 1;
                    let t = st.tick;
                    st.entries.get_mut(&e).unwrap().1 = t;
                    continue;
                }
                if st.inflight.contains(&e) {
                    continue;
                }
                st.inflight.insert(e);
            }
            let t0 = Instant::now();
            let x = Arc::new(self.cold[e].stage());
            self.stats
                .prestage_ns
                .fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
            self.stats.prestaged.fetch_add(1, Ordering::Relaxed);
            let mut st = self.hot.lock().unwrap();
            st.inflight.remove(&e);
            self.insert(&mut st, e, x, wanted);
            self.cv.notify_all();
        }
    }
}

/// Background pre-staging worker shared by a shard's expert stores.
pub struct Stager {
    tx: Sender<(usize, Vec<usize>)>,
}

impl Stager {
    pub fn spawn(stores: HashMap<usize, Arc<ExpertStore>>) -> Self {
        let (tx, rx) = channel::<(usize, Vec<usize>)>();
        std::thread::Builder::new()
            .name("tess-stager".into())
            .spawn(move || {
                while let Ok((layer, experts)) = rx.recv() {
                    if let Some(s) = stores.get(&layer) {
                        s.prestage(&experts);
                    }
                }
            })
            .expect("spawn stager");
        Stager { tx }
    }

    pub fn request(&self, layer: usize, experts: Vec<usize>) {
        let _ = self.tx.send((layer, experts));
    }
}

/// Multi-layer lookahead expert predictor.
#[derive(Clone, Debug)]
pub struct LookaheadPredictor {
    pub source_layer: usize,
    pub target_layer: usize,
    pub w: Linear,
    pub norm_w: Vec<f32>,
    pub top_k: usize,
    /// Extra experts pre-staged beyond top_k (confidence margin).
    pub margin: usize,
    pub lr: f32,
    pub predicted: u64,
    pub needed: u64,
    pub covered: u64,
    pub updates: u64,
}

impl LookaheadPredictor {
    /// Warm start: W_look = router_{L+ΔL}, applied to rmsnorm(h_L)·ffn_norm_{L+ΔL}.
    pub fn from_router(
        source_layer: usize,
        target_layer: usize,
        router: &Linear,
        target_ffn_norm: &[f32],
        top_k: usize,
        margin: usize,
    ) -> Self {
        LookaheadPredictor {
            source_layer,
            target_layer,
            w: Linear::new(router.dense().into_owned()),
            norm_w: target_ffn_norm.to_vec(),
            top_k,
            margin,
            lr: 0.05,
            predicted: 0,
            needed: 0,
            covered: 0,
            updates: 0,
        }
    }

    fn features(&self, h: &[f32]) -> Vec<f32> {
        let mut x = vec![0.0; h.len()];
        rms_norm(h, &self.norm_w, 1e-5, &mut x);
        x
    }

    fn scores(&self, h: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let x = self.features(h);
        let mut s = vec![0.0; self.w.out_features()];
        self.w.forward_vec(&x, &mut s);
        (x, s)
    }

    /// Union over batch rows of each row's top-(k+margin) predicted experts.
    pub fn predict(&mut self, h: &Tensor2D) -> Vec<usize> {
        let mut set = HashSet::new();
        for i in 0..h.rows {
            let (_, s) = self.scores(h.row(i));
            let mut idx: Vec<usize> = (0..s.len()).collect();
            idx.sort_by(|&a, &b| s[b].partial_cmp(&s[a]).unwrap());
            for &e in idx.iter().take(self.top_k + self.margin) {
                set.insert(e);
            }
        }
        let mut v: Vec<usize> = set.into_iter().collect();
        v.sort();
        self.predicted += v.len() as u64;
        v
    }

    /// Score a prediction against the experts actually used, and calibrate
    /// online on every row whose actual experts were not all predicted.
    pub fn calibrate(&mut self, h: &Tensor2D, predicted: &[usize], actual_per_row: &[Vec<usize>]) {
        let actual: HashSet<usize> = actual_per_row.iter().flatten().copied().collect();
        self.needed += actual.len() as u64;
        self.covered += actual.iter().filter(|e| predicted.contains(e)).count() as u64;
        for (i, act) in actual_per_row.iter().enumerate() {
            if i >= h.rows {
                break;
            }
            let (x, mut p) = self.scores(h.row(i));
            let mut order: Vec<usize> = (0..p.len()).collect();
            order.sort_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap());
            let row_top: HashSet<usize> = order.iter().take(self.top_k).copied().collect();
            if act.iter().all(|e| row_top.contains(e)) {
                continue;
            }
            // rank-1 update: W ← W − lr · (softmax(s) − y) xᵀ
            softmax_inplace(&mut p);
            let inv_k = 1.0 / act.len().max(1) as f32;
            for e in 0..p.len() {
                let g = p[e] - if act.contains(&e) { inv_k } else { 0.0 };
                let row = self.w.w.row_mut(e);
                for (wv, xv) in row.iter_mut().zip(&x) {
                    *wv -= self.lr * g * xv;
                }
            }
            self.updates += 1;
        }
    }

    pub fn recall(&self) -> f64 {
        if self.needed == 0 {
            1.0
        } else {
            self.covered as f64 / self.needed as f64
        }
    }
}

/// `PreWakeSignal` payload: target layer + expert ids.
pub fn encode_prewake(target_layer: usize, experts: &[usize]) -> Vec<u8> {
    let mut b = Vec::with_capacity(4 + experts.len() * 2);
    b.extend_from_slice(&(target_layer as u16).to_le_bytes());
    b.extend_from_slice(&(experts.len() as u16).to_le_bytes());
    for &e in experts {
        b.extend_from_slice(&(e as u16).to_le_bytes());
    }
    b
}

pub fn decode_prewake(b: &[u8]) -> (usize, Vec<usize>) {
    let layer = u16::from_le_bytes([b[0], b[1]]) as usize;
    let n = u16::from_le_bytes([b[2], b[3]]) as usize;
    (
        layer,
        (0..n)
            .map(|i| u16::from_le_bytes([b[4 + 2 * i], b[5 + 2 * i]]) as usize)
            .collect(),
    )
}
