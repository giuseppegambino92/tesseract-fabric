//! Composable logit sampler: repetition penalty → temperature → top-k →
//! top-p (nucleus) → min-p, or greedy when `temperature == 0`.
//!
//! [`SamplingParams::probs`] returns the *filtered target distribution* over
//! the whole vocabulary. Speculative verification (`verify_tree`) samples from
//! exactly that distribution, so every filter stays lossless under
//! speculation: the emitted tokens are distributed as if each were drawn
//! from `probs` one step at a time. The repetition penalty of a tree node
//! uses that node's own history (committed tokens + its tree path).

use crate::tensor::{softmax_inplace, Rng};

#[derive(Clone, Debug, PartialEq)]
pub struct SamplingParams {
    pub temperature: f32,
    /// Keep the k most likely tokens (0 = off).
    pub top_k: usize,
    /// Keep the smallest set whose mass reaches p (1 = off).
    pub top_p: f32,
    /// Drop tokens below `min_p · p_max` (0 = off).
    pub min_p: f32,
    /// > 1 discourages tokens seen in the last `repeat_last_n` (1 = off).
    pub repetition_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: u64,
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            repeat_last_n: 64,
            seed: 0,
        }
    }
}

impl SamplingParams {
    pub fn greedy() -> Self {
        Self::default()
    }

    pub fn is_greedy(&self) -> bool {
        self.temperature <= 0.0
    }

    /// Penalised logits (llama.cpp convention: positive logits are divided by
    /// the penalty, negative ones multiplied).
    pub fn penalise(&self, logits: &mut [f32], history: &[u32]) {
        if self.repetition_penalty == 1.0 || self.repeat_last_n == 0 {
            return;
        }
        let start = history.len().saturating_sub(self.repeat_last_n);
        let mut seen = std::collections::HashSet::new();
        for &t in &history[start..] {
            if (t as usize) < logits.len() && seen.insert(t) {
                let l = &mut logits[t as usize];
                *l = if *l > 0.0 {
                    *l / self.repetition_penalty
                } else {
                    *l * self.repetition_penalty
                };
            }
        }
    }

    /// The filtered next-token distribution for `logits` after `history`.
    pub fn probs(&self, logits: &[f32], history: &[u32]) -> Vec<f32> {
        let mut l = logits.to_vec();
        self.penalise(&mut l, history);
        let n = l.len();
        if self.is_greedy() {
            let best = crate::model::argmax(&l);
            let mut p = vec![0.0; n];
            p[best] = 1.0;
            return p;
        }
        for v in l.iter_mut() {
            *v /= self.temperature;
        }
        softmax_inplace(&mut l);
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap().then(a.cmp(&b)));
        let mut keep = n;
        if self.top_k > 0 {
            keep = keep.min(self.top_k);
        }
        if self.top_p < 1.0 {
            let mut c = 0.0f32;
            for (r, &i) in order.iter().enumerate().take(keep) {
                c += l[i];
                if c >= self.top_p {
                    keep = r + 1;
                    break;
                }
            }
        }
        if self.min_p > 0.0 {
            let floor = self.min_p * l[order[0]];
            keep = order[..keep]
                .iter()
                .take_while(|&&i| l[i] >= floor)
                .count()
                .max(1);
        }
        let mut p = vec![0.0f32; n];
        let mut s = 0.0f32;
        for &i in &order[..keep] {
            p[i] = l[i];
            s += l[i];
        }
        for v in p.iter_mut() {
            *v /= s;
        }
        p
    }

    /// Draw one token from `probs` (exact inverse-CDF sampling).
    pub fn sample(&self, probs: &[f32], rng: &mut Rng) -> u32 {
        crate::model::sample(probs, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filters_compose_and_normalise() {
        let logits = [2.0f32, 1.0, 0.5, -1.0, 3.0];
        let p = SamplingParams {
            temperature: 1.0,
            top_k: 3,
            ..Default::default()
        }
        .probs(&logits, &[]);
        assert_eq!(p.iter().filter(|v| **v > 0.0).count(), 3);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        let g = SamplingParams::greedy().probs(&logits, &[]);
        assert_eq!(g[4], 1.0);
        // repetition penalty flips the greedy choice
        let r = SamplingParams {
            repetition_penalty: 2.0,
            ..Default::default()
        }
        .probs(&logits, &[4]);
        assert_eq!(r[0], 1.0);
        let m = SamplingParams {
            temperature: 1.0,
            min_p: 0.3,
            ..Default::default()
        }
        .probs(&logits, &[]);
        assert_eq!(m.iter().filter(|v| **v > 0.0).count(), 2);
    }
}
