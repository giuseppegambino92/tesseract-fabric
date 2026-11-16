//! Adaptive multipath link striping.
//!
//! Any node pair may hold several logical links (Thunderbolt, 2.5GbE, Wi-Fi…).
//! The scheduler keeps per-link telemetry (EWMA one-way latency, effective
//! bandwidth, queued bytes) and splits a payload so every used link is
//! predicted to finish at the same instant — the water-filling optimum of
//!   ETTC_k = RTT_k/2 + Bytes_k/BW_k + QueueDelay_k.
//! A link whose observed latency balloons (brownout) or that misses a deadline
//! is penalised/excluded until telemetry recovers.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LinkKind {
    ThunderboltRdma,
    Ethernet10G,
    Ethernet2_5G,
    WiFi6E,
    Loopback,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinkProfile {
    pub kind: LinkKind,
    /// One-way propagation + stack latency in microseconds (RTT/2).
    pub one_way_us: f64,
    pub bandwidth_gbps: f64,
    /// Uniform jitter amplitude in microseconds.
    pub jitter_us: f64,
}

impl LinkProfile {
    /// Thunderbolt 5 RDMA class: 40 µs RTT, 80 Gbps.
    pub fn thunderbolt5() -> Self {
        LinkProfile {
            kind: LinkKind::ThunderboltRdma,
            one_way_us: 20.0,
            bandwidth_gbps: 80.0,
            jitter_us: 2.0,
        }
    }
    /// 2.5 GbE: 250 µs RTT.
    pub fn ethernet_2_5g() -> Self {
        LinkProfile {
            kind: LinkKind::Ethernet2_5G,
            one_way_us: 125.0,
            bandwidth_gbps: 2.5,
            jitter_us: 15.0,
        }
    }
    pub fn ethernet_10g() -> Self {
        LinkProfile {
            kind: LinkKind::Ethernet10G,
            one_way_us: 60.0,
            bandwidth_gbps: 10.0,
            jitter_us: 8.0,
        }
    }
    /// Wi-Fi 6E: 2.2 ms RTT, ~0.8 Gbps effective.
    pub fn wifi6e() -> Self {
        LinkProfile {
            kind: LinkKind::WiFi6E,
            one_way_us: 1100.0,
            bandwidth_gbps: 0.8,
            jitter_us: 250.0,
        }
    }
    /// Unshaped (real network / loopback as-is).
    pub fn unshaped() -> Self {
        LinkProfile {
            kind: LinkKind::Loopback,
            one_way_us: 0.0,
            bandwidth_gbps: f64::INFINITY,
            jitter_us: 0.0,
        }
    }

    pub fn is_unshaped(&self) -> bool {
        self.kind == LinkKind::Loopback
    }

    /// Serialisation time for `bytes` in nanoseconds.
    pub fn serialization_ns(&self, bytes: usize) -> f64 {
        if self.bandwidth_gbps.is_infinite() {
            0.0
        } else {
            bytes as f64 * 8.0 / self.bandwidth_gbps
        }
    }

    /// Unloaded one-way delivery estimate (ns).
    pub fn ettc_ns(&self, bytes: usize) -> f64 {
        self.one_way_us * 1000.0 + self.serialization_ns(bytes)
    }

    pub fn name(&self) -> &'static str {
        match self.kind {
            LinkKind::ThunderboltRdma => "TB5-RDMA",
            LinkKind::Ethernet10G => "10GbE",
            LinkKind::Ethernet2_5G => "2.5GbE",
            LinkKind::WiFi6E => "WiFi6E",
            LinkKind::Loopback => "loopback",
        }
    }

    pub fn encode(&self) -> [u8; 25] {
        let mut b = [0u8; 25];
        b[0] = self.kind as u8;
        b[1..9].copy_from_slice(&self.one_way_us.to_le_bytes());
        b[9..17].copy_from_slice(&self.bandwidth_gbps.to_le_bytes());
        b[17..25].copy_from_slice(&self.jitter_us.to_le_bytes());
        b
    }

    pub fn decode(b: &[u8]) -> Self {
        let kind = match b[0] {
            0 => LinkKind::ThunderboltRdma,
            1 => LinkKind::Ethernet10G,
            2 => LinkKind::Ethernet2_5G,
            3 => LinkKind::WiFi6E,
            _ => LinkKind::Loopback,
        };
        let f = |o: usize| f64::from_le_bytes(b[o..o + 8].try_into().unwrap());
        LinkProfile {
            kind,
            one_way_us: f(1),
            bandwidth_gbps: f(9),
            jitter_us: f(17),
        }
    }
}

#[derive(Clone, Debug)]
pub struct LinkTelemetry {
    pub nominal: LinkProfile,
    /// EWMA one-way latency estimate (ns), latency-only component.
    pub est_one_way_ns: f64,
    /// EWMA effective bandwidth (bytes/ns).
    pub est_bw_bytes_per_ns: f64,
    /// Bytes handed to this link whose delivery is still predicted pending.
    pub queued_until_ns: f64,
    pub healthy: bool,
    pub samples: u64,
    pub bytes_sent: u64,
    /// Last latency probe (heartbeat) time on this link.
    pub last_probe_ns: f64,
}

impl LinkTelemetry {
    fn new(p: LinkProfile) -> Self {
        LinkTelemetry {
            nominal: p,
            est_one_way_ns: p.one_way_us * 1000.0,
            est_bw_bytes_per_ns: if p.bandwidth_gbps.is_infinite() {
                1e9
            } else {
                p.bandwidth_gbps / 8.0
            },
            queued_until_ns: 0.0,
            healthy: true,
            samples: 0,
            bytes_sent: 0,
            last_probe_ns: 0.0,
        }
    }

    /// Estimated time-to-completion (ns from `now`) for `bytes`.
    pub fn ettc(&self, now: f64, bytes: usize) -> f64 {
        let queue = (self.queued_until_ns - now).max(0.0);
        self.est_one_way_ns + bytes as f64 / self.est_bw_bytes_per_ns + queue
    }
}

#[derive(Clone, Debug)]
pub struct MultipathScheduler {
    pub links: Vec<LinkTelemetry>,
    pub min_stripe: usize,
    pub ewma: f64,
}

impl MultipathScheduler {
    pub fn new(profiles: &[LinkProfile]) -> Self {
        MultipathScheduler {
            links: profiles.iter().map(|p| LinkTelemetry::new(*p)).collect(),
            min_stripe: 8 * 1024,
            ewma: 0.25,
        }
    }

    pub fn n_links(&self) -> usize {
        self.links.len()
    }

    /// Pick the single link with minimal ETTC.
    pub fn best_link(&self, now: f64, bytes: usize) -> usize {
        let mut best = 0;
        let mut bt = f64::INFINITY;
        for (i, l) in self.links.iter().enumerate() {
            let t = l.ettc(now, bytes) * if l.healthy { 1.0 } else { 50.0 };
            if t < bt {
                bt = t;
                best = i;
            }
        }
        best
    }

    /// Water-filling split of `bytes` across healthy links so all used links
    /// finish together. Returns (link, byte count) with counts summing to bytes.
    pub fn plan(&self, now: f64, bytes: usize) -> Vec<(usize, usize)> {
        let mut active: Vec<usize> = (0..self.links.len())
            .filter(|&i| self.links[i].healthy)
            .collect();
        if active.is_empty() {
            active = (0..self.links.len()).collect();
        }
        loop {
            // a_k = fixed delay, r_k = rate. Solve Σ (T − a_k)·r_k = B.
            let mut items: Vec<(f64, f64, usize)> = active
                .iter()
                .map(|&i| {
                    let l = &self.links[i];
                    (
                        l.est_one_way_ns + (l.queued_until_ns - now).max(0.0),
                        l.est_bw_bytes_per_ns,
                        i,
                    )
                })
                .collect();
            items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let mut used = 0;
            let mut t = 0.0;
            let mut sum_r = 0.0;
            let mut sum_ar = 0.0;
            for (k, &(a, r, _)) in items.iter().enumerate() {
                sum_r += r;
                sum_ar += a * r;
                let tk = (bytes as f64 + sum_ar) / sum_r;
                let next_a = items.get(k + 1).map(|x| x.0).unwrap_or(f64::INFINITY);
                used = k + 1;
                t = tk;
                if tk <= next_a {
                    break;
                }
            }
            let mut shares: Vec<(usize, usize)> = items[..used]
                .iter()
                .map(|&(a, r, i)| (i, ((t - a) * r).max(0.0) as usize))
                .collect();
            // drop stripes below the minimum size (header overhead dominates)
            if shares.len() > 1 {
                if let Some(pos) = shares.iter().position(|s| s.1 < self.min_stripe) {
                    let drop = shares[pos].0;
                    active.retain(|&x| x != drop);
                    continue;
                }
            }
            // fix rounding so totals match exactly
            let assigned: usize = shares.iter().map(|s| s.1).sum();
            if let Some(first) = shares.first_mut() {
                first.1 = first.1 + bytes - assigned.min(bytes);
            }
            shares.retain(|s| s.1 > 0);
            if shares.is_empty() {
                shares.push((items[0].2, bytes));
            }
            return shares;
        }
    }

    /// Record that `bytes` were scheduled on `link` (advances queue estimate).
    pub fn on_send(&mut self, link: usize, now: f64, bytes: usize) {
        let l = &mut self.links[link];
        let start = l.queued_until_ns.max(now);
        l.queued_until_ns = start + bytes as f64 / l.est_bw_bytes_per_ns;
        l.bytes_sent += bytes as u64;
    }

    /// Frames at or below this size are latency probes (heartbeats).
    pub const PROBE_BYTES: usize = 256;

    /// Links due for a latency probe (BBR-style: tiny heartbeats measure
    /// propagation latency; large stripes measure bandwidth net of it).
    pub fn probes_due(&mut self, now: f64, interval_ns: f64) -> Vec<usize> {
        let mut v = Vec::new();
        for (i, l) in self.links.iter_mut().enumerate() {
            if now - l.last_probe_ns > interval_ns {
                l.last_probe_ns = now;
                v.push(i);
            }
        }
        v
    }

    /// Telemetry from a delivery acknowledgement: measured one-way time of a
    /// frame of `bytes` sent at `sent_ns` and received at `recv_ns`.
    pub fn on_ack(&mut self, link: usize, bytes: usize, sent_ns: f64, recv_ns: f64) {
        let a = self.ewma;
        let l = &mut self.links[link];
        let measured = (recv_ns - sent_ns).max(1.0);
        if bytes <= Self::PROBE_BYTES {
            l.est_one_way_ns = (1.0 - a) * l.est_one_way_ns + a * measured;
        } else {
            let nominal_bw = if l.nominal.bandwidth_gbps.is_infinite() {
                1e9
            } else {
                l.nominal.bandwidth_gbps / 8.0
            };
            let bw_sample = bytes as f64 / (measured - l.est_one_way_ns).max(1_000.0);
            l.est_bw_bytes_per_ns = ((1.0 - a) * l.est_bw_bytes_per_ns + a * bw_sample)
                .clamp(nominal_bw / 100.0, nominal_bw * 1.5);
        }
        l.samples += 1;
        let nominal = l.nominal.one_way_us * 1000.0;
        l.healthy = l.est_one_way_ns < (8.0 * nominal).max(nominal + 2_000_000.0);
    }

    /// Deadline miss (straggler): sharply penalise the link.
    pub fn on_timeout(&mut self, link: usize) {
        let l = &mut self.links[link];
        l.est_one_way_ns *= 4.0;
        l.est_bw_bytes_per_ns /= 4.0;
        l.healthy = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn waterfill_equalises_finish_times() {
        let s = MultipathScheduler::new(&[LinkProfile::ethernet_2_5g(), LinkProfile::wifi6e()]);
        let plan = s.plan(0.0, 4 << 20);
        assert_eq!(plan.iter().map(|p| p.1).sum::<usize>(), 4 << 20);
        assert_eq!(plan.len(), 2);
        let t: Vec<f64> = plan.iter().map(|&(i, b)| s.links[i].ettc(0.0, b)).collect();
        assert!((t[0] - t[1]).abs() / t[0] < 0.01, "{t:?}");
    }

    #[test]
    fn small_payload_uses_fastest_link_only() {
        let s = MultipathScheduler::new(&[LinkProfile::wifi6e(), LinkProfile::thunderbolt5()]);
        let plan = s.plan(0.0, 2048);
        assert_eq!(plan, vec![(1, 2048)]);
    }
}
