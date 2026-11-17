//! Single-core throughput of the wire crypto (ChaCha20, Poly1305, the AEAD
//! and the record layer), plus the X25519 handshake cost.
//!
//! cargo run --release --example crypto_bench
//! TESSERACT_WIRE_THREADS=1 cargo run --release --example crypto_bench   # one core
use std::io::{Read, Write};
use std::time::Instant;
use tesseract_fabric::wire::auth::{crypto_threads, OpenedReader, SealedWriter};
use tesseract_fabric::wire::crypto::{chacha20_xor, open, poly1305, seal, x25519, X25519_BASE};

fn rate(label: &str, bytes: usize, f: &mut dyn FnMut()) {
    f();
    let mut best = f64::MAX;
    for _ in 0..7 {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
    }
    println!("{label:<28} {:>7.0} MB/s", bytes as f64 / best / 1e6);
}

fn main() {
    let n = 8 << 20;
    let mut buf: Vec<u8> = (0..n).map(|i| i as u8).collect();
    let key = [7u8; 32];
    let nonce = [1u8; 12];
    rate("chacha20", n, &mut || {
        chacha20_xor(&key, 1, &nonce, &mut buf)
    });
    rate("poly1305", n, &mut || {
        std::hint::black_box(poly1305(&key, &buf));
    });
    rate("aead seal", n, &mut || {
        std::hint::black_box(seal(&key, &nonce, b"aad", &mut buf));
    });
    let tag = seal(&key, &nonce, b"aad", &mut buf);
    rate("aead open (verify+decrypt)", n, &mut || {
        assert!(open(&key, &nonce, b"aad", &mut buf, &tag));
        seal(&key, &nonce, b"aad", &mut buf);
    });
    let mut wire = Vec::with_capacity(n + 1024);
    println!(
        "record layer: 1 MiB records, {} thread(s)",
        crypto_threads()
    );
    rate("record layer write", n, &mut || {
        wire.clear();
        SealedWriter::new(&mut wire, key).write_all(&buf).unwrap();
    });
    let mut out = Vec::with_capacity(n);
    rate("record layer read", n, &mut || {
        out.clear();
        OpenedReader::new(&wire[..], key)
            .read_to_end(&mut out)
            .unwrap();
    });
    for size in [64usize, 4096] {
        let mut small = vec![3u8; size];
        let iters = 20_000;
        let t = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(seal(&key, &nonce, b"aad", &mut small));
        }
        println!(
            "seal {size:>5} B record          {:>7.2} µs",
            t.elapsed().as_secs_f64() * 1e6 / iters as f64
        );
    }
    let iters = 200;
    let t = Instant::now();
    let mut k = [9u8; 32];
    for _ in 0..iters {
        k = x25519(&k, &X25519_BASE);
    }
    std::hint::black_box(k);
    println!(
        "x25519 scalar multiplication {:>7.1} µs",
        t.elapsed().as_secs_f64() * 1e6 / iters as f64
    );
}
