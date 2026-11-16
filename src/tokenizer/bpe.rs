//! Byte-level BPE and SentencePiece-style tokenizers.
//!
//! * **GPT-2 byte-level BPE** (`tokenizer.ggml.model = "gpt2"`, Hugging Face
//!   `"type": "BPE"`): text is pre-split with the GPT-2 pattern
//!   (contractions, letter runs, digit runs, punctuation runs, whitespace),
//!   each piece's UTF-8 bytes become byte tokens, and adjacent pairs merge in
//!   merge-rank order (lowest rank first, through a priority queue). Every
//!   byte has a token, so any string round-trips exactly.
//! * **SentencePiece** (`"llama"`): spaces become `▁`, adjacent symbols merge
//!   by highest token score, and anything left out of the vocabulary falls
//!   back to `<0xNN>` byte tokens.
//!
//! Decoding concatenates token bytes; [`StreamDecoder`] emits only complete
//! UTF-8 characters so multi-byte characters split across tokens stream
//! correctly.

use crate::loader::gguf::{GgufFile, GgufValue};
use crate::server::json::Json;
use std::collections::{BinaryHeap, HashMap};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenizerKind {
    Gpt2,
    SentencePiece,
}

/// Pre-tokenizer (llama.cpp `tokenizer.ggml.pre`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreTokenizer {
    /// GPT-2: `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
    Gpt2,
    /// SmolLM: every digit alone, then GPT-2.
    Smollm,
    /// Llama 3 (`llama-bpe`): case-insensitive contractions, one leading
    /// non-letter before letter runs, digits in groups of ≤ 3, punctuation
    /// runs absorbing trailing newlines, newline-terminated whitespace.
    Llama3,
    /// Qwen2 / Qwen3: as Llama 3 with single digits.
    Qwen2,
}

impl PreTokenizer {
    pub fn from_name(s: &str) -> Self {
        match s {
            "smollm" => PreTokenizer::Smollm,
            "llama-bpe" | "llama3" | "llama-v3" => PreTokenizer::Llama3,
            "qwen2" | "qwen3" | "deepseek-r1-qwen" => PreTokenizer::Qwen2,
            _ => PreTokenizer::Gpt2,
        }
    }
    pub fn split<'a>(&self, text: &'a str) -> Vec<&'a str> {
        match self {
            PreTokenizer::Gpt2 => gpt2_pretokenize(text),
            PreTokenizer::Smollm => {
                let mut out = Vec::new();
                let mut start = 0;
                for (i, c) in text.char_indices() {
                    if c.is_numeric() {
                        if start < i {
                            out.extend(gpt2_pretokenize(&text[start..i]));
                        }
                        out.push(&text[i..i + c.len_utf8()]);
                        start = i + c.len_utf8();
                    }
                }
                if start < text.len() {
                    out.extend(gpt2_pretokenize(&text[start..]));
                }
                out
            }
            PreTokenizer::Llama3 => llama3_pretokenize(text, 3),
            PreTokenizer::Qwen2 => llama3_pretokenize(text, 1),
        }
    }
}

/// The Llama 3 / Qwen2 pre-tokenizer pattern, matched by hand:
/// `(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,k}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
pub fn llama3_pretokenize(text: &str, max_digits: usize) -> Vec<&str> {
    let cs: Vec<(usize, char)> = text.char_indices().collect();
    let n = cs.len();
    let at = |i: usize| if i < n { Some(cs[i].1) } else { None };
    let off = |i: usize| if i < n { cs[i].0 } else { text.len() };
    let is_l = |c: char| c.is_alphabetic();
    let is_n = |c: char| c.is_numeric();
    let is_nl = |c: char| c == '\r' || c == '\n';
    let is_other = |c: char| !c.is_whitespace() && !c.is_alphabetic() && !c.is_numeric();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let c = cs[i].1;
        let end = (|| {
            // contractions (case-insensitive)
            if c == '\'' {
                for m in ["s", "t", "re", "ve", "m", "ll", "d"] {
                    let k = m.chars().count();
                    if (0..k).all(|t| {
                        at(i + 1 + t)
                            .is_some_and(|x| x.to_ascii_lowercase() == m.as_bytes()[t] as char)
                    }) {
                        return i + 1 + k;
                    }
                }
            }
            // [^\r\n\p{L}\p{N}]?\p{L}+
            let lead = !is_nl(c) && !is_l(c) && !is_n(c) && at(i + 1).is_some_and(is_l);
            let k = if lead { i + 1 } else { i };
            if at(k).is_some_and(is_l) {
                let mut j = k;
                while at(j).is_some_and(is_l) {
                    j += 1;
                }
                return j;
            }
            // \p{N}{1,k}
            if is_n(c) {
                let mut j = i;
                while j < i + max_digits && at(j).is_some_and(is_n) {
                    j += 1;
                }
                return j;
            }
            // ' ?[^\s\p{L}\p{N}]+[\r\n]*'
            let k = if c == ' ' && at(i + 1).is_some_and(is_other) {
                i + 1
            } else {
                i
            };
            if at(k).is_some_and(is_other) {
                let mut j = k;
                while at(j).is_some_and(is_other) {
                    j += 1;
                }
                while at(j).is_some_and(is_nl) {
                    j += 1;
                }
                return j;
            }
            // whitespace forms
            let mut j = i;
            while at(j).is_some_and(|x| x.is_whitespace()) {
                j += 1;
            }
            // \s*[\r\n]+ : up to the run's last newline
            if let Some(p) = (i..j).rev().find(|&p| is_nl(cs[p].1)) {
                return p + 1;
            }
            // \s+(?!\S) then \s+
            if j < n && j - i > 1 {
                return j - 1;
            }
            j
        })();
        out.push(&text[off(i)..off(end)]);
        i = end;
    }
    out
}

/// GGML token types.
pub const TOKEN_NORMAL: i32 = 1;
pub const TOKEN_UNKNOWN: i32 = 2;
pub const TOKEN_CONTROL: i32 = 3;
pub const TOKEN_USER_DEFINED: i32 = 4;
pub const TOKEN_UNUSED: i32 = 5;
pub const TOKEN_BYTE: i32 = 6;

#[derive(Clone, Debug)]
pub struct Tokenizer {
    pub kind: TokenizerKind,
    /// Token strings as stored in the model file.
    pub tokens: Vec<String>,
    pub token_type: Vec<i32>,
    pub scores: Vec<f32>,
    /// Merge rules "left right" (GPT-2), in rank order.
    pub merges: Vec<(String, String)>,
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    pub pre: PreTokenizer,
    /// Prepend BOS when encoding a prompt (`tokenizer.ggml.add_bos_token`).
    pub add_bos: bool,
    /// Chat template text from the model file (`tokenizer.chat_template`).
    pub chat_template: Option<String>,
    /// Raw bytes each token decodes to (control tokens decode to nothing).
    bytes: Vec<Vec<u8>>,
    vocab: HashMap<String, u32>,
    merge_rank: HashMap<(u32, u32), (u32, u32)>,
    /// GPT-2: byte → token id of its single-byte token. SPM: byte fallback ids.
    byte_token: [u32; 256],
}

#[derive(Debug)]
pub struct TokenizerError(pub String);
impl std::fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "tokenizer: {}", self.0)
    }
}
impl std::error::Error for TokenizerError {}

/// GPT-2's reversible byte ↔ printable-unicode map.
pub fn bytes_to_unicode() -> [char; 256] {
    let mut map = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        let printable =
            (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
        map[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            n += 1;
            char::from_u32(255 + n).unwrap()
        };
    }
    map
}

fn unicode_to_bytes() -> HashMap<char, u8> {
    bytes_to_unicode()
        .iter()
        .enumerate()
        .map(|(b, c)| (*c, b as u8))
        .collect()
}

/// Split text into GPT-2 pre-tokens (byte ranges).
pub fn gpt2_pretokenize(text: &str) -> Vec<&str> {
    let cs: Vec<(usize, char)> = text.char_indices().collect();
    let n = cs.len();
    let at = |i: usize| if i < n { Some(cs[i].1) } else { None };
    let off = |i: usize| if i < n { cs[i].0 } else { text.len() };
    let is_l = |c: char| c.is_alphabetic();
    let is_n = |c: char| c.is_numeric();
    let is_other = |c: char| !c.is_whitespace() && !c.is_alphabetic() && !c.is_numeric();
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let c = cs[i].1;
        // contractions
        if c == '\'' {
            let rest = &text[off(i + 1)..];
            let m = ["s", "t", "re", "ve", "m", "ll", "d"]
                .iter()
                .find(|s| rest.starts_with(**s));
            if let Some(m) = m {
                let j = i + 1 + m.chars().count();
                out.push(&text[off(i)..off(j)]);
                i = j;
                continue;
            }
        }
        let start = i;
        let lead = c == ' ' && at(i + 1).is_some_and(|d| !d.is_whitespace());
        let k = if lead { i + 1 } else { i };
        let first = at(k).unwrap();
        let class: Option<&dyn Fn(char) -> bool> = if is_l(first) {
            Some(&is_l)
        } else if is_n(first) {
            Some(&is_n)
        } else if is_other(first) {
            Some(&is_other)
        } else {
            None
        };
        if let Some(cl) = class {
            let mut j = k;
            while at(j).is_some_and(cl) {
                j += 1;
            }
            out.push(&text[off(start)..off(j)]);
            i = j;
            continue;
        }
        // whitespace run: \s+(?!\S) leaves the run's last char when a
        // non-space follows (it then leads the next piece, or stands alone)
        let mut j = i;
        while at(j).is_some_and(|d| d.is_whitespace()) {
            j += 1;
        }
        if j < n && j - i > 1 {
            j -= 1;
        }
        out.push(&text[off(i)..off(j)]);
        i = j;
    }
    out
}

impl Tokenizer {
    /// Build from GGUF-style arrays.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        kind: TokenizerKind,
        tokens: Vec<String>,
        token_type: Vec<i32>,
        scores: Vec<f32>,
        merges: Vec<(String, String)>,
        bos: Option<u32>,
        eos: Option<u32>,
    ) -> Result<Self, TokenizerError> {
        let n = tokens.len();
        let token_type = if token_type.len() == n {
            token_type
        } else {
            vec![TOKEN_NORMAL; n]
        };
        let scores = if scores.len() == n {
            scores
        } else {
            vec![0.0; n]
        };
        let vocab: HashMap<String, u32> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i as u32))
            .collect();
        let u2b = unicode_to_bytes();
        let mut byte_token = [u32::MAX; 256];
        let mut bytes = Vec::with_capacity(n);
        for (i, t) in tokens.iter().enumerate() {
            let ty = token_type[i];
            let b: Vec<u8> = if ty == TOKEN_CONTROL || ty == TOKEN_UNUSED {
                Vec::new()
            } else {
                match kind {
                    TokenizerKind::Gpt2 => t.chars().filter_map(|c| u2b.get(&c).copied()).collect(),
                    TokenizerKind::SentencePiece => {
                        if ty == TOKEN_BYTE {
                            let v = t
                                .strip_prefix("<0x")
                                .and_then(|h| h.strip_suffix('>'))
                                .and_then(|h| u8::from_str_radix(h, 16).ok())
                                .ok_or_else(|| TokenizerError(format!("bad byte token {t}")))?;
                            byte_token[v as usize] = i as u32;
                            vec![v]
                        } else {
                            t.replace('\u{2581}', " ").into_bytes()
                        }
                    }
                }
            };
            if kind == TokenizerKind::Gpt2
                && b.len() == 1
                && ty == TOKEN_NORMAL
                && byte_token[b[0] as usize] == u32::MAX
            {
                byte_token[b[0] as usize] = i as u32;
            }
            bytes.push(b);
        }
        // a vocabulary without some raw byte (real checkpoints drop unused
        // control bytes) maps it to the unknown token, as llama.cpp does
        let unk = token_type
            .iter()
            .position(|&t| t == TOKEN_UNKNOWN)
            .unwrap_or(0) as u32;
        for b in byte_token.iter_mut() {
            if *b == u32::MAX {
                *b = unk;
            }
        }
        let mut merge_rank = HashMap::new();
        for (r, (a, b)) in merges.iter().enumerate() {
            if let (Some(&ia), Some(&ib), Some(&im)) =
                (vocab.get(a), vocab.get(b), vocab.get(&format!("{a}{b}")))
            {
                merge_rank.entry((ia, ib)).or_insert((r as u32, im));
            }
        }
        Ok(Tokenizer {
            kind,
            tokens,
            token_type,
            scores,
            merges,
            bos,
            eos,
            pre: PreTokenizer::Gpt2,
            add_bos: kind == TokenizerKind::SentencePiece,
            chat_template: None,
            bytes,
            vocab,
            merge_rank,
            byte_token,
        })
    }

    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    pub fn token_id(&self, s: &str) -> Option<u32> {
        self.vocab.get(s).copied()
    }

    /// Raw bytes of token `t`.
    pub fn token_bytes(&self, t: u32) -> &[u8] {
        self.bytes
            .get(t as usize)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Encode text to token ids (no BOS added).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        match self.kind {
            TokenizerKind::Gpt2 => {
                let mut out = Vec::new();
                for piece in self.pre.split(text) {
                    let syms: Vec<u32> =
                        piece.bytes().map(|b| self.byte_token[b as usize]).collect();
                    out.extend(self.merge_by_rank(syms));
                }
                out
            }
            TokenizerKind::SentencePiece => self.encode_spm(text),
        }
    }

    /// Lowest-rank-first pair merging with a heap over a linked symbol list.
    fn merge_by_rank(&self, syms: Vec<u32>) -> Vec<u32> {
        let n = syms.len();
        if n < 2 {
            return syms;
        }
        let mut tok = syms;
        let mut next: Vec<usize> = (1..=n).collect();
        let mut prev: Vec<usize> = (0..n).map(|i| i.wrapping_sub(1)).collect();
        let mut alive = vec![true; n];
        // heap of (−rank, −position, left, right token ids at push time)
        let mut heap = BinaryHeap::new();
        let push =
            |heap: &mut BinaryHeap<(i64, i64, u32, u32)>, tok: &[u32], i: usize, j: usize| {
                if let Some(&(r, _)) = self.merge_rank.get(&(tok[i], tok[j])) {
                    heap.push((-(r as i64), -(i as i64), tok[i], tok[j]));
                }
            };
        for i in 0..n - 1 {
            push(&mut heap, &tok, i, i + 1);
        }
        while let Some((_, ni, a, b)) = heap.pop() {
            let i = (-ni) as usize;
            if !alive[i] {
                continue;
            }
            let j = next[i];
            if j >= n || !alive[j] || tok[i] != a || tok[j] != b {
                continue; // stale entry
            }
            tok[i] = self.merge_rank[&(a, b)].1;
            alive[j] = false;
            next[i] = next[j];
            if next[j] < n {
                prev[next[j]] = i;
            }
            if prev[i] != usize::MAX {
                push(&mut heap, &tok, prev[i], i);
            }
            if next[i] < n {
                push(&mut heap, &tok, i, next[i]);
            }
        }
        (0..n).filter(|&i| alive[i]).map(|i| tok[i]).collect()
    }

    fn encode_spm(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        let norm = format!("\u{2581}{}", text.replace(' ', "\u{2581}"));
        let mut syms: Vec<String> = norm.chars().map(|c| c.to_string()).collect();
        // greedily merge the adjacent pair whose concatenation scores highest
        loop {
            let mut best: Option<(f32, usize)> = None;
            for i in 0..syms.len().saturating_sub(1) {
                let cat = format!("{}{}", syms[i], syms[i + 1]);
                if let Some(&id) = self.vocab.get(&cat) {
                    let s = self.scores[id as usize];
                    if best.is_none_or(|(bs, _)| s > bs) {
                        best = Some((s, i));
                    }
                }
            }
            match best {
                Some((_, i)) => {
                    let r = syms.remove(i + 1);
                    syms[i].push_str(&r);
                }
                None => break,
            }
        }
        let mut out = Vec::new();
        for s in syms {
            match self.vocab.get(&s) {
                Some(&id) if self.token_type[id as usize] != TOKEN_BYTE => out.push(id),
                _ => {
                    let unk = self
                        .token_type
                        .iter()
                        .position(|&t| t == TOKEN_UNKNOWN)
                        .unwrap_or(0) as u32;
                    out.extend(s.bytes().map(|b| match self.byte_token[b as usize] {
                        u32::MAX => unk,
                        t => t,
                    }))
                }
            }
        }
        out
    }

    /// Encode text in which special tokens (control / user-defined, e.g.
    /// `<|im_start|>`) appear literally: those map to their ids, everything
    /// between them is encoded normally (llama.cpp `parse_special`).
    pub fn encode_special(&self, text: &str) -> Vec<u32> {
        let specials: Vec<(&str, u32)> = {
            let mut v: Vec<(&str, u32)> = self
                .tokens
                .iter()
                .enumerate()
                .filter(|(i, t)| {
                    matches!(self.token_type[*i], TOKEN_CONTROL | TOKEN_USER_DEFINED)
                        && !t.is_empty()
                })
                .map(|(i, t)| (t.as_str(), i as u32))
                .collect();
            v.sort_by_key(|(t, _)| std::cmp::Reverse(t.len()));
            v
        };
        let mut out = Vec::new();
        let (mut start, mut i) = (0, 0);
        let b = text.as_bytes();
        while i < b.len() {
            let hit = if b[i] == b'<' || b[i] == b'[' {
                specials.iter().find(|(t, _)| text[i..].starts_with(t))
            } else {
                None
            };
            match hit {
                Some(&(t, id)) => {
                    out.extend(self.encode(&text[start..i]));
                    out.push(id);
                    i += t.len();
                    start = i;
                }
                None => i += 1,
            }
        }
        out.extend(self.encode(&text[start..]));
        out
    }

    /// Id of a token string, if present.
    pub fn special(&self, s: &str) -> Option<u32> {
        self.token_id(s)
    }

    /// Decode to raw bytes (control tokens dropped).
    pub fn decode_bytes(&self, toks: &[u32]) -> Vec<u8> {
        let mut out: Vec<u8> = toks
            .iter()
            .flat_map(|&t| self.token_bytes(t).iter().copied())
            .collect();
        if self.kind == TokenizerKind::SentencePiece && out.first() == Some(&b' ') {
            out.remove(0);
        }
        out
    }

    /// Decode to a string (invalid UTF-8 becomes U+FFFD).
    pub fn decode(&self, toks: &[u32]) -> String {
        String::from_utf8_lossy(&self.decode_bytes(toks)).into_owned()
    }

    /// A streaming decoder for this tokenizer.
    pub fn stream(&self) -> StreamDecoder<'_> {
        StreamDecoder {
            tok: self,
            pending: Vec::new(),
            started: false,
        }
    }

    // --- model-file I/O ------------------------------------------------------

    /// Read `tokenizer.ggml.*` from a GGUF file.
    pub fn from_gguf(g: &GgufFile) -> Result<Self, TokenizerError> {
        let model = g.get_str("tokenizer.ggml.model").unwrap_or("gpt2");
        let kind = match model {
            "gpt2" => TokenizerKind::Gpt2,
            "llama" => TokenizerKind::SentencePiece,
            m => return Err(TokenizerError(format!("unsupported tokenizer model {m}"))),
        };
        let strs = |k: &str| -> Vec<String> {
            g.get(k)
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_str().map(String::from))
                        .collect()
                })
                .unwrap_or_default()
        };
        let tokens = strs("tokenizer.ggml.tokens");
        if tokens.is_empty() {
            return Err(TokenizerError("GGUF has no tokenizer.ggml.tokens".into()));
        }
        let token_type = g
            .get("tokenizer.ggml.token_type")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(1.0) as i32).collect())
            .unwrap_or_default();
        let scores = g
            .get("tokenizer.ggml.scores")
            .and_then(|v| v.as_array())
            .map(|a| a.iter().map(|x| x.as_f64().unwrap_or(0.0) as f32).collect())
            .unwrap_or_default();
        let merges = strs("tokenizer.ggml.merges")
            .into_iter()
            .filter_map(|m| {
                m.split_once(' ')
                    .map(|(a, b)| (a.to_string(), b.to_string()))
            })
            .collect();
        let id = |k: &str| g.get_u64(k).map(|v| v as u32);
        let mut t = Self::new(
            kind,
            tokens,
            token_type,
            scores,
            merges,
            id("tokenizer.ggml.bos_token_id"),
            id("tokenizer.ggml.eos_token_id"),
        )?;
        t.pre = PreTokenizer::from_name(g.get_str("tokenizer.ggml.pre").unwrap_or("default"));
        if let Some(GgufValue::Bool(b)) = g.get("tokenizer.ggml.add_bos_token") {
            t.add_bos = *b;
        }
        t.chat_template = g.get_str("tokenizer.chat_template").map(String::from);
        Ok(t)
    }

    /// The `tokenizer.ggml.*` metadata for a GGUF writer.
    pub fn gguf_metadata(&self) -> Vec<(String, GgufValue)> {
        let mut v = vec![
            (
                "tokenizer.ggml.model".to_string(),
                GgufValue::Str(match self.kind {
                    TokenizerKind::Gpt2 => "gpt2".into(),
                    TokenizerKind::SentencePiece => "llama".into(),
                }),
            ),
            (
                "tokenizer.ggml.tokens".into(),
                GgufValue::str_array(self.tokens.iter().cloned()),
            ),
            (
                "tokenizer.ggml.token_type".into(),
                GgufValue::Arr(
                    5,
                    self.token_type.iter().map(|&t| GgufValue::I32(t)).collect(),
                ),
            ),
            (
                "tokenizer.ggml.scores".into(),
                GgufValue::Arr(6, self.scores.iter().map(|&s| GgufValue::F32(s)).collect()),
            ),
        ];
        if !self.merges.is_empty() {
            v.push((
                "tokenizer.ggml.merges".into(),
                GgufValue::str_array(self.merges.iter().map(|(a, b)| format!("{a} {b}"))),
            ));
        }
        if let Some(b) = self.bos {
            v.push(("tokenizer.ggml.bos_token_id".into(), GgufValue::U32(b)));
        }
        if let Some(e) = self.eos {
            v.push(("tokenizer.ggml.eos_token_id".into(), GgufValue::U32(e)));
        }
        let pre = match self.pre {
            PreTokenizer::Gpt2 => "default",
            PreTokenizer::Smollm => "smollm",
            PreTokenizer::Llama3 => "llama-bpe",
            PreTokenizer::Qwen2 => "qwen2",
        };
        v.push(("tokenizer.ggml.pre".into(), GgufValue::Str(pre.into())));
        v.push((
            "tokenizer.ggml.add_bos_token".into(),
            GgufValue::Bool(self.add_bos),
        ));
        if let Some(t) = &self.chat_template {
            v.push(("tokenizer.chat_template".into(), GgufValue::Str(t.clone())));
        }
        v
    }

    /// Read a Hugging Face `tokenizer.json` (byte-level BPE models).
    pub fn from_hf_json(j: &Json) -> Result<Self, TokenizerError> {
        let model = j
            .get("model")
            .ok_or_else(|| TokenizerError("tokenizer.json has no model".into()))?;
        if model.get("type").and_then(|t| t.as_str()) != Some("BPE") {
            return Err(TokenizerError(
                "only BPE tokenizer.json models are supported".into(),
            ));
        }
        let vocab = model
            .get("vocab")
            .and_then(|v| v.as_object())
            .ok_or_else(|| TokenizerError("no vocab".into()))?;
        let mut added: Vec<(u32, String)> = j
            .get("added_tokens")
            .and_then(|a| a.as_array())
            .unwrap_or(&[])
            .iter()
            .filter_map(|t| {
                Some((
                    t.get("id")?.as_u64()? as u32,
                    t.get("content")?.as_str()?.to_string(),
                ))
            })
            .collect();
        let n = vocab
            .iter()
            .filter_map(|(_, v)| v.as_u64())
            .chain(added.iter().map(|(i, _)| *i as u64))
            .max()
            .map(|m| m as usize + 1)
            .unwrap_or(0);
        let mut tokens = vec![String::new(); n];
        let mut types = vec![TOKEN_UNUSED; n];
        for (k, v) in vocab {
            if let Some(i) = v.as_u64() {
                tokens[i as usize] = k.clone();
                types[i as usize] = TOKEN_NORMAL;
            }
        }
        added.sort();
        for (i, c) in &added {
            tokens[*i as usize] = c.clone();
            types[*i as usize] = TOKEN_CONTROL;
        }
        for (i, t) in tokens.iter_mut().enumerate() {
            if t.is_empty() {
                *t = format!("<|unused_{i}|>");
            }
        }
        let merges = model
            .get("merges")
            .and_then(|m| m.as_array())
            .unwrap_or(&[])
            .iter()
            .filter_map(|m| match m {
                Json::Str(s) => s
                    .split_once(' ')
                    .map(|(a, b)| (a.to_string(), b.to_string())),
                Json::Arr(p) if p.len() == 2 => {
                    Some((p[0].as_str()?.to_string(), p[1].as_str()?.to_string()))
                }
                _ => None,
            })
            .collect();
        let find = |name: &str| added.iter().find(|(_, c)| c == name).map(|(i, _)| *i);
        let bos = find("<|bos|>")
            .or_else(|| find("<s>"))
            .or_else(|| find("<|begin_of_text|>"));
        let eos = find("<|eos|>")
            .or_else(|| find("</s>"))
            .or_else(|| find("<|end_of_text|>"))
            .or_else(|| find("<|endoftext|>"));
        Self::new(
            TokenizerKind::Gpt2,
            tokens,
            types,
            Vec::new(),
            merges,
            bos,
            eos,
        )
    }

    /// A Hugging Face `tokenizer.json` for this (GPT-2 style) tokenizer.
    pub fn to_hf_json(&self) -> Json {
        let mut vocab = Vec::new();
        let mut added = Vec::new();
        for (i, t) in self.tokens.iter().enumerate() {
            if self.token_type[i] == TOKEN_CONTROL {
                added.push(Json::obj(vec![
                    ("id", Json::num(i as f64)),
                    ("content", Json::str(t.clone())),
                    ("special", Json::Bool(true)),
                ]));
            } else if self.token_type[i] != TOKEN_UNUSED {
                vocab.push((t.clone(), Json::num(i as f64)));
            }
        }
        Json::obj(vec![
            ("version", Json::str("1.0")),
            ("added_tokens", Json::Arr(added)),
            (
                "pre_tokenizer",
                Json::obj(vec![("type", Json::str("ByteLevel"))]),
            ),
            ("decoder", Json::obj(vec![("type", Json::str("ByteLevel"))])),
            (
                "model",
                Json::obj(vec![
                    ("type", Json::str("BPE")),
                    ("vocab", Json::Obj(vocab)),
                    (
                        "merges",
                        Json::Arr(
                            self.merges
                                .iter()
                                .map(|(a, b)| Json::str(format!("{a} {b}")))
                                .collect(),
                        ),
                    ),
                ]),
            ),
        ])
    }

    /// Train a GPT-2 byte-level BPE vocabulary of exactly `vocab_size`
    /// tokens on `corpus`: 256 byte tokens, `<|bos|>` / `<|eos|>` control
    /// tokens when there is room, merges of the most frequent adjacent pair
    /// (ties broken by token ids, so training is deterministic), and
    /// `<|unused_N|>` padding if the corpus runs out of pairs.
    pub fn train_bpe(corpus: &str, vocab_size: usize) -> Self {
        assert!(vocab_size >= 256, "byte-level BPE needs ≥ 256 tokens");
        let b2u = bytes_to_unicode();
        let mut tokens: Vec<String> = (0..256).map(|b| b2u[b].to_string()).collect();
        let mut types = vec![TOKEN_NORMAL; 256];
        let specials = if vocab_size >= 258 { 2 } else { 0 };
        let mut counts: HashMap<Vec<u32>, u64> = HashMap::new();
        for piece in gpt2_pretokenize(corpus) {
            *counts
                .entry(piece.bytes().map(|b| b as u32).collect())
                .or_insert(0) += 1;
        }
        let mut words: Vec<(Vec<u32>, u64)> = counts.into_iter().collect();
        words.sort();
        let mut merges = Vec::new();
        while tokens.len() + specials < vocab_size {
            let mut pairs: HashMap<(u32, u32), u64> = HashMap::new();
            for (w, c) in &words {
                for p in w.windows(2) {
                    *pairs.entry((p[0], p[1])).or_insert(0) += c;
                }
            }
            let best = pairs
                .into_iter()
                .max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0)));
            let Some(((l, r), _)) = best else { break };
            let id = tokens.len() as u32;
            merges.push((tokens[l as usize].clone(), tokens[r as usize].clone()));
            tokens.push(format!("{}{}", tokens[l as usize], tokens[r as usize]));
            types.push(TOKEN_NORMAL);
            for (w, _) in words.iter_mut() {
                let mut i = 0;
                let mut out = Vec::with_capacity(w.len());
                while i < w.len() {
                    if i + 1 < w.len() && w[i] == l && w[i + 1] == r {
                        out.push(id);
                        i += 2;
                    } else {
                        out.push(w[i]);
                        i += 1;
                    }
                }
                *w = out;
            }
        }
        let (mut bos, mut eos) = (None, None);
        if specials == 2 {
            bos = Some(tokens.len() as u32);
            tokens.push("<|bos|>".into());
            types.push(TOKEN_CONTROL);
            eos = Some(tokens.len() as u32);
            tokens.push("<|eos|>".into());
            types.push(TOKEN_CONTROL);
        }
        while tokens.len() < vocab_size {
            tokens.push(format!("<|unused_{}|>", tokens.len()));
            types.push(TOKEN_UNUSED);
        }
        Self::new(
            TokenizerKind::Gpt2,
            tokens,
            types,
            Vec::new(),
            merges,
            bos,
            eos,
        )
        .expect("trained vocabulary is complete")
    }

    /// The same vocabulary as a SentencePiece-style tokenizer (scores =
    /// −merge rank, `▁` for spaces, `<0xNN>` byte fallback).
    pub fn to_sentencepiece(&self) -> Self {
        let mut tokens = Vec::new();
        let mut types = Vec::new();
        let mut scores = Vec::new();
        for b in 0..256 {
            tokens.push(format!("<0x{b:02X}>"));
            types.push(TOKEN_BYTE);
            scores.push(0.0);
        }
        let mut seen: std::collections::HashSet<String> = Default::default();
        for (i, t) in self.tokens.iter().enumerate().skip(256) {
            let ty = self.token_type[i];
            if ty == TOKEN_NORMAL {
                let Ok(s) = std::str::from_utf8(self.token_bytes(i as u32)) else {
                    continue;
                };
                let piece = s.replace(' ', "\u{2581}");
                if seen.insert(piece.clone()) {
                    tokens.push(piece);
                    types.push(TOKEN_NORMAL);
                    scores.push(-(i as f32));
                }
            } else {
                tokens.push(t.clone());
                types.push(ty);
                scores.push(0.0);
            }
        }
        // the bare word-boundary piece must exist too
        let space = "\u{2581}".to_string();
        if seen.insert(space.clone()) {
            tokens.push(space);
            types.push(TOKEN_NORMAL);
            scores.push(-1e6);
        }
        let find = |name: &str| tokens.iter().position(|t| t == name).map(|i| i as u32);
        let (bos, eos) = (find("<|bos|>"), find("<|eos|>"));
        Self::new(
            TokenizerKind::SentencePiece,
            tokens,
            types,
            scores,
            Vec::new(),
            bos,
            eos,
        )
        .expect("valid SPM vocabulary")
    }
}

/// Incremental detokenizer: emits only complete UTF-8 characters.
pub struct StreamDecoder<'a> {
    tok: &'a Tokenizer,
    pending: Vec<u8>,
    started: bool,
}

impl StreamDecoder<'_> {
    /// Feed one token; returns the text that became complete.
    pub fn push(&mut self, t: u32) -> String {
        let mut b = self.tok.token_bytes(t);
        if self.tok.kind == TokenizerKind::SentencePiece
            && !self.started
            && b.first() == Some(&b' ')
        {
            b = &b[1..];
        }
        if !b.is_empty() {
            self.started = true;
        }
        self.pending.extend_from_slice(b);
        self.drain(false)
    }

    /// Flush whatever is left (incomplete sequences become U+FFFD).
    pub fn finish(&mut self) -> String {
        self.drain(true)
    }

    fn drain(&mut self, flush: bool) -> String {
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    return out;
                }
                Err(e) => {
                    let ok = e.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.pending[..ok]).unwrap());
                    match e.error_len() {
                        Some(n) => {
                            out.push('\u{FFFD}');
                            self.pending.drain(..ok + n);
                        }
                        None => {
                            self.pending.drain(..ok);
                            if flush {
                                out.push('\u{FFFD}');
                                self.pending.clear();
                            }
                            return out;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt2_pretokenizer_matches_the_reference_pattern() {
        assert_eq!(gpt2_pretokenize("Hello world"), vec!["Hello", " world"]);
        assert_eq!(
            gpt2_pretokenize("it's  2024!!\n"),
            vec!["it", "'s", " ", " 2024", "!!", "\n"]
        );
        assert_eq!(gpt2_pretokenize("a   b"), vec!["a", "  ", " b"]);
        assert_eq!(gpt2_pretokenize("x\n\ny"), vec!["x", "\n", "\n", "y"]);
    }
}
