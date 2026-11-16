//! Chat formatting from a model's `tokenizer.chat_template`.
//!
//! The Jinja template itself is not interpreted. Its *family* is recognised
//! from the markers it emits and that family's rendering is reproduced
//! exactly, including the template's own default system prompt (read from
//! the literal text in the template) and its stop tokens:
//!
//! * **ChatML** (`<|im_start|>role\n…<|im_end|>\n`): SmolLM2, Qwen2.5, Qwen3.
//!   Qwen3 templates with thinking support get the empty think block that
//!   `enable_thinking=false` produces.
//! * **Llama 3** (`<|start_header_id|>role<|end_header_id|>\n\n…<|eot_id|>`),
//!   with the 3.1/3.2 "Cutting Knowledge Date" system header when present.
//! * **Mistral / Llama 2** (`[INST] … [/INST]`).
//! * **Plain** (no template): `<|role|>` blocks, as V8.0 used.

use super::Tokenizer;

#[derive(Clone, Debug, PartialEq)]
pub enum ChatFormat {
    ChatMl {
        default_system: Option<String>,
        empty_think: bool,
    },
    Llama3 {
        date_header: bool,
    },
    Mistral,
    Plain,
}

/// The literal default system prompt a ChatML template inserts when the
/// conversation has none (`'<|im_start|>system\n…<|im_end|>'` in the source).
fn default_system(t: &str) -> Option<String> {
    let key = "<|im_start|>system";
    let mut rest = t;
    while let Some(i) = rest.find(key) {
        let after = &rest[i + key.len()..];
        // the newline is an escape inside a Jinja string, or a real one
        let after = after
            .strip_prefix("\\n")
            .or_else(|| after.strip_prefix('\n'))
            .unwrap_or(after);
        if let Some(j) = after.find("<|im_end|>") {
            let lit = &after[..j];
            if !lit.is_empty() && !lit.contains(['{', '\'', '+']) {
                return Some(lit.replace("\\n", "\n"));
            }
        }
        rest = after;
    }
    None
}

impl ChatFormat {
    pub fn detect(tok: &Tokenizer) -> Self {
        let Some(t) = tok.chat_template.as_deref() else {
            return if tok.token_id("<|im_start|>").is_some() {
                ChatFormat::ChatMl {
                    default_system: None,
                    empty_think: false,
                }
            } else {
                ChatFormat::Plain
            };
        };
        if t.contains("<|im_start|>") {
            ChatFormat::ChatMl {
                default_system: default_system(t),
                empty_think: t.contains("<think>") && t.contains("enable_thinking"),
            }
        } else if t.contains("<|start_header_id|>") {
            ChatFormat::Llama3 {
                date_header: t.contains("Cutting Knowledge Date"),
            }
        } else if t.contains("[INST]") {
            ChatFormat::Mistral
        } else {
            ChatFormat::Plain
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            ChatFormat::ChatMl {
                empty_think: true, ..
            } => "ChatML (Qwen3, thinking off)",
            ChatFormat::ChatMl { .. } => "ChatML",
            ChatFormat::Llama3 { .. } => "Llama 3",
            ChatFormat::Mistral => "Mistral [INST]",
            ChatFormat::Plain => "plain",
        }
    }

    /// Render `(role, content)` messages plus the assistant generation prompt.
    pub fn render(&self, tok: &Tokenizer, messages: &[(String, String)]) -> String {
        let mut s = String::new();
        match self {
            ChatFormat::ChatMl {
                default_system,
                empty_think,
            } => {
                if messages.first().map(|m| m.0.as_str()) != Some("system") {
                    if let Some(sys) = default_system {
                        s.push_str(&format!("<|im_start|>system\n{sys}<|im_end|>\n"));
                    }
                }
                for (r, c) in messages {
                    s.push_str(&format!("<|im_start|>{r}\n{c}<|im_end|>\n"));
                }
                s.push_str("<|im_start|>assistant\n");
                if *empty_think {
                    s.push_str("<think>\n\n</think>\n\n");
                }
            }
            ChatFormat::Llama3 { date_header } => {
                s.push_str("<|begin_of_text|>");
                let (sys, rest) = match messages.first() {
                    Some((r, c)) if r == "system" => (c.as_str(), &messages[1..]),
                    _ => ("", messages),
                };
                if *date_header || !sys.is_empty() {
                    s.push_str("<|start_header_id|>system<|end_header_id|>\n\n");
                    if *date_header {
                        s.push_str(
                            "Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n",
                        );
                    }
                    s.push_str(sys);
                    s.push_str("<|eot_id|>");
                }
                for (r, c) in rest {
                    s.push_str(&format!(
                        "<|start_header_id|>{r}<|end_header_id|>\n\n{}<|eot_id|>",
                        c.trim()
                    ));
                }
                s.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
            }
            ChatFormat::Mistral => {
                if let Some(b) = tok.bos.and_then(|b| tok.tokens.get(b as usize)) {
                    s.push_str(b);
                }
                let mut sys = String::new();
                for (r, c) in messages {
                    match r.as_str() {
                        "system" => sys = format!("{c}\n\n"),
                        "user" => {
                            s.push_str(&format!("[INST] {}{c} [/INST]", std::mem::take(&mut sys)))
                        }
                        _ => {
                            s.push_str(c);
                            if let Some(e) = tok.eos.and_then(|e| tok.tokens.get(e as usize)) {
                                s.push_str(e);
                            }
                        }
                    }
                }
            }
            ChatFormat::Plain => {
                for (r, c) in messages {
                    s.push_str(&format!("<|{r}|>\n{c}\n"));
                }
                s.push_str("<|assistant|>\n");
            }
        }
        s
    }

    /// Prompt token ids (special tokens parsed; BOS added when the
    /// tokenizer asks for it and the template did not emit one).
    pub fn encode(&self, tok: &Tokenizer, messages: &[(String, String)]) -> Vec<u32> {
        let text = self.render(tok, messages);
        let mut ids = tok.encode_special(&text);
        if tok.add_bos && !matches!(self, ChatFormat::Llama3 { .. } | ChatFormat::Mistral) {
            if let Some(b) = tok.bos {
                if ids.first() != Some(&b) {
                    ids.insert(0, b);
                }
            }
        }
        ids
    }
}

/// Token ids that end an assistant turn for this tokenizer.
pub fn stop_tokens(tok: &Tokenizer) -> Vec<u32> {
    let mut v: Vec<u32> = tok.eos.into_iter().collect();
    for s in [
        "<|im_end|>",
        "<|eot_id|>",
        "<|eom_id|>",
        "<|endoftext|>",
        "</s>",
        "<|end|>",
    ] {
        if let Some(id) = tok
            .token_id(s)
            .filter(|&i| tok.token_type[i as usize] == super::bpe::TOKEN_CONTROL)
        {
            if !v.contains(&id) {
                v.push(id);
            }
        }
    }
    v
}
