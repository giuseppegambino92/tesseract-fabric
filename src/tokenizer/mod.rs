//! Tokenization and sampling: byte-level BPE / SentencePiece tokenizers
//! loaded from GGUF metadata or `tokenizer.json`, streaming detokenization,
//! and the composable logit sampler.

pub mod bpe;
pub mod chat;
pub mod sampler;

pub use bpe::{PreTokenizer, StreamDecoder, Tokenizer, TokenizerKind};
pub use chat::{stop_tokens, ChatFormat};
pub use sampler::SamplingParams;

/// The in-tree training text for synthetic models' vocabularies (English,
/// Italian, code, JSON, numbers, emoji).
pub const TRAINING_CORPUS: &str = include_str!("corpus.txt");

/// The byte-level BPE vocabulary the engine trains for a synthetic model
/// with `vocab_size` tokens (deterministic: same corpus, same merges).
pub fn synthetic_tokenizer(vocab_size: usize) -> Tokenizer {
    Tokenizer::train_bpe(TRAINING_CORPUS, vocab_size)
}
