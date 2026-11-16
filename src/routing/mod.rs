//! Lookahead contextual expert router & hot-buffer pre-staging.

pub mod lookahead;

pub use lookahead::{
    decode_prewake, encode_prewake, ExpertStore, LookaheadPredictor, QExpert, Stager,
};
