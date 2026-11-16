//! Asynchronous speculative tree-braid engine: drafting, tree masks,
//! exact tree rejection sampling and braid strand management.

pub mod drafter;
pub mod tree_braid;

pub use drafter::{DraftConfig, FeatureDrafter, LowRankDraftHead, NgramTable, TreeDrafter};
pub use tree_braid::{
    build_tree_mask, tree_depths, verify_tree, MergedTree, ProposalKind, RooflineController,
    SpeculativeTree, TreeCommitFrame, VerifyOutcome,
};
