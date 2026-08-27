//! `featurize-core` provides compile-time-checked preprocessing pipelines for
//! numeric and image feature extraction (ML).
//!
//! Document here
//!
//! # Features
//!
//! - `burn` — convert pipeline output directly into a `burn` tensor.
//! - `candle` — convert pipeline output directly into a `candle` tensor.
//!
//! Both are off by default. The crate is `no_std` (with `alloc`) and builds for
//! `wasm32-unknown-unknown`

#![no_std]

extern crate alloc;

pub mod errors;
pub mod image;
pub mod ops;
pub mod pipeline;
pub mod prelude;
pub mod tensors;
pub mod traits;

/// Sentinel length marking a pipeline stage whose size is only known at
/// runtime; such stages resolve their buffer sizes via the `*_dynamic` methods
/// on [`Stage`](traits::Stage).
pub const DYNAMIC_SIZE: usize = 0;

pub(crate) const fn _const_max_usize(a: usize, b: usize) -> usize {
    if a > b {
        a
    } else {
        b
    }
}
