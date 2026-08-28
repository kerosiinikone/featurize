//! `featurize-core` provides compile-time-checked preprocessing pipelines for
//! simple numeric and image feature extraction (ML).
//!
//! Pre-built (*WIP*) library of image and standard operations for manipulating
//! data vectors via either [`TransformOp`](crate::traits::TransformOp) (resampling, index
//! remapping operations) or [`ElementOp`](crate::traits::ElementOp) (point-wise operations).
//! The crate allows for declaratively constructing fused preprocessing pipelines
//! with minimal abstraction overhead.
//!
//! # Quick Start
//!
//! ```
//! use featurize_core::prelude::*;
//! use featurize_core::errors::PropagateNan;
//!
//! // MNIST Example
//! let mut pipe = Pipeline::new_with::<f32, PropagateNan>()
//!     .apply_transform(Scale2D::<300, 300, 4, 28, 28, 4, f32>::new())
//!     .apply_transform(Grayscale::<28, 28, 4, _>::new())
//!     .apply_element(Div::new(255.0))
//!     .apply_element(Normalize::new(0.3081, 0.1307))
//!     .build();
//! ```
//!
//! # Architecture
//!
//! ## Static vs. Dynamic
//!
//! The crate was desgined to work with data vectors of known lengths to offer compile-time
//! safety for building pipelines. These do, *however*, restrict the crate
//! somewhat in its applications, so a dynamic variant was introduced. Dynamic in this sense
//! means that the pipeline allows for processing data vectors of arbitrary lengths not known
//! beforehand. This does mean that certain "static" operations (`TransformOp`) will not be
//! supported when opting for [`Pipeline::with_dynamic`](crate::pipeline::Pipeline::with_dynamic).
//!
//! ## Fusion
//!
//! The pipeline performs loop fusion in order to optimize the declaratively created structure.
//! Fusion is only possible with `ElementOp` and `TransformOp` which are purely index remapping
//! operations (associated type `IndexRemapping` set to `True`) as these are isomorphic and
//! therefore allow for operating within the same dimensions.
//!
//! # Features
//!
//! - `burn` — convert pipeline output directly into a `burn` tensor.
//! - `candle` — convert pipeline output directly into a `candle` tensor.
//!
//! Both are off by default. The crate is `no_std` (with `alloc`) and builds for
//! `wasm32-unknown-unknown`.
//!
//! # Troubleshooting
//!
//! ### Dynamic Bounds
//!
//! ```compile_fail
//! use featurize_core::prelude::*;
//!
//! // This will fail because the pipeline has dynamic operations
//! // but we're trying to build it statically (the implicit choice)
//! let pipe = Pipeline::new::<f32>()
//!     .apply_transform(Reverse::<0>::new())  // Dynamic
//!     .build();
//! ```
//!
//! Use `build_dynamic()` instead:
//! ```
//! use featurize_core::prelude::*;
//!
//! let pipe = Pipeline::with_dynamic::<f32>()
//!     .apply_transform(Reverse::<0>::new())
//!     .build_dynamic(256);
//! ```

#![no_std]
#![doc = include_str!("../../../README.md")]

extern crate alloc;

pub mod errors;
pub mod image;
pub mod ops;
pub mod pipeline;
pub mod prelude;
pub mod tensors;
pub mod traits;

/// Sentinel length marking a pipeline stage whose size is only known at
/// runtime; such stages resolve their buffer sizes via the `*_dynamic`.
pub const DYNAMIC_SIZE: usize = 0;

#[doc(hidden)]
pub(crate) const fn _const_max_usize(a: usize, b: usize) -> usize {
    if a > b {
        a
    } else {
        b
    }
}
