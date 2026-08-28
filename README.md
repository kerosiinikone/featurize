# featurize-core

Compile-time-checked, (WASM) `no_std` preprocessing pipelines for simple numeric and image
feature extraction. Pre-built (*WIP*) library of image and standard operations for manipulating data vectors via resampling, index remapping and point-wise operations. The crate allows for declaratively constructing fused preprocessing pipelines with minimal abstraction overhead.

## Quick Start

````rust
use featurize_core::prelude::*;
use featurize_core::errors::PropagateNan;

// MNIST Example
let mut pipe = Pipeline::new_with::<f32, PropagateNan>()
    .apply_transform(Scale2D::<300, 300, 4, 28, 28, 4, f32>::new())
    .apply_transform(Grayscale::<28, 28, 4, _>::new())
    .apply_element(Div::new(255.0))
    .apply_element(Normalize::new(0.3081, 0.1307))
    .build();
````

## Trade-offs and motivation

The crate is built around minimal abstraction overhead and compile-time safety feature preprocessing tasks. Considering the target environments (WASM) and the aforementioned goals, one major trade-off is between **runtime efficiency and binary size**. Monomorphization in the typestate composition increases the size of the crate and might cause issues in some resource-constrained `no_std` environments. This cannot currently be circumvented even with dynamic pipelines. 

Another trade-off is between NaN handling as a standalone operation (e.g. Python frameworks) in the chain and pipeline-wide NaN handling policies. As the crate does not make any assumptions about its consumers' the use-cases, the ability to "fail fast" in order to retain data integrity and avoid corrupted data in the front end application is necessary. Thus, the pipeline employs a policy-based strategy of allowing the consumer to choose between the behaviour of their pipelines depending on their preference between maximum efficiency (*default IEEE-754 NaN propagation behaviour, no-op*) and strict control (*failing fast on poor data*).

## Features

| Feature  | Description                                          |
| -------- | ---------------------------------------------------- |
| `burn`   | Convert pipeline output into a `burn` tensor.        |
| `candle` | Convert pipeline output into a `candle` tensor.      |

Both are disabled by default.

## Platform support

The crate is `#![no_std]` (using `alloc`) and is verified in CI against:

- `x86_64-unknown-linux-gnu`
- `wasm32-unknown-unknown`

## Status

The crate is currently in early stages and lacks many functionalities of production-grade libraries. The multi-stage pipelines still hinder performance somewhat in order to safely allow both static and dynamic bounds (despite the use of scratch buffers, see *TBD*). The standard operation set included in the crate is quite small and should be improved in the future. The failing tests comprise of precision errors in `prop_add_subtract_inverse` and `prop_multiply_associative`.
