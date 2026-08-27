# featurize-core

Compile-time-checked, WASM `no_std` preprocessing pipelines for numeric and image
feature extraction.

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

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your
option.
