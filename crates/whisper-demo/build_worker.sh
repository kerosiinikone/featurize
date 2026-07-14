#!/bin/bash

# Script to build the worker WASM module

set -e

echo "Building worker module..."
cd "$(dirname "$0")"

# Build the m binary for wasm32
cargo build --target wasm32-unknown-unknown --bin m --release

# Run wasm-bindgen on the output
wasm-bindgen --target web --out-dir build --out-name m \
  target/wasm32-unknown-unknown/release/m.wasm

echo "Worker module built successfully!"
echo "Output: build/m.js and build/m_bg.wasm"
