#!/bin/bash
set -e

cargo build --target wasm32-unknown-unknown --release --no-default-features --target-dir ./target

wasm-bindgen ./target/wasm32-unknown-unknown/release/app.wasm --out-dir build --target web
