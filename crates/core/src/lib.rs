#![no_std]

extern crate alloc;

pub mod errors;
pub mod image;
pub mod ops;
pub mod pipeline;
pub mod prelude;
pub mod tensors;
pub mod traits;

pub const DYNAMIC_SIZE: usize = 0;

pub(crate) const fn _const_max_usize(a: usize, b: usize) -> usize {
    if a > b {
        a
    } else {
        b
    }
}
