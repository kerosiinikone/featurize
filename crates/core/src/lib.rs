#![no_std]

extern crate alloc;

pub mod pipeline;
pub mod prelude;
pub mod traits;

#[cfg(test)]
mod tests {
    use alloc::vec;

    use crate::{
        pipeline::Pipeline,
        traits::{_TestOp, Scale},
    };

    use super::*;

    #[test]
    fn test_link() {
        let mut output = vec![0.0; 50];
        // Shape and stride need to match to the input -> execute time check
        // 28 x 28
        let pipe = Pipeline::new(vec![28; 2], 4)
            // SOMEHOW GET THE INPUT DATA SIZE ON ELEMENT OPS
            .apply_point(_TestOp { size: 784 })
            .apply_point(_TestOp { size: 784 })
            .apply_point(_TestOp { size: 784 })
            .apply_point(_TestOp { size: 784 })
            // 5 x 5
            .apply_resample(Scale {
                out_shape: vec![5; 2],
            });
        pipe.build()
            // INTO ADAPTER -> specify here or at init the tensor parameters
            .execute((vec![1.0; 1000]).as_slice(), output.as_mut_slice());

        assert_eq!(output[0], 16.0);
        assert_eq!(output[25], 0.0);
        assert_eq!(output[24], 16.0);
    }
}
