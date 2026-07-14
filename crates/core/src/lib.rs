#![no_std]

extern crate alloc;

pub mod mel_spectrogram;
pub mod pipeline;
pub mod prelude;
pub mod traits;

#[cfg(test)]
mod tests {
    #[test]
    fn test_pipe_mnist() {
        // 600 x 300 x 4 -> source
        // let mut output = vec![0.0; 784];
        // Shape and stride need to match to the input -> execute time check
        // 28 x 28
        // let pipe = Pipeline::new()
        //     .apply_transform(Scale::<>
        //     )
        //     .apply_transform(Grayscale {
        //         in_channels: 4,
        //         out_size: 784,
        //         invert: true,
        //     })
        //     .apply_point(Normalize {
        //         std: 0.3081,
        //         mean: 0.1307,
        //         size: 784,
        //     })
        //     .apply_point(Div {
        //         factor: 255f32,
        //         size: 784,
        //     });
        // let mut exec = pipe.build();
        // // INTO ADAPTER -> specify here or at init the tensor parameters
        // // Create a view for Burn to discourage allocation (to_vec)
        // exec.execute((vec![1.0; INPUT_LEN]).as_slice(), output.as_mut_slice());

        // assert_eq!(output[0], 16.0);
        // assert_eq!(output[25], 0.0);
        // assert_eq!(output[24], 16.0);
    }
}
