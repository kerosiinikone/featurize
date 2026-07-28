#[cfg(feature = "burn")]
pub mod burn_ext {
    use burn::tensor::backend::{Backend, BackendTypes};
    use burn::tensor::{Element, Shape, Tensor, TensorData};

    use crate::pipeline::PipeExec;
    use crate::traits::{Float, Stage};

    pub trait IntoBurnTensor<F: Float = f32> {
        fn into_burn_tensor<B, const D: usize>(
            &mut self,
            input: &[F],
            shape: impl Into<Shape>,
            device: &<B as BackendTypes>::Device,
        ) -> Tensor<B, D>
        where
            B: BackendTypes + Backend;
    }

    // Naive impl
    impl<F: Float + Element, S: Stage<F>> IntoBurnTensor<F> for PipeExec<S, F> {
        fn into_burn_tensor<B, const D: usize>(
            &mut self,
            input: &[F],
            shape: impl Into<Shape>,
            device: &<B as BackendTypes>::Device,
        ) -> Tensor<B, D>
        where
            B: BackendTypes + Backend,
        {
            let exec = self
                .stages
                .execute(input, &mut self.in_buf, &mut self.out_buf)
                .unwrap();
            let tensor_data = TensorData::new(exec.to_vec(), shape);
            Tensor::<B, D>::from_floats(tensor_data, &device)
        }
    }
}

#[cfg(feature = "burn")]
pub use burn_ext::IntoBurnTensor;
