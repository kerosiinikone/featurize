#[cfg(feature = "burn")]
mod burn_ext {
    use burn::tensor::backend::{Backend, BackendTypes};
    use burn::tensor::{Element, Shape, Tensor, TensorData};

    use crate::errors::{ErrorKind, NanHandler, PipeError};
    use crate::pipeline::PipeExec;
    use crate::traits::{Float, Stage};

    /// Convert the pipeline output directly into a `burn` [`Tensor`].
    ///
    /// The conversion performs exactly one copy of the pipeline output (from
    /// the pipeline's scratch buffer into the tensor-owned storage). The
    /// element type `F` is preserved end-to-end; any dtype conversion is
    /// deferred to the backend, which is a no-op when `F` matches the
    /// backend's float element type.
    ///
    /// The pipeline-wide NaN policy is applied exactly as in
    /// [`PipeExec::execute`]: the stage tree is monomorphized for the
    /// handler chosen at construction time.
    pub trait IntoBurnTensor<F: Float = f32> {
        fn to_burn_tensor<B, const D: usize>(
            &mut self,
            input: &[F],
            shape: impl Into<Shape>,
            device: &<B as BackendTypes>::Device,
        ) -> Result<Tensor<B, D>, PipeError>
        where
            B: BackendTypes + Backend;

        /// Byte-input variant mirroring [`PipeExec::execute_from_bytes`]:
        /// the input is reinterpreted as a slice of `F` before execution.
        fn to_burn_tensor_from_bytes<B, const D: usize>(
            &mut self,
            input: &[u8],
            shape: impl Into<Shape>,
            device: &<B as BackendTypes>::Device,
        ) -> Result<Tensor<B, D>, PipeError>
        where
            B: BackendTypes + Backend,
            F: bytemuck::Pod;
    }

    impl<F: Float + Element, S: Stage<F>, N: NanHandler> IntoBurnTensor<F> for PipeExec<S, N, F> {
        fn to_burn_tensor<B, const D: usize>(
            &mut self,
            input: &[F],
            shape: impl Into<Shape>,
            device: &<B as BackendTypes>::Device,
        ) -> Result<Tensor<B, D>, PipeError>
        where
            B: BackendTypes + Backend,
        {
            // `N` is the pipeline-wide, compile-time NaN policy
            let exec = self
                .stages
                .execute::<N>(input, &mut self.in_buf, &mut self.out_buf)?;

            // Pre-validate the requested shape against the actual output
            // length so a mismatch surfaces as a PipeError instead of a
            // panic inside burn.
            let shape: Shape = shape.into();
            if shape.num_elements() != exec.len() {
                return Err(PipeError::with_message(
                    ErrorKind::ShapeMismatch,
                    alloc::format!(
                        "shape {:?} requires {} elements, pipeline produced {}",
                        shape,
                        shape.num_elements(),
                        exec.len()
                    ),
                ));
            }

            let tensor_data = TensorData::new(exec.to_vec(), shape);
            Ok(Tensor::<B, D>::from_floats(tensor_data, device))
        }

        fn to_burn_tensor_from_bytes<B, const D: usize>(
            &mut self,
            input: &[u8],
            shape: impl Into<Shape>,
            device: &<B as BackendTypes>::Device,
        ) -> Result<Tensor<B, D>, PipeError>
        where
            B: BackendTypes + Backend,
            F: bytemuck::Pod,
        {
            // PipeExec::execute_from_bytes.
            let floats: &[F] = bytemuck::try_cast_slice(input)?;
            self.to_burn_tensor(floats, shape, device)
        }
    }
}

#[cfg(feature = "candle")]
mod candle_ext {
    use candle_core::{shape::ShapeWithOneHole, Device, Error, Tensor, WithDType};

    use crate::{
        errors::{ErrorKind::CandleTensorError, NanHandler, PipeError},
        pipeline::PipeExec,
        traits::{Float, Stage},
    };

    impl From<Error> for PipeError {
        fn from(value: Error) -> Self {
            PipeError::with_message(CandleTensorError, alloc::format!("{}", value))
        }
    }

    /// Convert the pipeline output directly into a `candle` [`Tensor`].
    ///
    /// The conversion performs exactly one copy of the pipeline output (from
    /// the pipeline's scratch buffer into the tensor-owned storage) and
    /// preserves the native dtype of `F` (e.g. `f32` stays `F32`). The
    /// requested `shape` may contain a single "hole", e.g. `((), 80, 3000)`.
    ///
    /// The pipeline-wide NaN policy is applied exactly as in
    /// [`PipeExec::execute`]: the stage tree is monomorphized for the
    /// handler chosen at construction time.
    pub trait IntoCandleTensor<F: Float = f32> {
        fn to_candle_tensor<S: ShapeWithOneHole>(
            &mut self,
            input: &[F],
            shape: S,
            device: &Device,
        ) -> Result<Tensor, PipeError>;

        /// Byte-input variant mirroring [`PipeExec::execute_from_bytes`]:
        /// the input is reinterpreted as a slice of `F` before execution.
        fn to_candle_tensor_from_bytes<S: ShapeWithOneHole>(
            &mut self,
            input: &[u8],
            shape: S,
            device: &Device,
        ) -> Result<Tensor, PipeError>
        where
            F: bytemuck::Pod;
    }

    impl<F: Float + WithDType, T: Stage<F>, N: NanHandler> IntoCandleTensor<F> for PipeExec<T, N, F> {
        fn to_candle_tensor<S: ShapeWithOneHole>(
            &mut self,
            input: &[F],
            shape: S,
            device: &Device,
        ) -> Result<Tensor, PipeError> {
            // `N` is the pipeline-wide, compile-time NaN policy
            let exec = self
                .stages
                .execute::<N>(input, &mut self.in_buf, &mut self.out_buf)?;

            // Single copy directly from the pipeline scratch buffer into the
            // tensor storage, preserving the native dtype of `F`. No f64
            // round-trip, no intermediate Vec, no silent conversion failures.
            let flat = Tensor::from_slice(exec, exec.len(), device)?;

            // `reshape` on a contiguous tensor is metadata-only (no copy) and
            // resolves one-hole shapes; a shape/length mismatch surfaces as a
            // candle Error and is mapped into a PipeError (with the original
            // message preserved).
            Ok(flat.reshape(shape)?)
        }

        fn to_candle_tensor_from_bytes<S: ShapeWithOneHole>(
            &mut self,
            input: &[u8],
            shape: S,
            device: &Device,
        ) -> Result<Tensor, PipeError>
        where
            F: bytemuck::Pod,
        {
            // PipeExec::execute_from_bytes.
            let floats: &[F] = bytemuck::try_cast_slice(input)?;
            self.to_candle_tensor(floats, shape, device)
        }
    }
}

#[cfg(feature = "burn")]
pub use burn_ext::IntoBurnTensor;

#[cfg(feature = "candle")]
pub use candle_ext::IntoCandleTensor;
