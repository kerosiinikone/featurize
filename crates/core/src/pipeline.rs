use crate::{
    errors::PipeError,
    traits::{
        EMark, ElementElement, ElementOp, Float, Fused, Head, IndexRemappable, IsTrue, Link, Stage,
        TMark, TransformElement, TransformOp, TransformTransform,
    },
};

// Check exports!
pub struct Static;
pub struct Dynamic;

/// Initializer
#[derive(Debug, Default, Clone)]
pub struct Pipeline;

/// Wrapper for the pipeline stages
///
/// Rotates a set of temporary buffers
/// to avoid allocating mid pipe
#[derive(Debug, Default, Clone)]
pub struct Pipe<T, F, State>
where
    T: Stage<F>,
    F: Float,
{
    // Pipe typestate
    // Accessing the underlying T -> helper methods
    stages: T,
    marker: core::marker::PhantomData<(State, F)>,
    // Options
}

#[derive(Debug, Default, Clone)]
pub struct PipeExec<T, F = f32>
where
    T: Stage<F>,
    F: Float,
{
    // Pipe typestate
    // Accessing the underlying T -> helper methods
    stages: T,
    // Options
    // Input data parameters
    #[allow(dead_code)]
    max_expected_input_length: Option<usize>,
    // These are allocated only at the build stage
    in_buf: alloc::boxed::Box<[F]>,
    out_buf: alloc::boxed::Box<[F]>,
}

impl Pipeline {
    pub fn new() -> PipelineStatic<f32> {
        PipelineStatic {
            marker: core::marker::PhantomData {},
        }
    }

    pub fn new_with_dynamic() -> PipelineDynamic<f32> {
        PipelineDynamic {
            marker: core::marker::PhantomData {},
        }
    }

    pub fn new_f64() -> PipelineStatic<f64> {
        PipelineStatic {
            marker: core::marker::PhantomData {},
        }
    }

    pub fn new_with_dynamic_f64() -> PipelineDynamic<f64> {
        PipelineDynamic {
            marker: core::marker::PhantomData {},
        }
    }
}

/// Static pipeline initializer
#[derive(Debug, Default, Clone)]
pub struct PipelineStatic<F: Float = f32> {
    marker: core::marker::PhantomData<(Static, F)>,
}

/// Dynamic pipeline initializer
#[derive(Debug, Default, Clone)]
pub struct PipelineDynamic<F: Float = f32> {
    marker: core::marker::PhantomData<(Dynamic, F)>,
}

impl<F: Float> PipelineStatic<F> {
    /// Creates the initialized pipe implicitly (with size)
    pub fn apply_point<T, const INPUT_LEN: usize>(
        self,
        op: T,
    ) -> Pipe<Head<T, EMark, INPUT_LEN>, F, Static>
    where
        T: ElementOp<F>,
    {
        Pipe {
            stages: Head {
                root_op: op,
                marker: core::marker::PhantomData {},
            },
            marker: core::marker::PhantomData {},
        }
    }
    /// Creates the initialized pipe implicitly (with size implied -> transform operation must be
    /// static!)
    pub fn apply_transform<T>(self, op: T) -> Pipe<Head<T, TMark>, F, Static>
    where
        T: TransformOp<F>,
    {
        const {
            assert!(T::IN_LEN > 0 && T::OUT_LEN > 0, "Dynamic bounds");
        }

        Pipe {
            stages: Head {
                root_op: op,
                marker: core::marker::PhantomData {},
            },
            marker: core::marker::PhantomData {},
        }
    }
}

impl<F: Float> PipelineDynamic<F> {
    /// Creates the initialized pipe implicitly
    pub fn apply_point<T>(self, op: T) -> Pipe<Head<T, EMark>, F, Dynamic>
    where
        T: ElementOp<F>,
    {
        Pipe {
            stages: Head {
                root_op: op,
                marker: core::marker::PhantomData {},
            },
            marker: core::marker::PhantomData {},
        }
    }

    /// Creates the initialized pipe implicitly
    pub fn apply_transform<T>(self, op: T) -> Pipe<Head<T, TMark>, F, Dynamic>
    where
        T: TransformOp<F>,
    {
        Pipe {
            stages: Head {
                root_op: op,
                marker: core::marker::PhantomData {},
            },
            marker: core::marker::PhantomData {},
        }
    }
}

impl<T, F> Pipe<T, F, Static>
where
    T: Stage<F>,
    F: Float,
{
    // The input data dims must be known at this point
    pub fn build(self) -> PipeExec<T, F> {
        const {
            assert!(T::MAX_BUF_SIZE > 0, "Pipeline has dynamic bounds");
        }
        let buf_size = T::MAX_BUF_SIZE;

        PipeExec {
            in_buf: alloc::vec![F::zero(); buf_size].into_boxed_slice(),
            out_buf: alloc::vec![F::zero(); buf_size].into_boxed_slice(),
            stages: self.stages,
            max_expected_input_length: None,
        }
    }
}

impl<T, F> Pipe<T, F, Dynamic>
where
    T: Stage<F>,
    F: Float,
{
    // Dynamic bound checking
    pub fn build_dynamic(self, max_expected_input_length: usize) -> PipeExec<T, F> {
        let buf_size = self.stages.max_buf_size_dynamic(max_expected_input_length);

        PipeExec {
            in_buf: alloc::vec![F::zero(); buf_size].into_boxed_slice(),
            out_buf: alloc::vec![F::zero(); buf_size].into_boxed_slice(),
            stages: self.stages,
            max_expected_input_length: Some(max_expected_input_length),
        }
    }
}

impl<T, F> PipeExec<T, F>
where
    T: Stage<F>,
    F: Float,
{
    pub fn execute(&mut self, input: &[F], output_buf: &mut [F]) -> Result<usize, PipeError> {
        let out_buf = &mut self.out_buf;
        let in_buf = &mut self.in_buf;

        let exec = self.stages.execute(input, in_buf, out_buf)?;
        let exec_len = exec.len();

        if output_buf.len() < exec_len {
            return Err(PipeError::new(crate::errors::ErrorKind::InvalidOutputSize));
        }

        output_buf[..exec_len].copy_from_slice(exec);
        Ok(exec_len)
    }

    pub fn output_len(&self) -> usize {
        self.stages
            .out_len_dynamic(self.max_expected_input_length.unwrap_or(0))
    }
}

impl<T, F, const INPUT_LEN: usize, State> Pipe<Head<T, EMark, INPUT_LEN>, F, State>
where
    T: ElementOp<F>,
    F: Float,
{
    pub fn apply_point<U>(
        self,
        op: U,
    ) -> Pipe<Head<Fused<T, U, ElementElement>, EMark, INPUT_LEN>, F, State>
    where
        U: ElementOp<F>,
    {
        let prev_head = self.stages;

        Pipe {
            stages: Head {
                marker: prev_head.marker,
                root_op: prev_head.root_op.fuse_element(op),
            },
            marker: core::marker::PhantomData {},
        }
    }

    pub fn apply_transform<U>(
        self,
        op: U,
    ) -> Pipe<Head<Fused<U, T, TransformElement>, TMark, INPUT_LEN>, F, State>
    where
        U: TransformOp<F>,
    {
        const {
            assert!(U::INTERNAL_IS_VALID, "Invalid input length");
        }
        let prev_head = self.stages;

        Pipe {
            stages: Head {
                root_op: prev_head.root_op.fuse_transform(op),
                marker: core::marker::PhantomData {},
            },
            marker: core::marker::PhantomData {},
        }
    }
}

impl<T, F, const INPUT_LEN: usize, State> Pipe<Head<T, TMark, INPUT_LEN>, F, State>
where
    T: TransformOp<F>,
    F: Float,
{
    pub fn apply_point<U>(
        self,
        op: U,
    ) -> Pipe<Head<Fused<T, U, TransformElement>, TMark, INPUT_LEN>, F, State>
    where
        U: ElementOp<F>,
    {
        let prev_head = self.stages;

        Pipe {
            stages: Head {
                marker: core::marker::PhantomData {},
                root_op: prev_head.root_op.fuse_element(op),
            },
            marker: core::marker::PhantomData {},
        }
    }

    pub fn apply_transform<U>(
        self,
        op: U,
    ) -> Pipe<Link<U, Head<T, TMark, INPUT_LEN>, TMark, F>, F, State>
    where
        U: TransformOp<F>,
    {
        const {
            assert!(
                U::INTERNAL_IS_VALID
                    && (T::OUT_LEN == U::IN_LEN || T::OUT_LEN == 0 || U::IN_LEN == 0),
                "Invalid input length"
            );
        }
        let prev_head = self.stages;

        Pipe {
            stages: Link {
                curr_op: op,
                prev_stage: prev_head,
                marker: core::marker::PhantomData {},
                _float_marker: core::marker::PhantomData {},
            },
            marker: core::marker::PhantomData {},
        }
    }
}

impl<T, F, const INPUT_LEN: usize, State> Pipe<Head<T, TMark, INPUT_LEN>, F, State>
where
    T: TransformOp<F> + IndexRemappable<F>,
    T::IndexRemapping: IsTrue,
    F: Float,
{
    pub fn apply_transform_fusable<U>(
        self,
        op: U,
    ) -> Pipe<Head<Fused<T, U, TransformTransform>, TMark, INPUT_LEN>, F, State>
    where
        U: TransformOp<F>,
        U::IndexRemapping: IsTrue,
    {
        const {
            assert!(
                U::INTERNAL_IS_VALID
                    && (T::OUT_LEN == U::IN_LEN || T::OUT_LEN == 0 || U::IN_LEN == 0),
                "Invalid input length"
            );
        }
        let prev_head = self.stages;

        Pipe {
            stages: Head {
                marker: core::marker::PhantomData {},
                root_op: prev_head.root_op.fuse_transform(op),
            },
            marker: core::marker::PhantomData {},
        }
    }
}

impl<T, S, F, State> Pipe<Link<T, S, EMark, F>, F, State>
where
    T: ElementOp<F>,
    S: Stage<F>,
    F: Float,
{
    pub fn apply_point<U>(self, op: U) -> Pipe<Link<Fused<T, U>, S, EMark, F>, F, State>
    where
        U: ElementOp<F>,
    {
        let stages = self.stages;

        Pipe {
            stages: Link {
                prev_stage: stages.prev_stage,
                curr_op: stages.curr_op.fuse_element(op),
                marker: core::marker::PhantomData {},
                _float_marker: core::marker::PhantomData {},
            },
            marker: core::marker::PhantomData {},
        }
    }

    pub fn apply_transform<U>(
        self,
        op: U,
    ) -> Pipe<Link<Fused<U, T, TransformElement>, S, TMark, F>, F, State>
    where
        U: TransformOp<F>,
    {
        const {
            // TODO: make sure the prev stage check is correct
            assert!(
                U::INTERNAL_IS_VALID
                    && (S::OUT_LEN == U::IN_LEN || S::OUT_LEN == 0 || U::IN_LEN == 0),
                "Invalid input length"
            );
        }
        let stages = self.stages;

        Pipe {
            stages: Link {
                prev_stage: stages.prev_stage,
                curr_op: stages.curr_op.fuse_transform(op),
                marker: core::marker::PhantomData {},
                _float_marker: core::marker::PhantomData {},
            },
            marker: core::marker::PhantomData {},
        }
    }
}

impl<T, S, F, State> Pipe<Link<T, S, TMark, F>, F, State>
where
    T: TransformOp<F>,
    S: Stage<F>,
    F: Float,
{
    pub fn apply_point<U>(
        self,
        op: U,
    ) -> Pipe<Link<Fused<T, U, TransformElement>, S, TMark, F>, F, State>
    where
        U: ElementOp<F>,
    {
        let stages = self.stages;

        Pipe {
            stages: Link {
                prev_stage: stages.prev_stage,
                curr_op: stages.curr_op.fuse_element(op),
                marker: core::marker::PhantomData {},
                _float_marker: core::marker::PhantomData {},
            },
            marker: core::marker::PhantomData {},
        }
    }

    pub fn apply_transform<U>(
        self,
        op: U,
    ) -> Pipe<Link<U, Link<T, S, TMark, F>, TMark, F>, F, State>
    where
        U: TransformOp<F>,
    {
        const {
            assert!(
                U::INTERNAL_IS_VALID
                    && (T::OUT_LEN == U::IN_LEN || T::OUT_LEN == 0 || U::IN_LEN == 0),
                "Invalid input length"
            );
        }
        let stages = self.stages;

        Pipe {
            stages: Link {
                prev_stage: stages,
                curr_op: op,
                marker: core::marker::PhantomData {},
                _float_marker: core::marker::PhantomData {},
            },
            marker: core::marker::PhantomData {},
        }
    }
}

impl<T, S, F, State> Pipe<Link<T, S, TMark, F>, F, State>
where
    T: TransformOp<F> + IndexRemappable<F>,
    T::IndexRemapping: IsTrue,
    S: Stage<F>,
    F: Float,
{
    pub fn apply_transform_fusable<U>(
        self,
        op: U,
    ) -> Pipe<Link<Fused<T, U, TransformTransform>, S, TMark, F>, F, State>
    where
        U: TransformOp<F>,
        U::IndexRemapping: IsTrue,
    {
        const {
            assert!(
                U::INTERNAL_IS_VALID
                    && (T::OUT_LEN == U::IN_LEN || T::OUT_LEN == 0 || U::IN_LEN == 0),
                "Invalid input length"
            );
        }
        let stages = self.stages;

        Pipe {
            stages: Link {
                marker: core::marker::PhantomData {},
                curr_op: stages.curr_op.fuse_transform(op),
                prev_stage: stages.prev_stage,
                _float_marker: core::marker::PhantomData {},
            },
            marker: core::marker::PhantomData {},
        }
    }
}
