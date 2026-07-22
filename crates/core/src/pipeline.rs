use crate::{
    errors::PipeError,
    traits::{
        EMark, ElementElement, ElementOp, Fused, Head, IndexRemappable, IsTrue, Link, Stage, TMark,
        TransformElement, TransformOp, TransformTransform,
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
// Generic over floating-point type?
// TODO: check double trait bounding?
pub struct Pipe<T, State>
where
    T: Stage,
{
    // Pipe typestate
    // Accessing the underlying T -> helper methods
    stages: T,
    marker: core::marker::PhantomData<State>,
    // Options
}

#[derive(Debug, Default, Clone)]
// Generic over floating-point type?
pub struct PipeExec<T>
where
    T: Stage,
{
    // Pipe typestate
    // Accessing the underlying T -> helper methods
    stages: T,
    // Options
    // Input data parameters
    #[allow(dead_code)]
    max_expected_input_length: Option<usize>,
    // These are allocated only at the build stage
    in_buf: alloc::boxed::Box<[f32]>,
    out_buf: alloc::boxed::Box<[f32]>,
}

impl Pipeline {
    pub fn new() -> PipelineStatic {
        PipelineStatic {
            marker: core::marker::PhantomData {},
        }
    }

    pub fn new_with_dynamic() -> PipelineDynamic {
        PipelineDynamic {
            marker: core::marker::PhantomData {},
        }
    }
}

/// Static pipeline initializer
#[derive(Debug, Default, Clone)]
pub struct PipelineStatic {
    marker: core::marker::PhantomData<Static>,
}

/// Dynamic pipeline initializer
#[derive(Debug, Default, Clone)]
pub struct PipelineDynamic {
    marker: core::marker::PhantomData<Dynamic>,
}

impl PipelineStatic {
    /// Creates the initialized pipe implicitly (with size)
    pub fn apply_point<T, const INPUT_LEN: usize>(
        self,
        op: T,
    ) -> Pipe<Head<T, EMark, INPUT_LEN>, Static>
    where
        T: ElementOp,
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
    pub fn apply_transform<T>(self, op: T) -> Pipe<Head<T, TMark>, Static>
    where
        T: TransformOp,
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

impl PipelineDynamic {
    /// Creates the initialized pipe implicitly
    pub fn apply_point<T>(self, op: T) -> Pipe<Head<T, EMark>, Dynamic>
    where
        T: ElementOp,
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
    pub fn apply_transform<T>(self, op: T) -> Pipe<Head<T, TMark>, Dynamic>
    where
        T: TransformOp,
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

impl<T> Pipe<T, Static>
where
    T: Stage,
{
    // The input data dims must be known at this point
    pub fn build(self) -> PipeExec<T> {
        const {
            assert!(T::MAX_BUF_SIZE > 0, "Pipeline has dynamic bounds");
        }
        let buf_size = T::MAX_BUF_SIZE;

        PipeExec {
            in_buf: alloc::vec![0f32; buf_size].into_boxed_slice(),
            out_buf: alloc::vec![0f32; buf_size].into_boxed_slice(),
            stages: self.stages,
            max_expected_input_length: None,
        }
    }
}

impl<T> Pipe<T, Dynamic>
where
    T: Stage,
{
    // Dynamic bound checking
    pub fn build_dynamic(self, max_expected_input_length: usize) -> PipeExec<T> {
        let buf_size = self.stages.max_buf_size_dynamic(max_expected_input_length);

        PipeExec {
            in_buf: alloc::vec![0f32; buf_size].into_boxed_slice(),
            out_buf: alloc::vec![0f32; buf_size].into_boxed_slice(),
            stages: self.stages,
            max_expected_input_length: Some(max_expected_input_length),
        }
    }
}

impl<T: Stage> PipeExec<T> {
    pub fn execute(&mut self, input: &[f32], output_buf: &mut [f32]) -> Result<usize, PipeError> {
        let out_buf = &mut self.out_buf;
        let in_buf = &mut self.in_buf;

        let exec = self.stages.execute(input, in_buf, out_buf)?;
        // TODO: ...?
        let exec_len = self.stages.out_len_dynamic(exec.len());

        if output_buf.len() < exec_len {
            return Err(PipeError::new(crate::errors::ErrorKind::InvalidOutputSize));
        }

        output_buf[..exec_len].copy_from_slice(exec);
        Ok(exec_len)
    }

    // pipe_output_len() -> returns the known output length OR the allocated buffer length via
    // dynamic
}

impl<T: ElementOp, const INPUT_LEN: usize, State> Pipe<Head<T, EMark, INPUT_LEN>, State> {
    pub fn apply_point<U>(
        self,
        op: U,
    ) -> Pipe<Head<Fused<T, U, ElementElement>, EMark, INPUT_LEN>, State>
    where
        U: ElementOp,
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
    ) -> Pipe<Head<Fused<U, T, TransformElement>, TMark, INPUT_LEN>, State>
    where
        U: TransformOp,
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

impl<T: TransformOp, const INPUT_LEN: usize, State> Pipe<Head<T, TMark, INPUT_LEN>, State> {
    pub fn apply_point<U>(
        self,
        op: U,
    ) -> Pipe<Head<Fused<T, U, TransformElement>, TMark, INPUT_LEN>, State>
    where
        U: ElementOp,
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

    pub fn apply_transform<U>(self, op: U) -> Pipe<Link<U, Head<T, TMark, INPUT_LEN>, TMark>, State>
    where
        U: TransformOp,
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
            },
            marker: core::marker::PhantomData {},
        }
    }
}

impl<T, const INPUT_LEN: usize, State> Pipe<Head<T, TMark, INPUT_LEN>, State>
where
    T: TransformOp + IndexRemappable,
    T::IndexRemapping: IsTrue,
{
    pub fn apply_transform_fusable<U>(
        self,
        op: U,
    ) -> Pipe<Head<Fused<T, U, TransformTransform>, TMark, INPUT_LEN>, State>
    where
        U: TransformOp,
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

impl<T: ElementOp, S: Stage, State> Pipe<Link<T, S, EMark>, State> {
    pub fn apply_point<U>(self, op: U) -> Pipe<Link<Fused<T, U>, S, EMark>, State>
    where
        U: ElementOp,
    {
        let stages = self.stages;

        Pipe {
            stages: Link {
                prev_stage: stages.prev_stage,
                curr_op: stages.curr_op.fuse_element(op),
                marker: core::marker::PhantomData {},
            },
            marker: core::marker::PhantomData {},
        }
    }

    pub fn apply_transform<U>(
        self,
        op: U,
    ) -> Pipe<Link<Fused<U, T, TransformElement>, S, TMark>, State>
    where
        U: TransformOp,
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
            },
            marker: core::marker::PhantomData {},
        }
    }
}

impl<T: TransformOp, S: Stage, State> Pipe<Link<T, S, TMark>, State> {
    pub fn apply_point<U>(self, op: U) -> Pipe<Link<Fused<T, U, TransformElement>, S, TMark>, State>
    where
        U: ElementOp,
    {
        let stages = self.stages;

        Pipe {
            stages: Link {
                prev_stage: stages.prev_stage,
                curr_op: stages.curr_op.fuse_element(op),
                marker: core::marker::PhantomData {},
            },
            marker: core::marker::PhantomData {},
        }
    }

    pub fn apply_transform<U>(self, op: U) -> Pipe<Link<U, Link<T, S, TMark>, TMark>, State>
    where
        U: TransformOp,
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
            },
            marker: core::marker::PhantomData {},
        }
    }
}

impl<T, S: Stage, State> Pipe<Link<T, S, TMark>, State>
where
    T: TransformOp + IndexRemappable,
    T::IndexRemapping: IsTrue,
{
    pub fn apply_transform_fusable<U>(
        self,
        op: U,
    ) -> Pipe<Link<Fused<T, U, TransformTransform>, S, TMark>, State>
    where
        U: TransformOp,
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
            },
            marker: core::marker::PhantomData {},
        }
    }
}
