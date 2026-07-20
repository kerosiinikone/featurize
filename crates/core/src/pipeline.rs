use core::marker::PhantomData;

use alloc::{boxed::Box, vec};

use crate::{
    errors::{ErrorKind, PipeError},
    traits::{
        EMark, ElementElement, ElementOp, Fused, Head, IndexRemappable, IsTrue, Link, Stage, TMark,
        TransformElement, TransformOp, TransformTransform,
    },
};

/// Initializer
#[allow(dead_code)]
#[derive(Debug, Default, Clone)]
pub struct Pipeline;

/// Wrapper for the pipeline stages
///
/// Rotates a set of temporary buffers
/// to avoid allocating mid pipe
#[allow(dead_code)]
#[derive(Debug, Default, Clone)]
// Generic over floating-point type?
pub struct Pipe<T>
where
    T: Stage,
{
    // Pipe typestate
    // Accessing the underlying T -> helper methods
    stages: T,
    // Options
}

#[allow(dead_code)]
#[derive(Debug, Default, Clone)]
// Generic over floating-point type?
// IF A FUTURE OPTION IS SPECIFIED, A NON-LENGTH GENERIC STRUCT IS USED INSTEAD (resulting in
// dynamic boud check)
pub struct PipeExec<T, const LEN: usize>
where
    T: Stage,
{
    // Pipe typestate
    // Accessing the underlying T -> helper methods
    stages: T,
    // Options
    // Input data parameters
    // These are allocated only at the build stage
    in_buf: Box<[f32]>,
    out_buf: Box<[f32]>,
    // BACKEND INTEGRATION THROUGH EXT TRAIT (not in core)
}

#[allow(dead_code)]
impl Pipeline {
    pub fn new() -> Pipeline {
        Pipeline {}
    }

    /// Creates the initialized pipe implicitly
    pub fn apply_point<T>(self, op: T) -> Pipe<Head<T, EMark>>
    where
        T: ElementOp,
    {
        Pipe {
            stages: Head {
                root_op: op,
                marker: PhantomData {},
            },
        }
    }

    /// Creates the initialized pipe implicitly
    pub fn apply_transform<T>(self, op: T) -> Pipe<Head<T, TMark>>
    where
        T: TransformOp,
    {
        Pipe {
            stages: Head {
                root_op: op,
                marker: PhantomData {},
            },
        }
    }
}

#[allow(dead_code)]
impl<T> Pipe<T>
where
    T: Stage,
{
    /// The input data dims must be known at the point, unless a (future) option is specified
    pub fn build<const LEN: usize>(self) -> PipeExec<T, LEN>
    where
        T: Stage,
    {
        let buf_size = self.stages.buf_size::<LEN>();

        PipeExec {
            in_buf: vec![0f32; buf_size].into_boxed_slice(),
            out_buf: vec![0f32; buf_size].into_boxed_slice(),
            stages: self.stages,
        }
    }
}

impl<T: Stage, const LEN: usize> PipeExec<T, LEN> {
    pub fn execute(&mut self, input: &[f32], output_buf: &mut [f32]) -> Result<(), PipeError>
    where
        T: Stage,
    {
        let output_len = self.output_len();
        let out_buf = &mut self.out_buf;
        let in_buf = &mut self.in_buf;

        if input.len() != LEN {
            return Err(PipeError::new(ErrorKind::InvalidInputSize));
        }

        if output_buf.len() < output_len {
            return Err(PipeError::new(ErrorKind::InvalidOutputSize));
        }

        let exec = &self
            .stages
            .execute::<LEN>(input, in_buf, out_buf)
            .expect("Failed");
        output_buf[..exec.len()].copy_from_slice(&exec);

        Ok(())
    }

    pub fn output_len(&self) -> usize {
        self.stages.output_len::<LEN>()
    }
}

impl<T: ElementOp> Pipe<Head<T, EMark>> {
    pub fn apply_point<U>(self, op: U) -> Pipe<Head<Fused<T, U, ElementElement>, EMark>>
    where
        U: ElementOp,
    {
        let prev_head = self.stages;

        Pipe {
            stages: Head {
                marker: prev_head.marker,
                root_op: prev_head.root_op.fuse_element(op),
            },
        }
    }

    pub fn apply_transform<U>(self, op: U) -> Pipe<Head<Fused<U, T, TransformElement>, TMark>>
    where
        U: TransformOp,
    {
        let prev_head = self.stages;

        Pipe {
            stages: Head {
                root_op: prev_head.root_op.fuse_transform(op),
                marker: PhantomData {},
            },
        }
    }
}

impl<T: TransformOp> Pipe<Head<T, TMark>> {
    pub fn apply_point<U>(self, op: U) -> Pipe<Head<Fused<T, U, TransformElement>, TMark>>
    where
        U: ElementOp,
    {
        let prev_head = self.stages;

        Pipe {
            stages: Head {
                marker: PhantomData {},
                root_op: prev_head.root_op.fuse_element(op),
            },
        }
    }

    pub fn apply_transform<U>(self, op: U) -> Pipe<Link<U, Head<T, TMark>, TMark>>
    where
        U: TransformOp,
    {
        let prev_head = self.stages;

        Pipe {
            stages: Link {
                curr_op: op,
                prev_stage: prev_head,
                marker: PhantomData {},
            },
        }
    }
}

impl<T> Pipe<Head<T, TMark>>
where
    T: TransformOp + IndexRemappable,
    T::IndexRemapping: IsTrue,
{
    pub fn apply_transform_fusable<U>(
        self,
        op: U,
    ) -> Pipe<Head<Fused<T, U, TransformTransform>, TMark>>
    where
        U: TransformOp,
        U::IndexRemapping: IsTrue,
    {
        let prev_head = self.stages;

        Pipe {
            stages: Head {
                marker: PhantomData {},
                root_op: prev_head.root_op.fuse_transform(op),
            },
        }
    }
}

impl<T: ElementOp, S: Stage> Pipe<Link<T, S, EMark>> {
    pub fn apply_point<U>(self, op: U) -> Pipe<Link<Fused<T, U>, S, EMark>>
    where
        U: ElementOp,
    {
        let stages = self.stages;

        Pipe {
            stages: Link {
                prev_stage: stages.prev_stage,
                curr_op: stages.curr_op.fuse_element(op),
                marker: PhantomData {},
            },
        }
    }

    pub fn apply_transform<U>(self, op: U) -> Pipe<Link<Fused<U, T, TransformElement>, S, TMark>>
    where
        U: TransformOp,
    {
        let stages = self.stages;

        Pipe {
            stages: Link {
                prev_stage: stages.prev_stage,
                curr_op: stages.curr_op.fuse_transform(op),
                marker: PhantomData {},
            },
        }
    }
}

impl<T: TransformOp, S: Stage> Pipe<Link<T, S, TMark>> {
    pub fn apply_point<U>(self, op: U) -> Pipe<Link<Fused<T, U, TransformElement>, S, TMark>>
    where
        U: ElementOp,
    {
        let stages = self.stages;

        Pipe {
            stages: Link {
                prev_stage: stages.prev_stage,
                curr_op: stages.curr_op.fuse_element(op),
                marker: PhantomData {},
            },
        }
    }

    pub fn apply_transform<U>(self, op: U) -> Pipe<Link<U, Link<T, S, TMark>, TMark>>
    where
        U: TransformOp,
    {
        let stages = self.stages;

        Pipe {
            stages: Link {
                prev_stage: stages,
                curr_op: op,
                marker: PhantomData {},
            },
        }
    }
}

impl<T, S: Stage> Pipe<Link<T, S, TMark>>
where
    T: TransformOp + IndexRemappable,
    T::IndexRemapping: IsTrue,
{
    pub fn apply_transform_fusable<U>(
        self,
        op: U,
    ) -> Pipe<Link<Fused<T, U, TransformTransform>, S, TMark>>
    where
        U: TransformOp,
        U::IndexRemapping: IsTrue,
    {
        let stages = self.stages;

        Pipe {
            stages: Link {
                marker: PhantomData {},
                curr_op: stages.curr_op.fuse_transform(op),
                prev_stage: stages.prev_stage,
            },
        }
    }
}
