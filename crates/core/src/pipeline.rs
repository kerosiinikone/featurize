use core::marker::PhantomData;

use alloc::{boxed::Box, vec};

use crate::traits::{
    EMark, ElementElement, ElementOp, Fused, Head, Link, Stage, TMark, TransformElement,
    TransformOp,
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
pub struct PipeExec<T>
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
    pub fn apply_point<T, const LEN: usize>(self, op: T) -> Pipe<Head<T, EMark, LEN>>
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
    pub fn apply_transform<T, const LEN: usize>(self, op: T) -> Pipe<Head<T, TMark, LEN>>
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
    pub fn build(self) -> PipeExec<T>
    where
        T: Stage,
    {
        let buf_size = self.stages.buf_size();

        PipeExec {
            in_buf: vec![0f32; buf_size].into_boxed_slice(),
            out_buf: vec![0f32; buf_size].into_boxed_slice(),
            stages: self.stages,
        }
    }
}

impl<T: Stage> PipeExec<T> {
    pub fn execute(&mut self, input: &[f32], output: &mut [f32])
    where
        T: Stage,
    {
        let out_buf = &mut self.out_buf;
        let in_buf = &mut self.in_buf;

        let exec = &self.stages.execute(input, in_buf, out_buf);
        output[..exec.len()].copy_from_slice(&exec);
    }

    pub fn output_shape(&self) -> usize {
        self.stages.output_shape()
    }
}

impl<T: ElementOp, const LEN: usize> Pipe<Head<T, EMark, LEN>> {
    pub fn apply_point<U>(self, op: U) -> Pipe<Head<Fused<T, U, ElementElement>, EMark, LEN>>
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

    pub fn apply_transform<U>(self, op: U) -> Pipe<Head<Fused<U, T, TransformElement>, TMark, LEN>>
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

impl<T: TransformOp, const LEN: usize> Pipe<Head<T, TMark, LEN>> {
    pub fn apply_point<U>(self, op: U) -> Pipe<Head<Fused<T, U, TransformElement>, TMark, LEN>>
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

    pub fn apply_transform<U>(self, op: U) -> Pipe<Link<U, Head<T, TMark, LEN>, TMark>>
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
