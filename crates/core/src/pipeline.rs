use core::marker::PhantomData;

use alloc::vec;
use alloc::vec::Vec;

use crate::traits::{
    ElementElement, ElementMark, ElementOp, FusedPoint, Head, Link, PipeOp, TransformElement,
    ResampleMark, TransformOp, RootOp, Stage,
};

/// Initializer
#[allow(dead_code)]
#[derive(Debug, Default, Clone)]
pub struct Pipeline {
    stride: usize,
    shape: Vec<usize>,
    in_buf: Option<Vec<f32>>,
    out_buf: Option<Vec<f32>>,
}

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
    // Input data parameters
    stride: usize,
    shape: Vec<usize>,
    // These are allocated only at the build stage
    in_buf: Option<Vec<f32>>,
    out_buf: Option<Vec<f32>>,
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
    stride: usize,
    shape: Vec<usize>,
    // These are allocated only at the build stage
    in_buf: Option<Vec<f32>>,
    out_buf: Option<Vec<f32>>,
    // BACKEND INTEGRATION THROUGH EXT TRAIT (not in core)
}

#[allow(dead_code)]
impl Pipeline {
    pub fn new(shape: Vec<usize>, stride: usize) -> Pipeline {
        Pipeline {
            shape,
            stride,
            in_buf: Default::default(),
            out_buf: Default::default(),
        }
    }

    /// Creates the initialized Pipe implicitly
    pub fn apply_point<T>(self, op: T) -> Pipe<Head<T, ElementMark>>
    where
        T: RootOp + ElementOp,
    {
        Pipe {
            stages: Head {
                root_op: op,
                marker: PhantomData {},
            },
            stride: self.stride,
            shape: self.shape,
            in_buf: self.in_buf,
            out_buf: self.out_buf,
        }
    }

    /// Creates the initialized Pipe implicitly
    pub fn apply_transform<T>(self, op: T) -> Pipe<Head<T, ResampleMark>>
    where
        T: RootOp + TransformOp,
    {
        Pipe {
            stages: Head {
                root_op: op,
                marker: PhantomData {},
            },
            stride: self.stride,
            shape: self.shape,
            in_buf: self.in_buf,
            out_buf: self.out_buf,
        }
    }
}

#[allow(dead_code)]
impl<T> Pipe<T>
where
    T: Stage,
{
    // Build the arenas based on buf_size of its operations and stages
    // return type of builder should be PipelineExec or sum
    pub fn build(self) -> PipeExec<T>
    where
        T: Stage,
    {
        let buf_size = self.stages.buf_size();

        PipeExec {
            in_buf: Some(vec![0.0; buf_size]),
            out_buf: Some(vec![0.0; buf_size]),
            stages: self.stages,
            stride: self.stride,
            shape: self.shape,
        }
    }

    // HELPERS
}

impl<T: Stage> PipeExec<T> {
    pub fn execute(&mut self, input: &[f32], output: &mut [f32])
    where
        T: Stage,
    {
        let out_buf = if let Some(buf) = &mut self.out_buf {
            buf.as_mut_slice()
        } else {
            panic!("empty buffer");
        };

        let in_buf = if let Some(buf) = &mut self.in_buf {
            buf.as_mut_slice()
        } else {
            panic!("empty buffer");
        };

        let exec = &self
            .stages
            .execute(input, in_buf, out_buf, self.shape.as_slice(), self.stride);
        output[..exec.len()].copy_from_slice(&exec);
    }
}

impl<T: RootOp + ElementOp> Pipe<Head<T, ElementMark>> {
    pub fn apply_point<U>(self, op: U) -> Pipe<Head<FusedPoint<T, U, ElementElement>, ElementMark>>
    where
        U: ElementOp + RootOp,
    {
        let prev_head = self.stages;

        Pipe {
            stages: Head {
                marker: prev_head.marker,
                root_op: prev_head.root_op._apply_point(op),
            },
            stride: self.stride,
            shape: self.shape,
            in_buf: self.in_buf,
            out_buf: self.out_buf,
        }
    }

    // FOR NOW
    pub fn apply_transform<U>(self, op: U) -> Pipe<Link<U, Head<T, ElementMark>, ResampleMark>>
    where
        U: TransformOp + PipeOp,
    {
        // DEBUG
        let stages = self.stages;

        Pipe {
            stages: Link {
                prev_stage: stages,
                curr_op: op,
                marker: PhantomData {},
            },
            stride: self.stride,
            shape: self.shape,
            in_buf: self.in_buf,
            out_buf: self.out_buf,
        }
    }
    // Everything that can fused is fused, otherwise a new stage is generated
    // fn apply_resample()
}

impl<T: RootOp + TransformOp> Pipe<Head<T, ResampleMark>> {
    pub fn apply_point<U>(
        self,
        op: U,
    ) -> Pipe<Head<FusedPoint<T, U, TransformElement>, ResampleMark>>
    where
        U: ElementOp + RootOp,
    {
        let prev_head = self.stages;

        Pipe {
            stages: Head {
                marker: prev_head.marker,
                root_op: prev_head.root_op._apply_point(op),
            },
            stride: self.stride,
            shape: self.shape,
            in_buf: self.in_buf,
            out_buf: self.out_buf,
        }
    }

    // FOR NOW
    pub fn apply_transform<U>(
        self,
        op: U,
    ) -> Pipe<Link<U, Head<T, ResampleMark>, ResampleMark>>
    where
        U: TransformOp + RootOp + PipeOp,
    {
        // DEBUG
        let prev_head = self.stages;

        Pipe {
            stages: Link {
                curr_op: op,
                prev_stage: prev_head,
                marker: PhantomData {},
            },
            stride: self.stride,
            shape: self.shape,
            in_buf: self.in_buf,
            out_buf: self.out_buf,
        }
    }
    // Everything that can fused is fused, otherwise a new stage is generated
    // fn apply_resample()
}

impl<T: PipeOp + ElementOp, S: Stage> Pipe<Link<T, S, ElementMark>> {
    pub fn apply_point<U>(self, op: U) -> Pipe<Link<FusedPoint<T, U>, S, ElementMark>>
    where
        U: ElementOp + PipeOp,
    {
        // DEBUG
        let stages = self.stages;

        Pipe {
            stages: Link {
                prev_stage: stages.prev_stage,
                curr_op: stages.curr_op._apply_point(op),
                marker: PhantomData {},
            },
            stride: self.stride,
            shape: self.shape,
            in_buf: self.in_buf,
            out_buf: self.out_buf,
        }
    }

    // TODO: later -> interchangeable
    pub fn apply_transform<U>(self, op: U) -> Pipe<Link<U, Link<T, S, ElementMark>, ResampleMark>>
    where
        U: TransformOp + PipeOp,
    {
        // DEBUG
        let stages = self.stages;

        Pipe {
            stages: Link {
                prev_stage: stages,
                curr_op: op,
                marker: PhantomData {},
            },
            stride: self.stride,
            shape: self.shape,
            in_buf: self.in_buf,
            out_buf: self.out_buf,
        }
    }
}

impl<T: PipeOp + TransformOp, S: Stage> Pipe<Link<T, S, ResampleMark>> {
    pub fn apply_point<U>(
        self,
        op: U,
    ) -> Pipe<Link<FusedPoint<T, U, TransformElement>, S, ResampleMark>>
    where
        U: ElementOp + PipeOp,
    {
        // DEBUG
        let stages = self.stages;

        Pipe {
            stages: Link {
                prev_stage: stages.prev_stage,
                curr_op: stages.curr_op._apply_point(op),
                marker: PhantomData {},
            },
            stride: self.stride,
            shape: self.shape,
            in_buf: self.in_buf,
            out_buf: self.out_buf,
        }
    }

    // TODO: should merge
    pub fn apply_transform<U>(self, op: U) -> Pipe<Link<U, Link<T, S, ResampleMark>, ResampleMark>>
    where
        U: TransformOp + PipeOp,
    {
        // DEBUG
        let stages = self.stages;

        Pipe {
            stages: Link {
                prev_stage: stages,
                curr_op: op,
                marker: PhantomData {},
            },
            stride: self.stride,
            shape: self.shape,
            in_buf: self.in_buf,
            out_buf: self.out_buf,
        }
    }
}
