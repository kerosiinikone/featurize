use core::marker::PhantomData;

use alloc::vec;
use alloc::vec::Vec;

use crate::traits::{
    ElementElement, ElementMark, ElementOp, Fused, Head, Link, ResampleMark, Stage,
    TransformElement, TransformOp,
};

/// Initializer
#[allow(dead_code)]
#[derive(Debug, Default, Clone)]
pub struct Pipeline {
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
    // These are allocated only at the build stage
    // TODO: not an option
    in_buf: Option<Vec<f32>>,
    out_buf: Option<Vec<f32>>,
    // BACKEND INTEGRATION THROUGH EXT TRAIT (not in core)
}

#[allow(dead_code)]
impl Pipeline {
    pub fn new() -> Pipeline {
        Pipeline {
            in_buf: Default::default(),
            out_buf: Default::default(),
        }
    }

    /// Creates the initialized pipe implicitly
    pub fn apply_point<T, const LEN: usize>(self, op: T) -> Pipe<Head<T, ElementMark, LEN>>
    where
        T: ElementOp,
    {
        Pipe {
            stages: Head {
                root_op: op,
                marker: PhantomData {},
            },
            in_buf: self.in_buf,
            out_buf: self.out_buf,
        }
    }

    /// Creates the initialized pipe implicitly
    pub fn apply_transform<T, const LEN: usize>(self, op: T) -> Pipe<Head<T, ResampleMark, LEN>>
    where
        T: TransformOp,
    {
        Pipe {
            stages: Head {
                root_op: op,
                marker: PhantomData {},
            },
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
        }
    }
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

        let exec = &self.stages.execute(input, in_buf, out_buf);
        output[..exec.len()].copy_from_slice(&exec);
    }
}

impl<T: ElementOp, const LEN: usize> Pipe<Head<T, ElementMark, LEN>> {
    pub fn apply_point<U>(self, op: U) -> Pipe<Head<Fused<T, U, ElementElement>, ElementMark, LEN>>
    where
        U: ElementOp,
    {
        let prev_head = self.stages;

        Pipe {
            stages: Head {
                marker: prev_head.marker,
                root_op: prev_head.root_op.fuse_element(op),
            },
            in_buf: self.in_buf,
            out_buf: self.out_buf,
        }
    }

    pub fn apply_transform<U>(
        self,
        op: U,
    ) -> Pipe<Head<Fused<U, T, TransformElement>, ResampleMark, LEN>>
    where
        U: TransformOp,
    {
        let prev_head = self.stages;

        Pipe {
            stages: Head {
                root_op: prev_head.root_op.fuse_after_transform(op),
                marker: PhantomData {},
            },
            in_buf: self.in_buf,
            out_buf: self.out_buf,
        }
    }
}

impl<T: TransformOp, const LEN: usize> Pipe<Head<T, ResampleMark, LEN>> {
    pub fn apply_point<U>(
        self,
        op: U,
    ) -> Pipe<Head<Fused<T, U, TransformElement>, ResampleMark, LEN>>
    where
        U: ElementOp,
    {
        let prev_head = self.stages;

        Pipe {
            stages: Head {
                marker: prev_head.marker,
                root_op: prev_head.root_op.fuse_element(op),
            },
            in_buf: self.in_buf,
            out_buf: self.out_buf,
        }
    }

    pub fn apply_transform<U>(
        self,
        op: U,
    ) -> Pipe<Link<U, Head<T, ResampleMark, LEN>, ResampleMark>>
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
            in_buf: self.in_buf,
            out_buf: self.out_buf,
        }
    }
}

impl<T: ElementOp, S: Stage> Pipe<Link<T, S, ElementMark>> {
    pub fn apply_point<U>(self, op: U) -> Pipe<Link<Fused<T, U>, S, ElementMark>>
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
            in_buf: self.in_buf,
            out_buf: self.out_buf,
        }
    }

    pub fn apply_transform<U>(
        self,
        op: U,
    ) -> Pipe<Link<Fused<U, T, TransformElement>, S, ResampleMark>>
    where
        U: TransformOp,
    {
        let stages = self.stages;

        Pipe {
            stages: Link {
                prev_stage: stages.prev_stage,
                curr_op: stages.curr_op.fuse_after_transform(op),
                marker: PhantomData {},
            },
            in_buf: self.in_buf,
            out_buf: self.out_buf,
        }
    }
}

impl<T: TransformOp, S: Stage> Pipe<Link<T, S, ResampleMark>> {
    pub fn apply_point<U>(self, op: U) -> Pipe<Link<Fused<T, U, TransformElement>, S, ResampleMark>>
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
            in_buf: self.in_buf,
            out_buf: self.out_buf,
        }
    }

    pub fn apply_transform<U>(self, op: U) -> Pipe<Link<U, Link<T, S, ResampleMark>, ResampleMark>>
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
            in_buf: self.in_buf,
            out_buf: self.out_buf,
        }
    }
}
