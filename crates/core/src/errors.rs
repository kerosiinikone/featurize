use alloc::string::String;
use core::{
    error,
    fmt::{Debug, Display},
};

// use crate::traits::Stage;

#[non_exhaustive]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    InvalidInputSize,
    InvalidOutputSize,
    InvalidComputation,
}

#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipeError {
    kind: ErrorKind,
    // Snapshot of the pipe (with a Display impl?)
    // stages: Option<S>,
    // source: Option<Box<dyn error::Error + Send + Sync>>,
    message: Option<String>,
    // TODO
    // backtrace: Option<backtrace::Backtrace>
}

impl Display for PipeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", "PipeError")
    }
}

impl error::Error for PipeError {}

impl PipeError
{
    pub fn new(kind: ErrorKind) -> PipeError {
        Self {
            kind,
            // stages: None,
            // source: None,
            message: None,
        }
    }
}
