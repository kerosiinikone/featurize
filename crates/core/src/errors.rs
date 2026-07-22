use alloc::string::String;
use core::{
    error,
    fmt::{Debug, Display},
};

#[non_exhaustive]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    InvalidInputSize,
    InvalidOutputSize,
    NaN,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NanHandling {
    Fail,
    Zero,
}

impl Default for NanHandling {
    fn default() -> Self {
        Self::Fail
    }
}

#[inline(always)]
pub fn check_finite(value: f32, handling: NanHandling) -> Result<f32, PipeError> {
    if value.is_finite() {
        Ok(value)
    } else {
        match handling {
            NanHandling::Fail => Err(PipeError::new(ErrorKind::NaN)),
            NanHandling::Zero => Ok(0.0),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipeError {
    kind: ErrorKind,
    // Snapshot of the pipe (with a Display impl?)
    // stages: Option<S>,
    // TODO: CONTEXT
    // SNAPSHOT of the stage (wrapper)
    // source: Option<Box<dyn error::Error + Send + Sync>>,
    message: Option<String>,
    // TODO
    // backtrace: Option<backtrace::Backtrace>
}

// TODO: match type
impl Display for PipeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", "PipeError")
    }
}

impl error::Error for PipeError {}

impl PipeError {
    pub fn new(kind: ErrorKind) -> PipeError {
        Self {
            kind,
            // stages: None,
            // source: None,
            message: None,
        }
    }

    pub fn kind(&self) -> ErrorKind {
        self.kind
    }
}
