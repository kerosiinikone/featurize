use alloc::string::String;
use core::{
    error,
    fmt::{Debug, Display},
};

/// Error kinds that can occur during pipeline execution
#[non_exhaustive]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// Input size doesn't match expected size
    InvalidInputSize,
    /// Output buffer is too small
    InvalidOutputSize,
    /// NaN or infinity encountered
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
pub fn check_finite<T: num_traits::Float>(value: T, handling: NanHandling) -> Result<T, PipeError> {
    if value.is_finite() {
        Ok(value)
    } else {
        match handling {
            NanHandling::Fail => Err(PipeError::new(ErrorKind::NaN)),
            NanHandling::Zero => Ok(T::zero()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PipeError {
    kind: ErrorKind,
    // SNAPSHOT of the stage (wrapper) -> impl Stage
    snapshot: Option<String>,
    message: Option<String>,
    // backtrace: Option<backtrace::Backtrace>
}

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
            message: None,
            snapshot: None,
        }
    }

    pub fn new_with_snapshot(kind: ErrorKind, snapshot: String) -> PipeError {
        Self {
            kind,
            message: None,
            snapshot: Some(snapshot),
        }
    }

    pub fn kind(&self) -> ErrorKind {
        self.kind
    }
}
