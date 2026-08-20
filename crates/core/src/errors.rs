use bytemuck::PodCastError;

/// Error kinds that can occur during pipeline execution
#[non_exhaustive]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// Invalid byte input
    InvalidByteInput,
    /// Input size doesn't match expected size
    InvalidInputSize,
    /// Output buffer is too small
    InvalidOutputSize,
    /// Requested tensor shape doesn't match the pipeline output length
    ShapeMismatch,
    /// NaN or infinity encountered
    NaN,
    /// Candle Error
    #[cfg(feature = "candle")]
    CandleTensorError,
}

#[non_exhaustive]
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

// TODO: checked for every value -> kills vectorization
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
    snapshot: Option<alloc::string::String>,
    message: Option<alloc::string::String>,
    // backtrace: Option<backtrace::Backtrace>
}

impl core::fmt::Display for PipeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "PipeError")?;
        if let Some(msg) = &self.message {
            write!(f, ": {}", msg)?;
        }
        if let Some(ctx) = &self.snapshot {
            write!(f, " ({})", ctx)?;
        }
        Ok(())
    }
}

impl core::error::Error for PipeError {}

impl From<PodCastError> for PipeError {
    fn from(value: PodCastError) -> Self {
        PipeError::with_message(ErrorKind::InvalidByteInput, alloc::format!("{}", value))
    }
}

impl PipeError {
    pub fn new(kind: ErrorKind) -> PipeError {
        Self {
            kind,
            message: None,
            snapshot: None,
        }
    }

    pub fn with_message(kind: ErrorKind, message: alloc::string::String) -> PipeError {
        Self {
            kind,
            message: Some(message),
            snapshot: None,
        }
    }

    pub fn with_snapshot(kind: ErrorKind, snapshot: alloc::string::String) -> PipeError {
        Self {
            kind,
            message: None,
            snapshot: Some(snapshot),
        }
    }

    pub fn message(&self) -> Option<&alloc::string::String> {
        self.message.as_ref()
    }

    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    pub fn snapshot(&self) -> Option<&alloc::string::String> {
        self.snapshot.as_ref()
    }
}
