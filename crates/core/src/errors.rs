use bytemuck::PodCastError;

use crate::traits::Float;

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
    #[cfg(feature = "candle")]
    ShapeMismatch,
    /// NaN or infinity encountered
    NaN,
    /// Candle Error
    #[cfg(feature = "candle")]
    CandleTensorError,
}

/// Compile-time (monomorphized) NaN / infinity policy.
///
/// The policy is chosen *once, per pipeline* (see
/// [`crate::pipeline::Pipeline`]) and is threaded through every stage and
/// operation as a generic type parameter. Because the implementors are
/// zero-sized and every method is `#[inline(always)]`, the compiler sees a
/// single, statically known check inside the computation loops:
///
/// * [`FailOnNan`] - one `is_finite` test plus an early return
///   ([`ErrorKind::NaN`]),
/// * [`ZeroOnNan`] - one `is_finite` test that lowers to a branchless
///   select,
/// * [`PropagateNan`] - no instructions at all; IEEE-754 values flow
///   through untouched.
///
/// This replaces the old runtime `NanHandling` field that had to be matched
/// on for every single element.
pub trait NanHandler: Default + Copy + 'static {
    /// The runtime-visible variant this policy corresponds to
    const HANDLING: NanHandling;

    /// Apply the policy to a freshly computed value
    fn check_finite<T: Float>(value: T) -> Result<T, PipeError>;
}

/// Runtime-visible description of the three NaN policies.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NanHandling {
    /// Abort the pipeline with [`ErrorKind::NaN`] on the first non-finite value
    Fail,
    /// Replace non-finite values with zero
    Zero,
    /// IEEE 754 - let NaN / infinity propagate untouched
    Propagate,
}

impl Default for NanHandling {
    fn default() -> Self {
        Self::Fail
    }
}

/// Fail fast: the first non-finite value aborts the pipeline
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FailOnNan;

/// Replace every non-finite value with zero
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ZeroOnNan;

/// Pure IEEE 754 semantics: never inspect the value
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PropagateNan;

impl NanHandler for FailOnNan {
    const HANDLING: NanHandling = NanHandling::Fail;

    #[inline(always)]
    fn check_finite<T: Float>(value: T) -> Result<T, PipeError> {
        if value.is_finite() {
            Ok(value)
        } else {
            Err(PipeError::new(ErrorKind::NaN))
        }
    }
}

impl NanHandler for ZeroOnNan {
    const HANDLING: NanHandling = NanHandling::Zero;

    #[inline(always)]
    fn check_finite<T: Float>(value: T) -> Result<T, PipeError> {
        // This lowers to a select / blend, keeping the
        // surrounding loop vectorizable
        if value.is_finite() {
            Ok(value)
        } else {
            Ok(T::zero())
        }
    }
}

impl NanHandler for PropagateNan {
    const HANDLING: NanHandling = NanHandling::Propagate;

    #[inline(always)]
    fn check_finite<T: Float>(value: T) -> Result<T, PipeError> {
        // Deliberately no check whatsoever: this call disappears entirely
        Ok(value)
    }
}

/// Dynamic (non-monomorphized) variant of the NaN check.
///
/// Kept for interop and for callers that genuinely need a runtime-selected
/// policy. Pipeline operations must use `N::check_finite` instead, since
/// this version forces the compiler to keep the match (and the early-exit
/// path) inside the computation loops.
#[inline(always)]
pub fn check_finite<T: num_traits::Float>(value: T, handling: NanHandling) -> Result<T, PipeError> {
    if value.is_finite() {
        Ok(value)
    } else {
        match handling {
            NanHandling::Fail => Err(PipeError::new(ErrorKind::NaN)),
            NanHandling::Zero => Ok(T::zero()),
            NanHandling::Propagate => Ok(value),
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
