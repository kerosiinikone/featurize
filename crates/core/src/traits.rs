use crate::{
    _const_max_usize,
    errors::{ErrorKind, NanHandling, PipeError},
};

/// Sealing module to prevent external implementations of Stage
mod sealed {
    pub trait Sealed<T> {}
}

/// Float trait bound for pipeline operations
pub trait Float:
    num_traits::Float + num_traits::FloatConst + Default + Copy + 'static + core::iter::Sum
{
}
impl Float for f32 {}
impl Float for f64 {}

/// The pipeline wrapper encloses these 'stages'
///
/// # Buffer-size invariants
///
/// Every `execute` implementation is responsible for *proving* the bounds
/// used by the unchecked inner computation loops:
///
/// * `data.len()` is validated against the stage's static `IN_LEN` (or the
///   operation's dynamic `in_len()`),
/// * the output length `n` is validated against the static `OUT_LEN` (when
///   present) *and* against the actual allocated buffer (`out_buf.len()`),
///   never only against the static constants.
///
/// This is what keeps the unchecked loops sound even for `build_dynamic`
/// pipes that receive inputs different from `max_expected_input_length`:
/// such inputs are rejected with `InvalidInputSize` / `InvalidOutputSize`
/// instead of invoking UB.
///
/// # Sealed Trait
///
/// This trait is sealed and cannot be implemented outside of this crate.
/// Users should implement `ElementOp` or `TransformOp` instead.
pub trait Stage<T: Float = f32>: sealed::Sealed<T>
where
    Self: Sized,
{
    // Iterator support -> build from typestate?
    const IN_LEN: usize;
    const OUT_LEN: usize;
    const MAX_BUF_SIZE: usize;

    fn execute<'i, 'o>(
        &self,
        data: &[T],
        in_buf: &'i mut [T],
        out_buf: &'o mut [T],
    ) -> Result<&'o [T], PipeError>;

    /// Internal; use with errors for better context
    fn snapshot(&self) -> alloc::string::String;

    /// Computes the maximum buffer size dynamically
    fn max_buf_size_dynamic(&self, input_len: usize) -> usize;

    /// Computes the pipe output length dynamically OR
    /// the allocated buffer size
    fn out_len_dynamic(&self, input_len: usize) -> usize;
}

/// Stage marks (curr op) - internal use only
#[doc(hidden)]
pub struct Transform;
#[doc(hidden)]
pub struct Element;

/// Generic over the root operation - internal use only
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Head<T, Mark, const INPUT_LEN: usize = 0> {
    pub(crate) root_op: T,
    pub(crate) marker: core::marker::PhantomData<Mark>,
}

/// Generic over the previous stage, current operation - internal use only
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Link<T, S, Mark, F>
where
    F: Float,
    S: Stage<F>,
{
    pub(crate) prev_stage: S,
    pub(crate) curr_op: T,
    pub(crate) marker: core::marker::PhantomData<(Mark, F)>,
}

/// Point-wise operations that work on individual elements
pub trait ElementOp<T: Float = f32>
where
    Self: Sized,
{
    fn compute(&self, data: T) -> Result<T, PipeError>;

    #[inline(always)]
    fn setup(&self) {}

    /// Get the NaN handling policy for this operation
    fn nan_handling(&self) -> NanHandling {
        NanHandling::Fail
    }

    /// Get the name of this operation for error context
    fn op_name(&self) -> alloc::string::String {
        alloc::string::String::from(core::any::type_name::<Self>())
    }

    #[inline(always)]
    fn fuse_element<U>(self, op: U) -> Fused<Self, U, ElementElement>
    where
        Self: Sized,
        U: ElementOp<T>,
    {
        Fused {
            prev_op: self,
            curr_op: op,
            marker: core::marker::PhantomData {},
        }
    }

    #[inline(always)]
    fn fuse_transform<U>(self, op: U) -> Fused<U, Self, TransformElement>
    where
        Self: Sized,
        U: TransformOp<T>,
    {
        Fused {
            prev_op: op,
            curr_op: self,
            marker: core::marker::PhantomData {},
        }
    }
}

// Associated types for IndexRemappable
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct True;
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct False;
pub trait IsTrue {}
impl IsTrue for True {}

/// Spatial transformation operations that map from output index to computed value by sampling from input
///
/// # Implementor contract (relied upon by unchecked stage loops)
///
/// The stages guarantee, before dispatching into an op:
/// * `input.len() == self.in_len(..)`,
/// * `out.len() == n == self.out_len(..)`.
///
/// In return, implementations must uphold:
/// * `execute` only reads `input[..in_len]` and only writes `out[..n]`,
/// * `compute(data, out_index)` is only sound for `out_index < out_len(..)`
///   with `data.len() == in_len(..)` and must not access anything outside
///   `data`,
/// * `map_index` maps every `out_index < out_len(..)` into `[0, in_len(..))`.
///
/// Operations whose bounds depend on *runtime* fields (e.g. crop offsets)
/// must validate those themselves (once per `execute` call, and with checked
/// access in `compute`, since fused wrappers call `compute` directly).
///
/// TODO: These dynamic operations define their own `in_len()` and `out_len()`. Dynamic operation example
/// can be found at TBD
pub trait TransformOp<T: Float = f32>
where
    Self: Sized,
{
    /// Define whether an operation is a pure index remapping (can be fused)
    type IndexRemapping;

    // Creating ops with helpers (either sized or not sized, global options?)
    const IN_LEN: usize = 0;
    const OUT_LEN: usize = 0;
    /// Check the validity of passed in data compared to what is known
    ///
    /// Asserted at *every* pipe-construction site (`apply_transform`,
    /// `apply_transform_fusable`, including heads of static and dynamic
    /// pipes). Ops may cite this in `// SAFETY:` comments.
    const INTERNAL_IS_VALID: bool = true;

    /// Get the NaN handling policy for this operation
    fn nan_handling(&self) -> NanHandling {
        NanHandling::Fail
    }

    /// Get the name of this operation for error context
    fn op_name(&self) -> alloc::string::String {
        alloc::string::String::from(core::any::type_name::<Self>())
    }

    /// Map output index to input index (for pure index-remapping operations),
    /// takes a `default_len` to allow for using in dynamic pipes.
    ///
    /// Default implementation is identity mapping
    ///
    /// # Contract
    /// For every `out_index < out_len(..)` the returned index must be
    /// `< in_len(..)` -- unchecked reads in fused execution paths rely on it.
    #[inline(always)]
    fn map_index(&self, out_index: usize, _default_len: usize) -> usize
    where
        Self::IndexRemapping: IsTrue,
    {
        out_index
    }

    /// Compute output value at given output index by sampling from input data
    ///
    /// # Contract
    /// Only callable with `index < out_len(..)` and
    /// `data.len() == in_len(..)`; the stage `execute` implementations verify
    /// this via `in_len()` / `out_len()` before entering their loops.
    #[inline(always)]
    fn compute(&self, data: &[T], index: usize) -> Result<T, PipeError> {
        debug_assert!(index < data.len());
        // SAFETY: by the contract above `index < out_len(..)`, and for the
        // default identity mapping `out_len(..) <= in_len(..) == data.len()`
        // must hold (ops with differing lengths override this method).
        Ok(unsafe { *data.get_unchecked(index) })
    }

    /// Execute the transformation using chunk-based iteration (for stride operations)
    /// or index-based iteration (for non-linear operations)
    ///
    /// Callers (the `Stage` impls) guarantee `out.len() == n == out_len(..)`
    /// and `input.len() == in_len(..)`.
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [T],
        input: &'i [T],
        n: usize,
    ) -> Result<&'o mut [T], PipeError>;

    /// Runtime setup / initialization method for operations
    #[inline(always)]
    fn setup(&self) {}

    /// When defining dynamic operations, overwrite this
    #[inline(always)]
    fn in_len(&self, _default_len: usize) -> usize {
        // Static operations cannot have a zero length, this
        // must be overwritten
        const {
            assert!(Self::IN_LEN != 0);
        }
        Self::IN_LEN
    }

    /// When defining dynamic operations, overwrite this
    #[inline(always)]
    fn out_len(&self, _default_len: usize) -> usize {
        // Static operations cannot have a zero length, this
        // must be overwritten
        const {
            assert!(Self::OUT_LEN != 0);
        }
        Self::OUT_LEN
    }

    #[inline(always)]
    fn fuse_element<U>(self, op: U) -> Fused<Self, U, TransformElement>
    where
        Self: Sized,
        U: ElementOp<T>,
    {
        Fused {
            prev_op: self,
            curr_op: op,
            marker: core::marker::PhantomData {},
        }
    }
}

/// Extension trait for index-remapping TransformOps that can be fused
pub trait IndexRemappable<T: Float = f32>: TransformOp<T>
where
    Self::IndexRemapping: IsTrue,
{
    #[inline(always)]
    fn fuse_transform<U>(self, op: U) -> Fused<Self, U, TransformTransform>
    where
        Self: Sized,
        U: TransformOp<T>,
        U::IndexRemapping: IsTrue,
    {
        Fused {
            prev_op: self,
            curr_op: op,
            marker: core::marker::PhantomData {},
        }
    }
}

/// Blanket implementation: any TransformOp with IndexRemapping = True is IndexRemappable
impl<T: Float, U> IndexRemappable<T> for U
where
    U: TransformOp<T>,
    U::IndexRemapping: IsTrue,
{
}

// Type aliases for convenience
pub type PipelineStatic32 = crate::pipeline::PipelineStatic<f32>;
pub type PipelineStatic64 = crate::pipeline::PipelineStatic<f64>;
pub type PipelineDynamic32 = crate::pipeline::PipelineDynamic<f32>;
pub type PipelineDynamic64 = crate::pipeline::PipelineDynamic<f64>;

/// Markers for assuring typestate - internal use only
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct ElementElement;
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct TransformTransform;
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct TransformElement;

/// Generic over the previous operation and current
/// Allows for fusing the operations - internal use only
#[doc(hidden)]
#[derive(Debug, Clone, Copy)]
pub struct Fused<T, S, FusedState = ElementElement> {
    pub(crate) prev_op: T,
    pub(crate) curr_op: S,

    marker: core::marker::PhantomData<FusedState>,
}

impl<T: Float, U: ElementOp<T>, S: ElementOp<T>> ElementOp<T> for Fused<U, S, ElementElement> {
    #[inline(always)]
    fn compute(&self, data: T) -> Result<T, PipeError> {
        let prev = self.prev_op.compute(data)?;
        self.curr_op.compute(prev)
    }

    #[inline(always)]
    fn setup(&self) {
        // Forward the runtime setup to both fused operations
        self.prev_op.setup();
        self.curr_op.setup();
    }

    fn op_name(&self) -> alloc::string::String {
        alloc::format!(
            "Fused<{}, {}>",
            self.prev_op.op_name(),
            self.curr_op.op_name()
        )
    }
}

impl<T: Float, U: ElementOp<T>, S: ElementOp<T>> TransformOp<T> for Fused<U, S, ElementElement> {
    type IndexRemapping = False;

    #[inline(always)]
    fn compute(&self, data: &[T], index: usize) -> Result<T, PipeError> {
        debug_assert!(index < data.len());
        // SAFETY: per the TransformOp contract the caller guarantees
        // `index < out_len(..)` and, for element ops,
        // `out_len(..) == in_len(..) == data.len()`
        let prev = self
            .prev_op
            .compute(unsafe { *data.get_unchecked(index) })?;
        self.curr_op.compute(prev)
    }

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [T],
        input: &'i [T],
        n: usize,
    ) -> Result<&'o mut [T], PipeError> {
        debug_assert!(input.len() >= n && out.len() >= n);
        for out_index in 0..n {
            // SAFETY: the stage guarantees `out.len() == n` and
            // `input.len() == in_len(..) >= n` before dispatching here
            unsafe {
                *out.get_unchecked_mut(out_index) = TransformOp::compute(self, input, out_index)?;
            }
        }
        Ok(out)
    }

    #[inline(always)]
    fn setup(&self) {
        ElementOp::setup(&self.prev_op);
        ElementOp::setup(&self.curr_op);
    }

    fn op_name(&self) -> alloc::string::String {
        alloc::format!(
            "Fused<{}, {}>",
            self.prev_op.op_name(),
            self.curr_op.op_name()
        )
    }
}

impl<T: Float, U: TransformOp<T>, S: ElementOp<T>> TransformOp<T>
    for Fused<U, S, TransformElement>
{
    type IndexRemapping = U::IndexRemapping;

    const IN_LEN: usize = U::IN_LEN;
    const OUT_LEN: usize = U::OUT_LEN;
    const INTERNAL_IS_VALID: bool = U::INTERNAL_IS_VALID;

    #[inline(always)]
    fn compute(&self, data: &[T], index: usize) -> Result<T, PipeError> {
        // Delegates the input access to `prev_op.compute`, which upholds (or
        // checks) its own bound contract
        let prev = self.prev_op.compute(data, index)?;
        self.curr_op.compute(prev)
    }

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [T],
        input: &'i [T],
        n: usize,
    ) -> Result<&'o mut [T], PipeError> {
        debug_assert!(out.len() >= n);
        for out_index in 0..n {
            // SAFETY: `out_index < n == out.len()` (guaranteed by the stage);
            // the input access is performed by `prev_op.compute` under the
            // TransformOp contract (`input.len() == in_len(..)`)
            unsafe {
                *out.get_unchecked_mut(out_index) = TransformOp::compute(self, input, out_index)?;
            }
        }
        Ok(out)
    }

    #[inline(always)]
    fn setup(&self) {
        TransformOp::setup(&self.prev_op);
        ElementOp::setup(&self.curr_op);
    }

    #[inline(always)]
    fn in_len(&self, default_len: usize) -> usize {
        self.prev_op.in_len(default_len)
    }

    #[inline(always)]
    fn out_len(&self, default_len: usize) -> usize {
        self.prev_op.out_len(default_len)
    }

    fn op_name(&self) -> alloc::string::String {
        alloc::format!(
            "Fused<{}, {}>",
            self.prev_op.op_name(),
            self.curr_op.op_name()
        )
    }
}

impl<T: Float, U: TransformOp<T>, S: TransformOp<T>> TransformOp<T>
    for Fused<U, S, TransformTransform>
where
    S::IndexRemapping: IsTrue,
    U::IndexRemapping: IsTrue,
{
    type IndexRemapping = True;

    const OUT_LEN: usize = S::OUT_LEN;
    const IN_LEN: usize = U::IN_LEN;
    const INTERNAL_IS_VALID: bool = S::IN_LEN == U::OUT_LEN || S::IN_LEN == 0 || U::IN_LEN == 0;

    #[inline(always)]
    fn map_index(&self, out_index: usize, default_len: usize) -> usize
    where
        Self::IndexRemapping: IsTrue,
    {
        // The `map_index` contract composes: curr_op maps [0, OUT_LEN) into
        // [0, curr_op::in_len) == [0, prev_op::out_len) (validated via
        // INTERNAL_IS_VALID), which prev_op maps into [0, prev_op::in_len)
        self.curr_op.map_index(out_index, default_len)
    }

    #[inline(always)]
    fn compute(&self, data: &[T], out_index: usize) -> Result<T, PipeError> {
        let intermediate_index = self.map_index(out_index, data.len());
        debug_assert!(self.prev_op.map_index(intermediate_index, data.len()) < data.len());
        // SAFETY: `out_index < out_len(..)` (caller contract) and the
        // composed `map_index` maps into `[0, in_len(..))` where
        // `in_len(..) == data.len()` (stage contract)
        //
        // This is required as the `self.prev_op` might contain a fused
        // element operation which requires mutating the elements
        Ok(self.prev_op.compute(data, intermediate_index)?)
    }

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [T],
        input: &'i [T],
        n: usize,
    ) -> Result<&'o mut [T], PipeError> {
        debug_assert!(out.len() >= n);
        for out_index in 0..n {
            // SAFETY: `out_index < n == out.len()` (guaranteed by the stage);
            // the input access is bounded by the `map_index` contract (see
            // `compute`)
            unsafe {
                *out.get_unchecked_mut(out_index) = self.compute(input, out_index)?;
            }
        }
        Ok(out)
    }

    #[inline(always)]
    fn setup(&self) {
        TransformOp::setup(&self.prev_op);
        TransformOp::setup(&self.curr_op);
    }

    #[inline(always)]
    fn in_len(&self, default_len: usize) -> usize {
        self.prev_op.in_len(default_len)
    }

    #[inline(always)]
    fn out_len(&self, default_len: usize) -> usize {
        self.curr_op.out_len(default_len)
    }

    fn op_name(&self) -> alloc::string::String {
        alloc::format!(
            "Fused<{}, {}>",
            self.prev_op.op_name(),
            self.curr_op.op_name()
        )
    }
}

impl<T: Float, U: ElementOp<T>, const INPUT_LEN: usize> sealed::Sealed<T>
    for Head<U, Element, INPUT_LEN>
{
}

impl<T: Float, U: ElementOp<T>, const INPUT_LEN: usize> Stage<T> for Head<U, Element, INPUT_LEN> {
    const IN_LEN: usize = INPUT_LEN;
    const OUT_LEN: usize = INPUT_LEN;
    const MAX_BUF_SIZE: usize = INPUT_LEN;

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        data: &[T],
        _in_buf: &'i mut [T],
        out_buf: &'o mut [T],
    ) -> Result<&'o [T], PipeError> {
        self.root_op.setup();

        let exec_len = if INPUT_LEN > 0 { INPUT_LEN } else { data.len() };

        if exec_len != data.len() {
            return Err(PipeError::with_snapshot(
                ErrorKind::InvalidInputSize,
                self.snapshot(),
            ));
        }

        // Checked against the *actual* allocation, not only the static
        // constant: a static `build()` guarantees
        // `out_buf.len() >= MAX_BUF_SIZE >= INPUT_LEN`, while a
        // `build_dynamic()` pipe may receive inputs larger than
        // `max_expected_input_length` -- those are rejected here
        if out_buf.len() < exec_len {
            return Err(PipeError::with_snapshot(
                ErrorKind::InvalidOutputSize,
                self.snapshot(),
            ));
        }

        // SAFETY: `i < exec_len`, and both `exec_len == data.len()` and
        // `exec_len <= out_buf.len()` were verified above, so neither the
        // read nor the write can go out of bounds
        for i in 0..exec_len {
            unsafe {
                *out_buf.get_unchecked_mut(i) = self.root_op.compute(*data.get_unchecked(i))?;
            }
        }
        Ok(&out_buf[..exec_len])
    }

    #[inline(always)]
    fn snapshot(&self) -> alloc::string::String {
        alloc::format!("Head<{}>", self.root_op.op_name())
    }

    #[inline(always)]
    fn max_buf_size_dynamic(&self, input_len: usize) -> usize {
        if INPUT_LEN > 0 {
            INPUT_LEN
        } else {
            input_len
        }
    }

    #[inline(always)]
    fn out_len_dynamic(&self, input_len: usize) -> usize {
        if INPUT_LEN > 0 {
            INPUT_LEN
        } else {
            input_len
        }
    }
}

impl<T: Float, U: TransformOp<T>, const INPUT_LEN: usize> sealed::Sealed<T>
    for Head<U, Transform, INPUT_LEN>
{
}

impl<T: Float, U: TransformOp<T>, const INPUT_LEN: usize> Stage<T>
    for Head<U, Transform, INPUT_LEN>
{
    const IN_LEN: usize = U::IN_LEN;
    const OUT_LEN: usize = U::OUT_LEN;
    const MAX_BUF_SIZE: usize = _const_max_usize(INPUT_LEN, U::OUT_LEN);

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        data: &[T],
        _in_buf: &'i mut [T],
        out_buf: &'o mut [T],
    ) -> Result<&'o [T], PipeError> {
        self.root_op.setup();

        let data_len = if Self::IN_LEN > 0 {
            Self::IN_LEN
        } else {
            data.len()
        };

        // Runtime guardrail, optimized if the last operation was dynamic
        if data.len() != data_len {
            return Err(PipeError::with_snapshot(
                ErrorKind::InvalidInputSize,
                self.snapshot(),
            ));
        }

        let expected_in = self.root_op.in_len(data_len);

        // The operation in_len must never be zero as it is
        // either statically or dynamically set
        if expected_in == 0 || data_len != expected_in {
            return Err(PipeError::with_snapshot(
                ErrorKind::InvalidInputSize,
                self.snapshot(),
            ));
        }

        let n = self.root_op.out_len(data_len);

        // `n` is validated against the static bound (when present) *and*
        // against the actual allocation, so the slicing below cannot panic
        // and the unchecked writes inside `TransformOp::execute` stay in
        // bounds
        if n == 0 || (Self::OUT_LEN > 0 && n > Self::OUT_LEN) || n > out_buf.len() {
            return Err(PipeError::with_snapshot(
                ErrorKind::InvalidOutputSize,
                self.snapshot(),
            ));
        }
        let out = &mut out_buf[0..n];

        // The TransformOp contract now holds: `data.len() == in_len(..)`
        // and `out.len() == n == out_len(..)` -- the op's execute may rely
        // on exactly these bounds and must not access anything beyond them
        Ok(self.root_op.execute(out, data, n)?)
    }

    #[inline(always)]
    fn snapshot(&self) -> alloc::string::String {
        alloc::format!("Head<{}>", self.root_op.op_name())
    }

    #[inline(always)]
    fn max_buf_size_dynamic(&self, input_len: usize) -> usize {
        self.root_op.out_len(input_len)
    }

    #[inline(always)]
    fn out_len_dynamic(&self, input_len: usize) -> usize {
        self.root_op.out_len(input_len)
    }
}

impl<T: Float, U: TransformOp<T>, S: Stage<T>> sealed::Sealed<T> for Link<U, S, Transform, T> {}

impl<T: Float, U: TransformOp<T>, S: Stage<T>> Stage<T> for Link<U, S, Transform, T> {
    const IN_LEN: usize = S::OUT_LEN;
    const OUT_LEN: usize = U::OUT_LEN;
    const MAX_BUF_SIZE: usize = _const_max_usize(S::MAX_BUF_SIZE, U::OUT_LEN);

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        data: &[T],
        in_buf: &'i mut [T],
        out_buf: &'o mut [T],
    ) -> Result<&'o [T], PipeError> {
        let prev_out = self.prev_stage.execute(data, out_buf, in_buf)?;
        self.curr_op.setup();

        let data_len = if Self::IN_LEN > 0 {
            Self::IN_LEN
        } else {
            prev_out.len()
        };

        // Runtime guardrail, optimized if the last operation was dynamic
        if prev_out.len() != data_len {
            return Err(PipeError::with_snapshot(
                ErrorKind::InvalidInputSize,
                self.snapshot(),
            ));
        }

        let expected_in = self.curr_op.in_len(data_len);

        // The operation `in_len` must never be zero as it is
        // either statically or dynamically set
        if expected_in == 0 || data_len != expected_in {
            return Err(PipeError::with_snapshot(
                ErrorKind::InvalidInputSize,
                self.snapshot(),
            ));
        }

        let n = self.curr_op.out_len(data_len);

        // `n` is validated against the static bound (when present) *and*
        // against the actual allocation (see Head<_, Transform>)
        if n == 0 || (Self::OUT_LEN > 0 && n > Self::OUT_LEN) || n > out_buf.len() {
            return Err(PipeError::with_snapshot(
                ErrorKind::InvalidOutputSize,
                self.snapshot(),
            ));
        }

        let out = &mut out_buf[0..n];

        // TransformOp contract holds: `prev_out.len() == in_len(..)` and
        // `out.len() == n == out_len(..)`. `prev_out` borrows `in_buf` while
        // `out` borrows `out_buf`; the buffers are distinct allocations, so
        // no aliasing occurs
        Ok(self.curr_op.execute(out, prev_out, n)?)
    }

    #[inline(always)]
    fn snapshot(&self) -> alloc::string::String {
        alloc::format!("Link<{}>", self.curr_op.op_name())
    }

    #[inline(always)]
    fn max_buf_size_dynamic(&self, input_len: usize) -> usize {
        let prev_max = self.prev_stage.max_buf_size_dynamic(input_len);
        let prev_out_len = self.prev_stage.out_len_dynamic(input_len);
        let curr_out = self.curr_op.out_len(prev_out_len);
        core::cmp::max(prev_max, curr_out)
    }

    #[inline(always)]
    fn out_len_dynamic(&self, input_len: usize) -> usize {
        let prev_out_len = self.prev_stage.out_len_dynamic(input_len);
        self.curr_op.out_len(prev_out_len)
    }
}

impl<T: Float, U: ElementOp<T>, S: Stage<T>> sealed::Sealed<T> for Link<U, S, Element, T> {}

impl<T: Float, U: ElementOp<T>, S: Stage<T>> Stage<T> for Link<U, S, Element, T> {
    const IN_LEN: usize = S::OUT_LEN;
    const OUT_LEN: usize = S::OUT_LEN;
    const MAX_BUF_SIZE: usize = S::MAX_BUF_SIZE;

    #[inline(always)]
    fn execute<'i, 'o>(
        &self,
        data: &[T],
        in_buf: &'i mut [T],
        out_buf: &'o mut [T],
    ) -> Result<&'o [T], PipeError> {
        let prev_out = self.prev_stage.execute(data, out_buf, in_buf)?;
        self.curr_op.setup();

        let n = prev_out.len();

        // Checked against the *actual* allocation (see Head<_, Element>)
        if out_buf.len() < n {
            return Err(PipeError::with_snapshot(
                ErrorKind::InvalidOutputSize,
                self.snapshot(),
            ));
        }

        // SAFETY: `i < n`, `n == prev_out.len()` and `n <= out_buf.len()`
        // (checked above), so both accesses stay in bounds. `prev_out`
        // borrows `in_buf` while we write to `out_buf`; the buffers are
        // distinct allocations, so no aliasing occurs
        for i in 0..n {
            unsafe {
                *out_buf.get_unchecked_mut(i) = self.curr_op.compute(*prev_out.get_unchecked(i))?;
            }
        }
        Ok(&out_buf[..n])
    }

    #[inline(always)]
    fn snapshot(&self) -> alloc::string::String {
        alloc::format!("Link<{}>", self.curr_op.op_name())
    }

    #[inline(always)]
    fn max_buf_size_dynamic(&self, input_len: usize) -> usize {
        self.prev_stage.max_buf_size_dynamic(input_len)
    }

    #[inline(always)]
    fn out_len_dynamic(&self, input_len: usize) -> usize {
        self.prev_stage.out_len_dynamic(input_len)
    }
}
