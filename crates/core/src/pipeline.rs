use crate::{
    errors::{FailOnNan, NanHandler, NanHandling, PipeError},
    traits::{
        Element, ElementElement, ElementOp, Float, Fused, Head, IndexRemappable, IsTrue,
        Link, Stage, Transform, TransformElement, TransformOp, TransformTransform,
    },
};

/// Markers for pipeline type - internal use only
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Static;
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Dynamic;

/// Initializer
#[derive(Debug, Default, Clone)]
pub struct Pipeline;

/// Wrapper for the pipeline stages
///
/// Rotates a set of temporary buffers to avoid allocating mid pipe
///
/// The `N` type parameter is the pipeline-wide NaN policy
/// ([`NanHandler`]); it is chosen once at construction time and is threaded
/// through every stage / operation so the computation loops are
/// monomorphized for exactly one check.
#[derive(Debug, Default, Clone)]
pub struct Pipe<T, F, State, N = FailOnNan>
where
    T: Stage<F>,
    F: Float,
    N: NanHandler,
{
    // Pipe typestate
    // Accessing the underlying T -> helper methods
    pub(crate) stages: T,
    pub(crate) marker: core::marker::PhantomData<(State, F, N)>,
}

/// Executable pipeline with allocated buffers
///
/// Scratch buffers `in_buf` and `out_buf` handle the allocations
/// needed between `Link<...>` stages as here the original input
/// buffer cannot be reused
#[derive(Debug, Default, Clone)]
pub struct PipeExec<T, N = FailOnNan, F = f32>
where
    T: Stage<F>,
    F: Float,
    N: NanHandler,
{
    // Pipe typestate
    pub(crate) stages: T,
    #[allow(dead_code)]
    pub(crate) max_expected_input_length: Option<usize>,
    // These are allocated only at the build stage
    pub(crate) in_buf: alloc::boxed::Box<[F]>,
    pub(crate) out_buf: alloc::boxed::Box<[F]>,

    pub(crate) marker: core::marker::PhantomData<N>,
}

/// Object-safe, type-erased view of a built pipeline executor
///
/// The concrete type of a `PipeExec` encodes the full stage composition.
/// `PipeExecutor` erases that type behind a vtable while keeping the
/// pipeline itself fully static and monomorphized internally: the only
/// runtime cost is a single dynamic dispatch per `execute` call. Note that
/// the NaN policy is *also* baked in at monomorphization time, so erasing
/// the type does not reintroduce any per-element branching.
///
/// Use `PipeExec::boxed` (or the `BoxedPipeExec` alias) to obtain one:
///
/// ```ignore
/// let preprocessor: BoxedPipeExec = Pipeline::new()
///     .apply_transform(...)
///     .apply_element(...)
///     .build()
///     .boxed();
/// ```
///
/// [`execute`]: PipeExecutor::execute
//
pub trait PipeExecutor<F: Float = f32> {
    /// See [`PipeExec::execute`]
    fn execute(&mut self, input: &[F], output_buf: &mut [F]) -> Result<usize, PipeError>;

    /// See [`PipeExec::output_len`]
    fn output_len(&self) -> usize;

    /// The (compile-time selected) NaN policy of this pipeline
    fn nan_handling(&self) -> NanHandling;
}

/// Convenience alias for a boxed, type-erased pipeline executor
pub type BoxedPipeExec<F = f32> = alloc::boxed::Box<dyn PipeExecutor<F>>;

/// Blanket implementation: every built pipeline executor is a
/// (type-erasable) `PipeExecutor`
impl<T, N, F> PipeExecutor<F> for PipeExec<T, N, F>
where
    T: Stage<F>,
    F: Float,
    N: NanHandler,
{
    #[inline(always)]
    fn execute(&mut self, input: &[F], output_buf: &mut [F]) -> Result<usize, PipeError> {
        PipeExec::execute(self, input, output_buf)
    }

    #[inline(always)]
    fn output_len(&self) -> usize {
        PipeExec::output_len(self)
    }

    #[inline(always)]
    fn nan_handling(&self) -> NanHandling {
        N::HANDLING
    }
}

/// Must be documented why the pipeline struct constructors
/// return a different type with each method.
impl Pipeline {
    /// Static pipeline with the default (fail fast) NaN policy
    pub fn new<T: Float>() -> PipelineStatic<T, FailOnNan> {
        PipelineStatic {
            marker: core::marker::PhantomData,
        }
    }

    /// Static pipeline with an explicit, pipeline-wide NaN policy
    ///
    /// ```ignore
    /// let pipe = Pipeline::new_with::<f32, ZeroOnNan>()
    ///     .apply_element::<_, 128>(Div::new(255.0))
    ///     .build();
    /// ```
    pub fn new_with<T: Float, N: NanHandler>() -> PipelineStatic<T, N> {
        PipelineStatic {
            marker: core::marker::PhantomData,
        }
    }

    /// Dynamic pipeline with the default (fail fast) NaN policy
    pub fn with_dynamic<T: Float>() -> PipelineDynamic<T, FailOnNan> {
        PipelineDynamic {
            marker: core::marker::PhantomData,
        }
    }

    /// Dynamic pipeline with an explicit, pipeline-wide NaN policy
    pub fn with_dynamic_and<T: Float, N: NanHandler>() -> PipelineDynamic<T, N> {
        PipelineDynamic {
            marker: core::marker::PhantomData,
        }
    }
}

/// Static pipeline initializer - internal use only
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PipelineStatic<F: Float = f32, N: NanHandler = FailOnNan> {
    pub(crate) marker: core::marker::PhantomData<(Static, F, N)>,
}

/// Dynamic pipeline initializer - internal use only
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PipelineDynamic<F: Float = f32, N: NanHandler = FailOnNan> {
    pub(crate) marker: core::marker::PhantomData<(Dynamic, F, N)>,
}

impl<F: Float, N: NanHandler> PipelineStatic<F, N> {
    /// Select the pipeline-wide NaN policy before adding any operation
    pub fn nan_handling<N2: NanHandler>(self) -> PipelineStatic<F, N2> {
        PipelineStatic {
            marker: core::marker::PhantomData,
        }
    }

    /// Creates the initialized pipe implicitly (with size)
    pub fn apply_element<T, const INPUT_LEN: usize>(
        self,
        op: T,
    ) -> Pipe<Head<T, Element, INPUT_LEN>, F, Static, N>
    where
        T: ElementOp<F>,
    {
        Pipe {
            stages: Head {
                root_op: op,
                marker: core::marker::PhantomData {},
            },
            marker: core::marker::PhantomData {},
        }
    }

    /// Creates the initialized pipe implicitly (with size implied -> transform operation must be
    /// static!)
    pub fn apply_transform<T>(self, op: T) -> Pipe<Head<T, Transform>, F, Static, N>
    where
        T: TransformOp<F>,
    {
        const {
            assert!(T::IN_LEN > 0 && T::OUT_LEN > 0, "Dynamic bounds");
            // Ops with an internal consistency requirement (e.g. Truncate:
            // NEW_LEN <= ORIGINAL_LEN) rely on this being asserted at every
            // construction site: unchecked reads in their `execute` cite it.
            assert!(T::INTERNAL_IS_VALID, "Invalid transform bounds");
        }

        Pipe {
            stages: Head {
                root_op: op,
                marker: core::marker::PhantomData {},
            },
            marker: core::marker::PhantomData {},
        }
    }
}

impl<F: Float, N: NanHandler> PipelineDynamic<F, N> {
    /// Select the pipeline-wide NaN policy before adding any operation
    pub fn nan_handling<N2: NanHandler>(self) -> PipelineDynamic<F, N2> {
        PipelineDynamic {
            marker: core::marker::PhantomData,
        }
    }

    /// Creates the initialized pipe implicitly
    pub fn apply_element<T>(self, op: T) -> Pipe<Head<T, Element>, F, Dynamic, N>
    where
        T: ElementOp<F>,
    {
        Pipe {
            stages: Head {
                root_op: op,
                marker: core::marker::PhantomData {},
            },
            marker: core::marker::PhantomData {},
        }
    }

    /// Creates the initialized pipe implicitly
    pub fn apply_transform<T>(self, op: T) -> Pipe<Head<T, Transform>, F, Dynamic, N>
    where
        T: TransformOp<F>,
    {
        const {
            // Even in a dynamic pipe the *internal* consistency of an op is
            // known at compile time (it only depends on its own const
            // generics) -- assert it here so ops like `Truncate` can rely on
            // it in their unchecked `execute` paths.
            assert!(T::INTERNAL_IS_VALID, "Invalid transform bounds");
        }

        Pipe {
            stages: Head {
                root_op: op,
                marker: core::marker::PhantomData {},
            },
            marker: core::marker::PhantomData {},
        }
    }
}

impl<T, F, N> Pipe<T, F, Static, N>
where
    T: Stage<F>,
    F: Float,
    N: NanHandler,
{
    // The input data dims must be known at this point
    //
    // Buffer-size invariant: `MAX_BUF_SIZE` is the maximum of every stage's
    // static output length (and the head's input length), so each stage's
    // intermediate result fits into either scratch buffer. The `Stage`
    // implementations additionally re-check against the *actual* buffer
    // length at runtime before entering any unchecked loop.
    pub fn build(self) -> PipeExec<T, N, F> {
        const {
            assert!(T::MAX_BUF_SIZE > 0, "Pipeline has dynamic bounds");
        }
        let buf_size = T::MAX_BUF_SIZE;

        PipeExec {
            in_buf: alloc::vec![F::zero(); buf_size].into_boxed_slice(),
            out_buf: alloc::vec![F::zero(); buf_size].into_boxed_slice(),
            stages: self.stages,
            max_expected_input_length: None,
            marker: core::marker::PhantomData,
        }
    }
}

impl<T, F, N> Pipe<T, F, Dynamic, N>
where
    T: Stage<F>,
    F: Float,
    N: NanHandler,
{
    /// Dynamic bound checking
    ///
    /// Buffers are sized for `max_expected_input_length` by walking the
    /// stages with `max_buf_size_dynamic`. Note that this implicitly assumes
    /// an op's `out_len()` is monotonic in its input length; if an actual
    /// input (smaller or larger than the expected maximum) produces an
    /// intermediate result that exceeds the allocation, the `Stage`
    /// implementations reject it at runtime with `InvalidInputSize` /
    /// `InvalidOutputSize` -- it can never lead to out-of-bounds access.
    pub fn build_dynamic(self, max_expected_input_length: usize) -> PipeExec<T, N, F> {
        let buf_size = self.stages.max_buf_size_dynamic(max_expected_input_length);

        PipeExec {
            in_buf: alloc::vec![F::zero(); buf_size].into_boxed_slice(),
            out_buf: alloc::vec![F::zero(); buf_size].into_boxed_slice(),
            stages: self.stages,
            max_expected_input_length: Some(max_expected_input_length),
            marker: core::marker::PhantomData,
        }
    }
}

impl<T, N, F> PipeExec<T, N, F>
where
    T: Stage<F>,
    F: Float,
    N: NanHandler,
{
    pub fn execute(&mut self, input: &[F], output_buf: &mut [F]) -> Result<usize, PipeError> {
        let out_buf = &mut self.out_buf;
        let in_buf = &mut self.in_buf;

        // Every stage validates its input length and the actual scratch
        // buffer capacity before running its (unchecked) computation loop.
        //
        // `N` is the pipeline-wide, compile-time NaN policy: the whole stage
        // tree below is monomorphized for it, so only the selected check is
        // emitted inside the loops.
        let exec = self.stages.execute::<N>(input, in_buf, out_buf)?;
        let exec_len = exec.len();

        if output_buf.len() < exec_len {
            return Err(PipeError::new(crate::errors::ErrorKind::InvalidOutputSize));
        }

        output_buf[..exec_len].copy_from_slice(exec);
        Ok(exec_len)
    }

    pub fn execute_from_bytes(
        &mut self,
        input: &[u8],
        output_buf: &mut [F],
    ) -> Result<usize, PipeError>
    where
        F: bytemuck::Pod,
    {
        let floats: &[F] = bytemuck::try_cast_slice(input)?;
        self.execute(floats, output_buf)
    }

    pub fn output_len(&self) -> usize {
        self.stages
            .out_len_dynamic(self.max_expected_input_length.unwrap_or(0))
    }

    /// The (compile-time selected) NaN policy of this pipeline
    #[inline(always)]
    pub fn nan_handling(&self) -> NanHandling {
        N::HANDLING
    }

    /// Erase the concrete pipeline type behind a [`PipeExecutor`] vtable.
    ///
    /// This is the recommended way to *store* a built pipeline (e.g. in a
    /// struct field of a consumer crate) without naming the deeply nested
    /// stage type. The pipeline remains fully static internally; only the
    /// entry point is dynamically dispatched.
    pub fn boxed(self) -> BoxedPipeExec<F>
    where
        T: 'static,
    {
        alloc::boxed::Box::new(self)
    }
}

impl<T, F, N, const INPUT_LEN: usize, State> Pipe<Head<T, Element, INPUT_LEN>, F, State, N>
where
    T: ElementOp<F>,
    F: Float,
    N: NanHandler,
{
    pub fn apply_element<U>(
        self,
        op: U,
    ) -> Pipe<Head<Fused<T, U, ElementElement>, Element, INPUT_LEN>, F, State, N>
    where
        U: ElementOp<F>,
    {
        let prev_head = self.stages;

        Pipe {
            stages: Head {
                marker: prev_head.marker,
                root_op: prev_head.root_op.fuse_element(op),
            },
            marker: core::marker::PhantomData {},
        }
    }

    pub fn apply_transform<U>(
        self,
        op: U,
    ) -> Pipe<Head<Fused<U, T, TransformElement>, Transform, INPUT_LEN>, F, State, N>
    where
        U: TransformOp<F>,
    {
        const {
            assert!(U::INTERNAL_IS_VALID, "Invalid transform bounds");
            // The element head's static length (when present) must match the
            // transform's static input length (when present)
            assert!(
                INPUT_LEN == 0 || U::IN_LEN == 0 || U::IN_LEN == INPUT_LEN,
                "Invalid input length"
            );
        }
        let prev_head = self.stages;

        Pipe {
            stages: Head {
                root_op: prev_head.root_op.fuse_transform(op),
                marker: core::marker::PhantomData {},
            },
            marker: core::marker::PhantomData {},
        }
    }
}

impl<T, F, N, const INPUT_LEN: usize, State> Pipe<Head<T, Transform, INPUT_LEN>, F, State, N>
where
    T: TransformOp<F>,
    F: Float,
    N: NanHandler,
{
    pub fn apply_element<U>(
        self,
        op: U,
    ) -> Pipe<Head<Fused<T, U, TransformElement>, Transform, INPUT_LEN>, F, State, N>
    where
        U: ElementOp<F>,
    {
        let prev_head = self.stages;

        Pipe {
            stages: Head {
                marker: core::marker::PhantomData {},
                root_op: prev_head.root_op.fuse_element(op),
            },
            marker: core::marker::PhantomData {},
        }
    }

    pub fn apply_transform<U>(
        self,
        op: U,
    ) -> Pipe<Link<U, Head<T, Transform, INPUT_LEN>, Transform, F>, F, State, N>
    where
        U: TransformOp<F>,
    {
        const {
            assert!(
                U::INTERNAL_IS_VALID
                    && (T::OUT_LEN == U::IN_LEN || T::OUT_LEN == 0 || U::IN_LEN == 0),
                "Invalid input length"
            );
        }
        let prev_head = self.stages;

        Pipe {
            stages: Link {
                curr_op: op,
                prev_stage: prev_head,
                marker: core::marker::PhantomData {},
            },
            marker: core::marker::PhantomData {},
        }
    }
}

impl<T, F, N, const INPUT_LEN: usize, State> Pipe<Head<T, Transform, INPUT_LEN>, F, State, N>
where
    T: TransformOp<F> + IndexRemappable<F>,
    T::IndexRemapping: IsTrue,
    F: Float,
    N: NanHandler,
{
    pub fn apply_transform_fusable<U>(
        self,
        op: U,
    ) -> Pipe<Head<Fused<T, U, TransformTransform>, Transform, INPUT_LEN>, F, State, N>
    where
        U: TransformOp<F>,
        U::IndexRemapping: IsTrue,
    {
        const {
            assert!(
                U::INTERNAL_IS_VALID
                    && (T::OUT_LEN == U::IN_LEN || T::OUT_LEN == 0 || U::IN_LEN == 0),
                "Invalid input length"
            );
        }
        let prev_head = self.stages;

        Pipe {
            stages: Head {
                marker: core::marker::PhantomData {},
                root_op: prev_head.root_op.fuse_transform(op),
            },
            marker: core::marker::PhantomData {},
        }
    }
}

impl<T, S, F, N, State> Pipe<Link<T, S, Element, F>, F, State, N>
where
    T: ElementOp<F>,
    S: Stage<F>,
    F: Float,
    N: NanHandler,
{
    pub fn apply_element<U>(self, op: U) -> Pipe<Link<Fused<T, U>, S, Element, F>, F, State, N>
    where
        U: ElementOp<F>,
    {
        let stages = self.stages;

        Pipe {
            stages: Link {
                prev_stage: stages.prev_stage,
                curr_op: stages.curr_op.fuse_element(op),
                marker: core::marker::PhantomData {},
            },
            marker: core::marker::PhantomData {},
        }
    }

    pub fn apply_transform<U>(
        self,
        op: U,
    ) -> Pipe<Link<Fused<U, T, TransformElement>, S, Transform, F>, F, State, N>
    where
        U: TransformOp<F>,
    {
        const {
            assert!(
                U::INTERNAL_IS_VALID
                    && (S::OUT_LEN == U::IN_LEN || S::OUT_LEN == 0 || U::IN_LEN == 0),
                "Invalid input length"
            );
        }
        let stages = self.stages;

        Pipe {
            stages: Link {
                prev_stage: stages.prev_stage,
                curr_op: stages.curr_op.fuse_transform(op),
                marker: core::marker::PhantomData {},
            },
            marker: core::marker::PhantomData {},
        }
    }
}

impl<T, S, F, N, State> Pipe<Link<T, S, Transform, F>, F, State, N>
where
    T: TransformOp<F>,
    S: Stage<F>,
    F: Float,
    N: NanHandler,
{
    pub fn apply_element<U>(
        self,
        op: U,
    ) -> Pipe<Link<Fused<T, U, TransformElement>, S, Transform, F>, F, State, N>
    where
        U: ElementOp<F>,
    {
        let stages = self.stages;

        Pipe {
            stages: Link {
                prev_stage: stages.prev_stage,
                curr_op: stages.curr_op.fuse_element(op),
                marker: core::marker::PhantomData {},
            },
            marker: core::marker::PhantomData {},
        }
    }

    pub fn apply_transform<U>(
        self,
        op: U,
    ) -> Pipe<Link<U, Link<T, S, Transform, F>, Transform, F>, F, State, N>
    where
        U: TransformOp<F>,
    {
        const {
            assert!(
                U::INTERNAL_IS_VALID
                    && (T::OUT_LEN == U::IN_LEN || T::OUT_LEN == 0 || U::IN_LEN == 0),
                "Invalid input length"
            );
        }
        let stages = self.stages;

        Pipe {
            stages: Link {
                prev_stage: stages,
                curr_op: op,
                marker: core::marker::PhantomData {},
            },
            marker: core::marker::PhantomData {},
        }
    }
}

impl<T, S, F, N, State> Pipe<Link<T, S, Transform, F>, F, State, N>
where
    T: TransformOp<F> + IndexRemappable<F>,
    T::IndexRemapping: IsTrue,
    S: Stage<F>,
    F: Float,
    N: NanHandler,
{
    pub fn apply_transform_fusable<U>(
        self,
        op: U,
    ) -> Pipe<Link<Fused<T, U, TransformTransform>, S, Transform, F>, F, State, N>
    where
        U: TransformOp<F>,
        U::IndexRemapping: IsTrue,
    {
        const {
            assert!(
                U::INTERNAL_IS_VALID
                    && (T::OUT_LEN == U::IN_LEN || T::OUT_LEN == 0 || U::IN_LEN == 0),
                "Invalid input length"
            );
        }
        let stages = self.stages;

        Pipe {
            stages: Link {
                marker: core::marker::PhantomData {},
                curr_op: stages.curr_op.fuse_transform(op),
                prev_stage: stages.prev_stage,
            },
            marker: core::marker::PhantomData {},
        }
    }
}
