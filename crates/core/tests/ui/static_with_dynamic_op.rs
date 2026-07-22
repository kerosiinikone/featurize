use featurize_core::prelude::*;

struct DynamicIdentity;

impl TransformOp for DynamicIdentity {
    type IndexRemapping = False;
    const IN_LEN: usize = 0;
    const OUT_LEN: usize = 0;
    
    fn execute<'i, 'o>(
        &self,
        out: &'o mut [f32],
        input: &'i [f32],
        n: usize,
    ) -> Result<&'o mut [f32], PipeError> {
        out[..n].copy_from_slice(&input[..n]);
        Ok(out)
    }
}

fn main() {
    let _pipe = Pipeline::new()
        .apply_transform(DynamicIdentity)
        .build();
}
