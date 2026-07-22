use featurize_core::prelude::*;

fn main() {
    let _pipe = Pipeline::new()
        .apply_transform(Truncate::<10, 6>)
        .apply_transform(Transpose::<2, 4>)
        .build();
}
