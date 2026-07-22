use featurize_core::prelude::*;

fn main() {
    let _pipe = Pipeline::new()
        .apply_transform(Reverse::<10>)
        .apply_transform_fusable(Truncate::<10, 8>)
        .apply_transform_fusable(Reverse::<5>)
        .build();
}
