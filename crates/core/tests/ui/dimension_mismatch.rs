use featurize_core::prelude::*;

fn main() {
    let _pipe = Pipeline::new()
        .apply_transform(Truncate::<10, 8>)
        .apply_transform(Truncate::<5, 3>)
        .build();
}
