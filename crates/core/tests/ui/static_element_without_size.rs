use featurize_core::prelude::*;

fn main() {
    let _pipe = Pipeline::new()
        .apply_point(Multiply { factor: 2.0, ..Default::default() })
        .build();
}
