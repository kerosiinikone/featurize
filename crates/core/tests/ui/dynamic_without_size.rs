use featurize_core::prelude::*;

fn main() {
    let _pipe = Pipeline::new_with_dynamic()
        .apply_point(Multiply { factor: 2.0, ..Default::default() })
        .build();
}
