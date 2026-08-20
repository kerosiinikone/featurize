use alloc::string::String;
use alloc::vec;
use js_sys::*;

use crate::model::Model;
use crate::state::build_and_load_model;
use burn::tensor::Tensor;
#[cfg(target_family = "wasm")]
use wasm_bindgen::prelude::*;

use featurize_core::prelude::*;

#[cfg(feature = "flex")]
use burn::backend::Flex as Backend;
#[cfg(feature = "wgpu")]
use burn::backend::Wgpu as Backend;

#[cfg_attr(target_family = "wasm", wasm_bindgen(start))]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Mnist structure that corresponds to JavaScript class.
/// See:[exporting-rust-struct](https://rustwasm.github.io/wasm-bindgen/contributing/design/exporting-rust-struct.html)
#[cfg_attr(target_family = "wasm", wasm_bindgen)]
pub struct Mnist {
    model: Option<Model<Backend>>,
    // Type-erased view into the pipeline structure
    preprocessor: BoxedPipeExec<f32>,
}

#[cfg_attr(target_family = "wasm", wasm_bindgen)]
impl Mnist {
    /// Constructor called by JavaScripts with the new keyword.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(constructor))]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();
        Self {
            model: None,
            preprocessor: Pipeline::new()
                .apply_transform(Grayscale::<300, 300, 4, f32>::new().with_inversion())
                .apply_transform(Scale2D::<300, 300, 1, 28, 28, 1, _>::new())
                .apply_element(Div::new(255.0))
                .apply_element(Normalize::new(0.3081, 0.1307))
                .build()
                .boxed(),
        }
    }

    /// Returns the inference results.
    ///
    /// This method is called from JavaScript via generated wrapper code by wasm-bindgen.
    ///
    /// # Arguments
    ///
    /// * `input` - A f32 slice of input 300x300 image
    ///
    /// See bindgen support types for passing and returning arrays:
    /// * [number-slices](https://rustwasm.github.io/wasm-bindgen/reference/types/number-slices.html)
    /// * [boxed-number-slices](https://rustwasm.github.io/wasm-bindgen/reference/types/boxed-number-slices.html)
    ///
    pub async fn inference(&mut self, rgba_data: &[f32]) -> Result<js_sys::Object, String> {
        if self.model.is_none() {
            self.model = Some(build_and_load_model().await);
        }
        let mut preprocessed_data = vec![0.0; 28 * 28];

        if self
            .preprocessor
            .execute(rgba_data, &mut preprocessed_data)
            .is_err()
        {
            return Err("Err at pipeline execution".into());
        }

        let preprocessed_array = Array::new();
        for value in preprocessed_data.iter() {
            preprocessed_array.push(&(*value).into());
        }

        let device = Default::default();

        let model = self.model.as_ref().unwrap();
        let input = Tensor::<Backend, 1>::from_floats(preprocessed_data.as_slice(), &device)
            .reshape([1, 28, 28]);

        let output: Tensor<Backend, 2> = model.forward(input);

        let output = burn::tensor::activation::softmax(output, 1);

        let output = output.into_data_async().await.unwrap();

        let predictions = Array::new();
        for value in output.iter::<f32>() {
            predictions.push(&value.into());
        }

        let result = js_sys::Object::new();
        js_sys::Reflect::set(&result, &"predictions".into(), &predictions).unwrap();
        js_sys::Reflect::set(&result, &"preprocessed".into(), &preprocessed_array).unwrap();

        Ok(result)
    }
}
