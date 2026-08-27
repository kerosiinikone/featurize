use crate::model::Model;
use burn::{
    module::Module,
    prelude::Device,
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
};

#[cfg(feature = "flex")]
use burn::backend::Flex as Backend;
#[cfg(feature = "wgpu")]
use burn::backend::Wgpu as Backend;

// Trained parameters in the burnpack format, produced by the `mnist` example
// (`model.into_record().save(..)`) and copied here. Regenerate with the same command if the
// model architecture changes.
static STATE_ENCODED: &[u8] = include_bytes!("../model.bin");

/// Builds and loads trained parameters into the model.
pub async fn build_and_load_model() -> Model<Backend> {
    #[cfg(feature = "flex")]
    let device: Device<Backend> = Default::default();
    #[cfg(feature = "wgpu")]
    let device = Device::<Backend>::wgpu_async(Default::default()).await;

    let model = Model::new(&device);
    let record = BinBytesRecorder::<FullPrecisionSettings>::default()
        .load(STATE_ENCODED.to_vec(), &device)
        .expect("Failed to load model record");

    model.load_record(record)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_model_decodes_into_architecture() {
        let device: Device<Backend> = Default::default();
        let model = Model::<Backend>::new(&device);
        // `load_record` validates that every model parameter is present with a matching shape; a
        // stale/mismatched asset would panic here.

        let record = BinBytesRecorder::<FullPrecisionSettings>::default()
            .load(STATE_ENCODED.to_vec(), &device)
            .expect("Failed to load model record");

        let _model = model.load_record(record);
    }
}
