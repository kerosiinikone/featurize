pub fn add(left: u64, right: u64) -> u64 {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn process_image_data(data: &[f32]) -> Vec<f32> {
    // Placeholder function to prove the WASM boundary works
    data.to_vec()
}
