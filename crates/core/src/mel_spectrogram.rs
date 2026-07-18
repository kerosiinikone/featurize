use crate::traits::{False, TransformOp};
use alloc::vec;
use alloc::vec::Vec;

#[allow(unused_imports)]
use num_traits::Float as _;

pub trait Float: num_traits::Float + num_traits::FloatConst + num_traits::NumAssign {}

impl Float for f32 {}
impl Float for f64 {}

// https://github.com/ggerganov/whisper.cpp/blob/4774d2feb01a772a15de81ffc34b34a1f294f020/whisper.cpp#L2357
fn fft<T: Float>(inp: &[T]) -> Vec<T> {
    let n = inp.len();
    let zero = T::zero();
    if n == 1 {
        return vec![inp[0], zero];
    }
    if n % 2 == 1 {
        return dft(inp);
    }
    let mut out = vec![zero; n * 2];

    let mut even = Vec::with_capacity(n / 2);
    let mut odd = Vec::with_capacity(n / 2);

    for (i, &inp) in inp.iter().enumerate() {
        if i % 2 == 0 {
            even.push(inp)
        } else {
            odd.push(inp);
        }
    }

    let even_fft = fft(&even);
    let odd_fft = fft(&odd);

    let two_pi = T::PI() + T::PI();
    let n_t = T::from(n).unwrap();
    for k in 0..n / 2 {
        let k_t = T::from(k).unwrap();
        let theta = two_pi * k_t / n_t;
        let re = theta.cos();
        let im = -theta.sin();

        let re_odd = odd_fft[2 * k];
        let im_odd = odd_fft[2 * k + 1];

        out[2 * k] = even_fft[2 * k] + re * re_odd - im * im_odd;
        out[2 * k + 1] = even_fft[2 * k + 1] + re * im_odd + im * re_odd;

        out[2 * (k + n / 2)] = even_fft[2 * k] - re * re_odd + im * im_odd;
        out[2 * (k + n / 2) + 1] = even_fft[2 * k + 1] - re * im_odd - im * re_odd;
    }
    out
}

// https://github.com/ggerganov/whisper.cpp/blob/4774d2feb01a772a15de81ffc34b34a1f294f020/whisper.cpp#L2337
fn dft<T: Float>(inp: &[T]) -> Vec<T> {
    let zero = T::zero();
    let n = inp.len();
    let two_pi = T::PI() + T::PI();

    let mut out = Vec::with_capacity(2 * n);
    let n_t = T::from(n).unwrap();
    for k in 0..n {
        let k_t = T::from(k).unwrap();
        let mut re = zero;
        let mut im = zero;

        for (j, &inp) in inp.iter().enumerate() {
            let j_t = T::from(j).unwrap();
            let angle = two_pi * k_t * j_t / n_t;
            re += inp * angle.cos();
            im -= inp * angle.sin();
        }

        out.push(re);
        out.push(im);
    }
    out
}

#[allow(clippy::too_many_arguments)]
// https://github.com/ggerganov/whisper.cpp/blob/4774d2feb01a772a15de81ffc34b34a1f294f020/whisper.cpp#L2414
fn log_mel_spectrogram_w<T: Float>(
    ith: usize,
    hann: &[T],
    samples: &[T],
    filters: &[T],
    fft_size: usize,
    fft_step: usize,
    speed_up: bool,
    n_len: usize,
    n_mel: usize,
    n_threads: usize,
) -> Vec<T> {
    let n_fft = if speed_up {
        1 + fft_size / 4
    } else {
        1 + fft_size / 2
    };

    let zero = T::zero();
    let half = T::from(0.5).unwrap();
    let mut fft_in = vec![zero; fft_size];
    let mut mel = vec![zero; n_len * n_mel];

    for i in (ith..n_len).step_by(n_threads) {
        let offset = i * fft_step;

        // apply Hanning window
        for j in 0..fft_size {
            fft_in[j] = if offset + j < samples.len() {
                hann[j] * samples[offset + j]
            } else {
                zero
            }
        }

        // FFT -> mag^2
        let mut fft_out: Vec<T> = fft(&fft_in);

        for j in 0..fft_size {
            fft_out[j] = fft_out[2 * j] * fft_out[2 * j] + fft_out[2 * j + 1] * fft_out[2 * j + 1];
        }
        for j in 1..fft_size / 2 {
            let v = fft_out[fft_size - j];
            fft_out[j] += v;
        }

        if speed_up {
            // scale down in the frequency domain results in a speed up in the time domain
            for j in 0..n_fft {
                fft_out[j] = half * (fft_out[2 * j] + fft_out[2 * j + 1]);
            }
        }

        // mel spectrogram
        for j in 0..n_mel {
            let mut sum = zero;
            for k in 0..n_fft {
                sum += fft_out[k] * filters[j * n_fft + k];
            }
            mel[j * n_len + i] = T::max(sum, T::from(1e-10).unwrap()).log10();
        }
    }
    mel
}

/// Log Mel Spectrogram transformation operation
/// Converts audio samples to mel spectrogram representation
#[derive(Debug, Clone)]
pub struct LogMelSpectrogram<const N_FFT: usize, const HOP_LENGTH: usize, const N_MEL: usize> {
    pub filters: Vec<f32>,
    pub speed_up: bool,
    pub pad_chunk_length: usize,
    hann_window: Vec<f32>,
    // TODO: static size at comptime -> const generic
    output_len: usize,
}

impl<const N_FFT: usize, const HOP_LENGTH: usize, const N_MEL: usize>
    LogMelSpectrogram<N_FFT, HOP_LENGTH, N_MEL>
{
    pub fn new(
        filters: Vec<f32>,
        speed_up: bool,
        pad_chunk_length: usize,
        input_len: usize,
    ) -> Self {
        let half = 0.5f32;
        let one = 1.0f32;
        let two_pi = core::f32::consts::PI + core::f32::consts::PI;
        let fft_size_t = N_FFT as f32;

        let hann_window: Vec<f32> = (0..N_FFT)
            .map(|i| half * (one - ((two_pi * i as f32) / fft_size_t).cos()))
            .collect();

        let n_len = input_len / HOP_LENGTH;
        let pad = 100 * pad_chunk_length / 2;
        let n_len = if !n_len.is_multiple_of(pad) {
            (n_len / pad + 1) * pad
        } else {
            n_len
        };
        let n_len = n_len + pad;
        let output_len = n_len * N_MEL;

        Self {
            filters,
            speed_up,
            pad_chunk_length,
            hann_window,
            output_len,
        }
    }

    fn compute_mel_spectrogram(&self, samples: &[f32]) -> Vec<f32> {
        let zero = 0.0f32;
        let n_len = samples.len() / HOP_LENGTH;
        let pad = 100 * self.pad_chunk_length / 2;
        let n_len = if !n_len.is_multiple_of(pad) {
            (n_len / pad + 1) * pad
        } else {
            n_len
        };
        let n_len = n_len + pad;

        // Pad samples
        let samples_padded = {
            let mut padded = samples.to_vec();
            let to_add = n_len * HOP_LENGTH - samples.len();
            padded.extend(core::iter::repeat(zero).take(to_add));
            padded
        };

        // Use a single thread for now
        let mut mel = log_mel_spectrogram_w(
            0,
            &self.hann_window,
            &samples_padded,
            &self.filters,
            N_FFT,
            HOP_LENGTH,
            self.speed_up,
            n_len,
            N_MEL,
            1,
        );

        // Normalize
        let mmax = mel
            .iter()
            .max_by(|&u, &v| u.partial_cmp(v).unwrap_or(core::cmp::Ordering::Greater))
            .copied()
            .unwrap_or(zero)
            - 8.0;

        for m in mel.iter_mut() {
            let v = f32::max(*m, mmax);
            *m = v / 4.0 + 1.0;
        }

        mel
    }
}

impl<const N_FFT: usize, const HOP_LENGTH: usize, const N_MEL: usize> TransformOp
    for LogMelSpectrogram<N_FFT, HOP_LENGTH, N_MEL>
{
    type IndexRemapping = False;

    #[inline(always)]
    fn execute<'i, 'o>(&self, out: &'o mut [f32], input: &'i [f32], _n: usize) -> &'o mut [f32] {
        let mel = self.compute_mel_spectrogram(input);
        let copy_len = mel.len().min(out.len());
        out[..copy_len].copy_from_slice(&mel[..copy_len]);
        out
    }

    #[inline(always)]
    fn buffer_size(&self) -> usize {
        self.output_len
    }

    #[inline(always)]
    fn output_shape(&self) -> usize {
        self.output_len
    }
}
