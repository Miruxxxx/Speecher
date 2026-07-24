//! Stateful mono resampling from the device rate down to the ASR rate.
//!
//! Stateful is the whole point: the Python path learned this the hard way
//! (docs/ROADMAP.md, stage 2 — per-chunk `resample_poly` left filter-edge
//! artifacts at every chunk boundary), so one long-lived resampler carries its
//! overlap across calls. WASAPI packets do not divide evenly into the FFT
//! chunk, so leftovers wait here for the next packet.

use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};

pub struct MonoResampler {
    inner: Option<Fft<f32>>,
    pending: Vec<f32>,
    out_buf: Vec<f32>,
}

impl MonoResampler {
    /// Equal rates give a pass-through: no filter, no delay, no allocation.
    pub fn new(input_sr: usize, output_sr: usize) -> Result<Self, String> {
        if input_sr == output_sr {
            return Ok(Self {
                inner: None,
                pending: Vec::new(),
                out_buf: Vec::new(),
            });
        }
        // ~20 ms of input per chunk: small enough that a packet rarely waits
        // for the next one, large enough for a decent anti-aliasing window.
        let chunk = (input_sr / 50).max(64);
        let fft = Fft::<f32>::new(input_sr, output_sr, chunk, 1, FixedSync::Input)
            .map_err(|e| format!("resampler {input_sr}->{output_sr} Hz: {e}"))?;
        let out_buf = vec![0.0; fft.output_frames_max()];
        Ok(Self {
            inner: Some(fft),
            pending: Vec::with_capacity(chunk * 2),
            out_buf,
        })
    }

    /// Feed mono samples at the input rate, append whatever is ready at the
    /// output rate. Anything short of a full chunk stays buffered.
    pub fn push(&mut self, input: &[f32], out: &mut Vec<f32>) -> Result<(), String> {
        let Some(fft) = self.inner.as_mut() else {
            out.extend_from_slice(input);
            return Ok(());
        };
        self.pending.extend_from_slice(input);

        let capacity = self.out_buf.len();
        let mut consumed = 0;
        loop {
            let need = fft.input_frames_next();
            if self.pending.len() - consumed < need {
                break;
            }
            let chunk = &self.pending[consumed..consumed + need];
            let input_adapter = InterleavedSlice::new(chunk, 1, need)
                .map_err(|e| format!("resampler input adapter: {e}"))?;
            let mut output_adapter = InterleavedSlice::new_mut(&mut self.out_buf, 1, capacity)
                .map_err(|e| format!("resampler output adapter: {e}"))?;
            let (frames_in, frames_out) = fft
                .process_into_buffer(&input_adapter, &mut output_adapter, None)
                .map_err(|e| format!("resample failed: {e}"))?;
            out.extend_from_slice(&self.out_buf[..frames_out]);
            consumed += frames_in;
        }
        self.pending.drain(..consumed);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_rates_pass_through_unchanged() {
        let mut r = MonoResampler::new(16000, 16000).unwrap();
        let mut out = Vec::new();
        r.push(&[0.1, -0.2, 0.3], &mut out).unwrap();
        assert_eq!(out, vec![0.1, -0.2, 0.3]);
    }

    #[test]
    fn downsampling_yields_roughly_the_rate_ratio() {
        let mut r = MonoResampler::new(48000, 16000).unwrap();
        let mut out = Vec::new();
        // 1 s of input in 10 ms packets, i.e. packets smaller than the chunk.
        for _ in 0..100 {
            r.push(&vec![0.0; 480], &mut out).unwrap();
        }
        let produced = out.len() as f32;
        assert!(
            (produced - 16000.0).abs() < 1000.0,
            "expected ~16000 output frames, got {produced}"
        );
    }

    #[test]
    fn odd_input_rate_is_supported() {
        let mut r = MonoResampler::new(44100, 16000).unwrap();
        let mut out = Vec::new();
        for _ in 0..100 {
            r.push(&vec![0.0; 441], &mut out).unwrap();
        }
        let produced = out.len() as f32;
        assert!(
            (produced - 16000.0).abs() < 1000.0,
            "expected ~16000 output frames, got {produced}"
        );
    }

    #[test]
    fn a_constant_signal_survives_resampling() {
        let mut r = MonoResampler::new(48000, 16000).unwrap();
        let mut out = Vec::new();
        for _ in 0..50 {
            r.push(&vec![0.5; 960], &mut out).unwrap();
        }
        // Skip the filter warm-up, then the level must be back at the input.
        let tail = &out[out.len() / 2..];
        let mean = tail.iter().sum::<f32>() / tail.len() as f32;
        assert!((mean - 0.5).abs() < 0.01, "mean drifted to {mean}");
    }
}
