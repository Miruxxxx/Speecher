//! WASAPI frames -> mono f32, the only shape the ASR path cares about.
//!
//! Shared mode hands us the device mix format as-is (no AUTOCONVERTPCM), so
//! the sample type is whatever the endpoint runs at: float32 on nearly every
//! modern device, but 16/24/32-bit PCM still shows up on older drivers and on
//! virtual cables. Downmixing is a plain channel average, matching what the
//! Python supervisor did with `numpy.mean(axis=1)`.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleFormat {
    F32,
    I16,
    I24,
    I32,
}

impl SampleFormat {
    pub const fn bytes(self) -> usize {
        match self {
            SampleFormat::I16 => 2,
            SampleFormat::I24 => 3,
            SampleFormat::I32 | SampleFormat::F32 => 4,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            SampleFormat::F32 => "f32",
            SampleFormat::I16 => "i16",
            SampleFormat::I24 => "i24",
            SampleFormat::I32 => "i32",
        }
    }
}

fn sample_at(bytes: &[u8], offset: usize, fmt: SampleFormat) -> f32 {
    match fmt {
        SampleFormat::F32 => f32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ]),
        SampleFormat::I16 => {
            let v = i16::from_le_bytes([bytes[offset], bytes[offset + 1]]);
            v as f32 / 32768.0
        }
        SampleFormat::I24 => {
            // Sign-extend the packed 3-byte sample into an i32.
            let v = ((bytes[offset] as i32) << 8)
                | ((bytes[offset + 1] as i32) << 16)
                | ((bytes[offset + 2] as i32) << 24);
            (v >> 8) as f32 / 8_388_608.0
        }
        SampleFormat::I32 => {
            let v = i32::from_le_bytes([
                bytes[offset],
                bytes[offset + 1],
                bytes[offset + 2],
                bytes[offset + 3],
            ]);
            v as f32 / 2_147_483_648.0
        }
    }
}

/// Append `bytes` (interleaved, `channels` channels of `fmt`) to `out` as mono.
///
/// A trailing partial frame is ignored rather than guessed at; WASAPI hands
/// out whole frames, so this only guards against a caller slicing badly.
pub fn decode_mono(bytes: &[u8], fmt: SampleFormat, channels: usize, out: &mut Vec<f32>) {
    if channels == 0 {
        return;
    }
    let frame_bytes = fmt.bytes() * channels;
    let frames = bytes.len() / frame_bytes;
    out.reserve(frames);
    for f in 0..frames {
        let base = f * frame_bytes;
        let mut acc = 0.0f32;
        for c in 0..channels {
            acc += sample_at(bytes, base + c * fmt.bytes(), fmt);
        }
        out.push(acc / channels as f32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_stereo_is_averaged() {
        let mut bytes = Vec::new();
        for s in [1.0f32, 0.0, -0.5, 0.5] {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let mut out = Vec::new();
        decode_mono(&bytes, SampleFormat::F32, 2, &mut out);
        assert_eq!(out, vec![0.5, 0.0]);
    }

    #[test]
    fn i16_is_scaled_to_unit_range() {
        let mut bytes = Vec::new();
        for s in [i16::MIN, 0, 16384] {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let mut out = Vec::new();
        decode_mono(&bytes, SampleFormat::I16, 1, &mut out);
        assert_eq!(out, vec![-1.0, 0.0, 0.5]);
    }

    #[test]
    fn i32_is_scaled_to_unit_range() {
        let mut bytes = Vec::new();
        for s in [i32::MIN, 0, 1 << 30] {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let mut out = Vec::new();
        decode_mono(&bytes, SampleFormat::I32, 1, &mut out);
        assert_eq!(out, vec![-1.0, 0.0, 0.5]);
    }

    #[test]
    fn i24_sign_extends() {
        // -1.0, 0.0, +0.5 as packed little-endian 24-bit.
        let bytes = [0x00, 0x00, 0x80, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40];
        let mut out = Vec::new();
        decode_mono(&bytes, SampleFormat::I24, 1, &mut out);
        assert_eq!(out, vec![-1.0, 0.0, 0.5]);
    }

    #[test]
    fn trailing_partial_frame_is_dropped() {
        let mut bytes = Vec::new();
        for s in [1.0f32, 1.0, 1.0] {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let mut out = Vec::new();
        decode_mono(&bytes, SampleFormat::F32, 2, &mut out);
        assert_eq!(out, vec![1.0]);
    }

    #[test]
    fn eight_channels_average_to_one() {
        let mut bytes = Vec::new();
        for s in [1.0f32, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0] {
            bytes.extend_from_slice(&s.to_le_bytes());
        }
        let mut out = Vec::new();
        decode_mono(&bytes, SampleFormat::F32, 8, &mut out);
        assert_eq!(out, vec![0.0]);
    }
}
