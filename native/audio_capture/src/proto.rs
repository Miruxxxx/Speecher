//! The stderr side of the protocol: one JSON object per line.
//!
//! stdout carries nothing but raw mono f32 PCM at the target rate, so every
//! word the parent needs to read goes here instead. Keeping the two streams
//! apart is what lets the parent treat stdout as an infinite byte pipe.

use std::io::Write;

fn emit(value: serde_json::Value) {
    let mut err = std::io::stderr().lock();
    let _ = writeln!(err, "{value}");
    let _ = err.flush();
}

/// Sent once, right after the stream is open: everything the parent needs to
/// interpret the bytes on stdout and to log which endpoint we ended up on.
pub fn info(
    device: &str,
    source: &str,
    input_sample_rate: u32,
    input_channels: u16,
    input_format: &str,
    output_sample_rate: usize,
) {
    emit(serde_json::json!({
        "kind": "info",
        "device": device,
        "source": source,
        "input_sample_rate": input_sample_rate,
        "input_channels": input_channels,
        "input_format": input_format,
        "output_sample_rate": output_sample_rate,
        "output_channels": 1,
    }));
}

pub fn log(text: &str) {
    emit(serde_json::json!({ "kind": "log", "text": text }));
}

/// Unrecoverable for this process. The parent decides whether to restart us.
pub fn fatal(text: &str) {
    emit(serde_json::json!({ "kind": "fatal", "text": text }));
}
