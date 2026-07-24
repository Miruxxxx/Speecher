//! WASAPI loopback (or microphone) capture, event driven, shared mode.
//!
//! Why this exists at all: PyAudioWPatch/PortAudio corrupts the heap of its
//! host process after minutes of an *idle* loopback stream
//! (docs/CODE_REVIEW_2026-07-15.md B1). This talks to WASAPI directly, so the
//! failure mode it was isolated from is simply not in the process any more.
//!
//! Silence policy, deliberately identical to the PortAudio path: an idle
//! loopback endpoint produces no packets at all, and we invent none. Frames
//! WASAPI *does* hand us are always forwarded, including the ones flagged
//! AUDCLNT_BUFFERFLAGS_SILENT (emitted as zeros), because those are real time
//! on the device timeline and the ASR word timestamps are derived from the
//! frame count.

use std::sync::atomic::{AtomicBool, Ordering};

use wasapi::{
    initialize_mta, Device, DeviceEnumerator, Direction, SampleType, StreamMode, WaveFormat,
};

use crate::convert::{decode_mono, SampleFormat};
use crate::proto;
use crate::resample::MonoResampler;

/// Shared-mode capture buffer. Well above the ~10 ms device period so a stalled
/// parent (a full stdout pipe) costs latency before it costs frames.
const BUFFER_HNS: i64 = 1_000_000; // 100 ms in 100-ns units

/// How long to block on the stream event before re-checking `stop`. A timeout
/// is the normal state in silence, not an error.
const EVENT_WAIT_MS: u32 = 500;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// What the machine plays: capture on a render endpoint (loopback).
    Loopback,
    /// What the machine hears: a regular capture endpoint.
    Mic,
}

impl Source {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "loopback" => Some(Source::Loopback),
            "mic" | "microphone" | "input" => Some(Source::Mic),
            _ => None,
        }
    }

    /// The endpoint direction to open. Loopback lives on a *render* endpoint;
    /// wasapi turns (Render device, Capture stream) into AUDCLNT_STREAMFLAGS_LOOPBACK.
    fn endpoint(self) -> Direction {
        match self {
            Source::Loopback => Direction::Render,
            Source::Mic => Direction::Capture,
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Source::Loopback => "loopback",
            Source::Mic => "mic",
        }
    }
}

/// Setup errors are the caller's problem (wrong device, no endpoint); device
/// errors happen mid-stream (endpoint invalidated by a hardware switch) and are
/// worth a restart.
pub enum CaptureError {
    Setup(String),
    Device(String),
}

impl CaptureError {
    pub fn message(&self) -> &str {
        match self {
            CaptureError::Setup(m) | CaptureError::Device(m) => m,
        }
    }

    pub const fn exit_code(&self) -> i32 {
        match self {
            CaptureError::Setup(_) => 1,
            CaptureError::Device(_) => 2,
        }
    }
}

fn com_init() -> Result<(), String> {
    initialize_mta()
        .ok()
        .map_err(|e| format!("COM initialize_mta failed: {e}"))
}

/// First endpoint whose friendly name contains `hint`, else the default one.
///
/// Note the difference from the PortAudio path: there the hint matched a
/// "... (loopback)" *input* device invented by PortAudio; here it matches the
/// name of the output endpoint itself.
fn pick_device(
    enumerator: &DeviceEnumerator,
    direction: Direction,
    hint: &str,
) -> Result<Device, String> {
    let hint = hint.trim().to_lowercase();
    if !hint.is_empty() {
        let collection = enumerator
            .get_device_collection(&direction)
            .map_err(|e| format!("cannot enumerate {direction} devices: {e}"))?;
        for device in &collection {
            let Ok(device) = device else { continue };
            let Ok(name) = device.get_friendlyname() else {
                continue;
            };
            if name.to_lowercase().contains(&hint) {
                return Ok(device);
            }
        }
        proto::log(&format!(
            "no active {direction} endpoint matches hint '{hint}'; falling back to the default one"
        ));
    }
    enumerator
        .get_default_device(&direction)
        .map_err(|e| format!("no default {direction} endpoint: {e}"))
}

fn sample_format(format: &WaveFormat) -> Result<SampleFormat, String> {
    let bits = format.get_bitspersample();
    let subformat = format
        .get_subformat()
        .map_err(|e| format!("unsupported mix format: {e}"))?;
    match (subformat, bits) {
        (SampleType::Float, 32) => Ok(SampleFormat::F32),
        (SampleType::Int, 16) => Ok(SampleFormat::I16),
        (SampleType::Int, 24) => Ok(SampleFormat::I24),
        (SampleType::Int, 32) => Ok(SampleFormat::I32),
        (t, b) => Err(format!("unsupported mix format: {t} {b} bit")),
    }
}

/// Enumerate endpoints as JSON, for `--list-devices`.
pub fn list_devices() -> Result<String, String> {
    com_init()?;
    let enumerator = DeviceEnumerator::new().map_err(|e| format!("device enumerator: {e}"))?;
    let mut root = serde_json::Map::new();

    for (key, direction) in [
        ("render", Direction::Render),
        ("capture", Direction::Capture),
    ] {
        let default_id = enumerator
            .get_default_device(&direction)
            .ok()
            .and_then(|d| d.get_id().ok());
        let collection = enumerator
            .get_device_collection(&direction)
            .map_err(|e| format!("cannot enumerate {direction} devices: {e}"))?;

        let mut items = Vec::new();
        for device in &collection {
            let Ok(device) = device else { continue };
            let id = device.get_id().unwrap_or_default();
            let name = device
                .get_friendlyname()
                .unwrap_or_else(|_| "<unknown>".to_string());
            let mix = device
                .get_iaudioclient()
                .and_then(|client| client.get_mixformat())
                .ok();
            items.push(serde_json::json!({
                "name": name,
                "is_default": Some(&id) == default_id.as_ref(),
                "sample_rate": mix.as_ref().map(|f| f.get_samplespersec()),
                "channels": mix.as_ref().map(|f| f.get_nchannels()),
                "format": mix.as_ref().and_then(|f| sample_format(f).ok()).map(|f| f.as_str()),
                "id": id,
            }));
        }
        root.insert(key.to_string(), serde_json::Value::Array(items));
    }
    Ok(serde_json::Value::Object(root).to_string())
}

/// Capture until `stop` is set, handing mono `target_sr` samples to `sink`.
///
/// `sink` returning a broken pipe means the parent is gone: that is a normal
/// shutdown, not a failure.
pub fn run(
    device_hint: &str,
    source: Source,
    target_sr: usize,
    stop: &AtomicBool,
    sink: &mut dyn FnMut(&[f32]) -> std::io::Result<()>,
) -> Result<(), CaptureError> {
    com_init().map_err(CaptureError::Setup)?;
    let enumerator = DeviceEnumerator::new()
        .map_err(|e| CaptureError::Setup(format!("device enumerator: {e}")))?;
    let device =
        pick_device(&enumerator, source.endpoint(), device_hint).map_err(CaptureError::Setup)?;
    let name = device
        .get_friendlyname()
        .unwrap_or_else(|_| "<unknown>".to_string());

    let mut client = device
        .get_iaudioclient()
        .map_err(|e| CaptureError::Setup(format!("cannot activate '{name}': {e}")))?;
    let format = client
        .get_mixformat()
        .map_err(|e| CaptureError::Setup(format!("cannot read mix format of '{name}': {e}")))?;
    let sample_fmt = sample_format(&format).map_err(CaptureError::Setup)?;
    let channels = format.get_nchannels() as usize;
    let input_sr = format.get_samplespersec() as usize;
    let frame_bytes = sample_fmt.bytes() * channels;

    let (default_period, _min_period) = client
        .get_device_period()
        .map_err(|e| CaptureError::Setup(format!("cannot read device period: {e}")))?;
    let mode = StreamMode::EventsShared {
        // No AUTOCONVERTPCM: we downmix and resample ourselves, so the audio
        // handed to the ASR does not depend on the quality of the OS resampler.
        autoconvert: false,
        buffer_duration_hns: BUFFER_HNS.max(default_period),
    };
    client
        .initialize_client(&format, &Direction::Capture, &mode)
        .map_err(|e| CaptureError::Setup(format!("cannot open '{name}' for capture: {e}")))?;

    let event = client
        .set_get_eventhandle()
        .map_err(|e| CaptureError::Setup(format!("cannot get stream event: {e}")))?;
    let capture_client = client
        .get_audiocaptureclient()
        .map_err(|e| CaptureError::Setup(format!("cannot get capture client: {e}")))?;
    let mut resampler = MonoResampler::new(input_sr, target_sr).map_err(CaptureError::Setup)?;

    client
        .start_stream()
        .map_err(|e| CaptureError::Setup(format!("cannot start stream: {e}")))?;
    proto::info(
        &name,
        source.as_str(),
        input_sr as u32,
        channels as u16,
        sample_fmt.as_str(),
        target_sr,
    );

    let mut raw: Vec<u8> = Vec::new();
    let mut mono: Vec<f32> = Vec::new();
    let mut out: Vec<f32> = Vec::new();
    let mut discontinuities: u64 = 0;

    while !stop.load(Ordering::Relaxed) {
        loop {
            let packet = capture_client
                .get_next_packet_size()
                .map_err(|e| CaptureError::Device(format!("get_next_packet_size failed: {e}")))?
                .unwrap_or(0) as usize;
            if packet == 0 {
                break;
            }
            let needed = packet * frame_bytes;
            if raw.len() < needed {
                raw.resize(needed, 0);
            }
            let (frames, info) = capture_client
                .read_from_device(&mut raw[..needed])
                .map_err(|e| CaptureError::Device(format!("read_from_device failed: {e}")))?;
            if frames == 0 {
                break;
            }
            if info.flags.data_discontinuity {
                // Expected on loopback every time the endpoint wakes up from
                // silence, so this is a counter, not an alarm. It only means
                // something when it climbs while audio is playing.
                discontinuities += 1;
                if discontinuities % 50 == 1 {
                    proto::log(&format!(
                        "stream gap #{discontinuities} (normal after silence on loopback)"
                    ));
                }
            }

            mono.clear();
            if info.flags.silent {
                // MSDN: the buffer contents are undefined when SILENT is set.
                mono.resize(frames as usize, 0.0);
            } else {
                decode_mono(
                    &raw[..frames as usize * frame_bytes],
                    sample_fmt,
                    channels,
                    &mut mono,
                );
            }

            out.clear();
            resampler
                .push(&mono, &mut out)
                .map_err(CaptureError::Device)?;
            if !out.is_empty() {
                if let Err(e) = sink(&out) {
                    if e.kind() == std::io::ErrorKind::BrokenPipe {
                        let _ = client.stop_stream();
                        return Ok(());
                    }
                    return Err(CaptureError::Device(format!("stdout write failed: {e}")));
                }
            }
        }
        // Timeout = nothing rendered on the endpoint. Normal; loop and re-check stop.
        let _ = event.wait_for_event(EVENT_WAIT_MS);
    }

    let _ = client.stop_stream();
    Ok(())
}
