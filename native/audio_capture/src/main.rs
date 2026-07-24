//! Speecher's capture child: WASAPI in, mono f32 PCM out.
//!
//! Contract with the Python parent (src/audio/rust_capture.py):
//!   stdout  raw little-endian f32 samples, mono, at --sample-rate. Nothing else.
//!   stderr  one JSON object per line: info / log / fatal (see proto.rs).
//!   stdin   closing it (or the parent dying) is the stop signal.
//!   exit    0 = clean stop, 1 = setup failure, 2 = device failure mid-stream.
//!
//! With --list-devices it instead prints one JSON object of endpoints to
//! stdout and exits; that is what scripts/list_audio_devices.py reads.

use std::io::{Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};

mod capture;
mod convert;
mod proto;
mod resample;

use capture::Source;

const USAGE: &str = "\
audio_capture -- WASAPI loopback/microphone capture for Speecher

Usage:
  audio_capture [--device-hint <substring>] [--source loopback|mic]
                [--sample-rate <hz>]
  audio_capture --list-devices
  audio_capture --help

Options:
  --device-hint <s>  Case-insensitive substring of the endpoint's friendly
                     name. Empty (default) means the default endpoint.
  --source <s>       loopback (default) captures what is played back;
                     mic captures a recording endpoint.
  --sample-rate <n>  Output sample rate in Hz (default 16000).
  --list-devices     Print endpoints as JSON and exit.
";

struct Args {
    device_hint: String,
    source: Source,
    sample_rate: usize,
    list_devices: bool,
    help: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            device_hint: String::new(),
            source: Source::Loopback,
            sample_rate: 16000,
            list_devices: false,
            help: false,
        }
    }
}

fn parse_args<I: Iterator<Item = String>>(args: I) -> Result<Args, String> {
    let mut parsed = Args::default();
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        let mut value = || {
            args.next()
                .ok_or_else(|| format!("{arg} needs a value (see --help)"))
        };
        match arg.as_str() {
            "--device-hint" => parsed.device_hint = value()?,
            "--source" => {
                let raw = value()?;
                parsed.source =
                    Source::parse(&raw).ok_or_else(|| format!("unknown --source '{raw}'"))?;
            }
            "--sample-rate" => {
                let raw = value()?;
                parsed.sample_rate = raw
                    .parse()
                    .map_err(|_| format!("--sample-rate '{raw}' is not a number"))?;
                if parsed.sample_rate == 0 {
                    return Err("--sample-rate must be > 0".to_string());
                }
            }
            "--list-devices" => parsed.list_devices = true,
            "--help" | "-h" => parsed.help = true,
            other => return Err(format!("unknown argument '{other}' (see --help)")),
        }
    }
    Ok(parsed)
}

/// Watch stdin so an exiting parent takes us with it: the read returns EOF as
/// soon as the pipe's write end is gone, whether the parent asked or crashed.
fn watch_parent(stop: &'static AtomicBool) {
    std::thread::spawn(move || {
        let mut buf = [0u8; 64];
        let mut stdin = std::io::stdin().lock();
        loop {
            match stdin.read(&mut buf) {
                Ok(0) | Err(_) => break,
                Ok(_) => continue,
            }
        }
        stop.store(true, Ordering::Relaxed);
    });
}

fn main() {
    let args = match parse_args(std::env::args().skip(1)) {
        Ok(args) => args,
        Err(message) => {
            proto::fatal(&message);
            std::process::exit(1);
        }
    };

    if args.help {
        print!("{USAGE}");
        return;
    }

    if args.list_devices {
        match capture::list_devices() {
            Ok(json) => println!("{json}"),
            Err(message) => {
                proto::fatal(&message);
                std::process::exit(1);
            }
        }
        return;
    }

    // 'static so the stdin watcher can hold it without an Arc; it lives for
    // the whole process either way.
    static STOP: AtomicBool = AtomicBool::new(false);
    watch_parent(&STOP);

    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::new(stdout.lock());
    let mut bytes: Vec<u8> = Vec::new();
    let mut sink = |samples: &[f32]| -> std::io::Result<()> {
        bytes.clear();
        bytes.reserve(samples.len() * 4);
        for sample in samples {
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
        out.write_all(&bytes)?;
        // Flush every packet: buffering here would show up as latency in the
        // transcript, and the packets are ~20 ms anyway.
        out.flush()
    };

    match capture::run(
        &args.device_hint,
        args.source,
        args.sample_rate,
        &STOP,
        &mut sink,
    ) {
        Ok(()) => {}
        Err(err) => {
            proto::fatal(err.message());
            std::process::exit(err.exit_code());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Args, String> {
        parse_args(args.iter().map(|s| s.to_string()))
    }

    #[test]
    fn defaults_are_loopback_at_16k() {
        let args = parse(&[]).unwrap();
        assert_eq!(args.source, Source::Loopback);
        assert_eq!(args.sample_rate, 16000);
        assert!(args.device_hint.is_empty());
        assert!(!args.list_devices);
    }

    #[test]
    fn parses_a_full_command_line() {
        let args = parse(&[
            "--device-hint",
            "HyperX",
            "--source",
            "mic",
            "--sample-rate",
            "48000",
        ])
        .unwrap();
        assert_eq!(args.device_hint, "HyperX");
        assert_eq!(args.source, Source::Mic);
        assert_eq!(args.sample_rate, 48000);
    }

    #[test]
    fn hint_with_spaces_stays_one_value() {
        let args = parse(&["--device-hint", "Realtek Digital Output"]).unwrap();
        assert_eq!(args.device_hint, "Realtek Digital Output");
    }

    #[test]
    fn bad_input_is_rejected() {
        assert!(parse(&["--source", "speakers"]).is_err());
        assert!(parse(&["--sample-rate", "many"]).is_err());
        assert!(parse(&["--sample-rate", "0"]).is_err());
        assert!(parse(&["--sample-rate"]).is_err());
        assert!(parse(&["--nonsense"]).is_err());
    }
}
