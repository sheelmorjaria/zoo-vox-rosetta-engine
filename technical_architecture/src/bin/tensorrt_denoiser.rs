//! TensorRT FP16 Denoiser CLI
//!
//! Real-time audio denoising using TensorRT FP16
//!
//! # Usage
//!
//! ```bash
//! # Denoise a single file
//! tensorrt_denoiser --input noisy.wav --output clean.wav --engine dns64_fp16.trt
//!
//! # Batch process
//! tensorrt_denoiser --input-dir ./noisy/ --output-dir ./clean/ --engine dns64_fp16.trt
//!
//! # With overlap-add
//! tensorrt_denoiser --input noisy.wav --output clean.wav --engine dns64_fp16.trt --overlap 100
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "tensorrt_denoiser")]
#[command(about = "Real-time audio denoising with TensorRT FP16", long_about = None)]
struct Args {
    /// Input audio file or directory
    #[arg(short, long)]
    input: PathBuf,

    /// Output audio file or directory
    #[arg(short, long)]
    output: PathBuf,

    /// TensorRT engine file (.trt or .onnx)
    #[arg(short, long, default_value = "dns64_fp16.trt")]
    engine: PathBuf,

    /// Overlap for overlap-add (in ms)
    #[arg(long, default_value = "100")]
    overlap: f32,

    /// Number of threads
    #[arg(long, default_value = "4")]
    threads: i32,

    /// Use CUDA (if available)
    #[arg(long, default_value = "true")]
    cuda: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    log::info!("╔════════════════════════════════════════════════════════════╗");
    log::info!("║          TensorRT FP16 Denoiser (292x realtime)            ║");
    log::info!("╚════════════════════════════════════════════════════════════╝");

    // Check if input is a directory or file
    if args.input.is_dir() {
        process_directory(&args)?;
    } else {
        process_file(&args)?;
    }

    Ok(())
}

fn process_file(args: &Args) -> Result<()> {
    log::info!("Loading engine: {:?}", args.engine);

    // Load denoiser
    let config = technical_architecture::tensorrt_denoiser::TensorRTConfig {
        sample_rate: 16000,
        chunk_size: 48000,
        use_cuda: args.cuda,
        num_threads: args.threads,
    };

    let denoiser = technical_architecture::tensorrt_denoiser::TensorRTDenoiser::with_config(
        &args.engine,
        config,
    )
    .with_context(|| "Failed to load TensorRT engine")?;

    let info = denoiser.info();
    log::info!("Engine loaded:");
    log::info!("  Input: {:?}", info.input_names);
    log::info!("  Output: {:?}", info.output_names);
    log::info!("  Sample rate: {} Hz", info.sample_rate);
    log::info!(
        "  Chunk size: {} samples ({}s)",
        info.chunk_size,
        info.chunk_size as f32 / info.sample_rate as f32
    );

    // Load input audio
    log::info!("Loading input: {:?}", args.input);
    let (audio, source_sr) = load_audio(&args.input)?;

    log::info!("  Duration: {:.2}s", audio.len() as f32 / source_sr as f32);
    log::info!("  Sample rate: {} Hz", source_sr);

    // Denoise
    log::info!("Denoising...");
    let start = std::time::Instant::now();

    let clean_audio = if args.overlap > 0.0 {
        denoiser.denoise_overlap_add(&audio, args.overlap)?
    } else {
        denoiser.denoise(&audio)?
    };

    let elapsed = start.elapsed();
    let rtf = (audio.len() as f32 / source_sr as f32) / (elapsed.as_secs_f32());

    log::info!("  Processing time: {:.1}ms", elapsed.as_secs_f32() * 1000.0);
    log::info!("  Realtime factor: {:.0}x", rtf);

    // Save output
    log::info!("Saving output: {:?}", args.output);
    save_audio(&args.output, &clean_audio, 16000)?;

    log::info!("Done!");

    Ok(())
}

fn process_directory(args: &Args) -> Result<()> {
    log::info!("Processing directory: {:?}", args.input);

    // Create output directory
    std::fs::create_dir_all(&args.output)?;

    // Load denoiser once
    let config = technical_architecture::tensorrt_denoiser::TensorRTConfig {
        sample_rate: 16000,
        chunk_size: 48000,
        use_cuda: args.cuda,
        num_threads: args.threads,
    };

    let denoiser = technical_architecture::tensorrt_denoiser::TensorRTDenoiser::with_config(
        &args.engine,
        config,
    )?;

    // Process all .wav files
    let mut total_processed = 0;
    let mut total_rtf = 0.0;

    for entry in std::fs::read_dir(&args.input)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().map(|e| e == "wav").unwrap_or(false) {
            let output_path = args.output.join(path.file_name().unwrap());

            log::info!("Processing: {:?}", path);

            let (audio, source_sr) = load_audio(&path)?;

            let start = std::time::Instant::now();
            let clean = if args.overlap > 0.0 {
                denoiser.denoise_overlap_add(&audio, args.overlap)?
            } else {
                denoiser.denoise(&audio)?
            };
            let elapsed = start.elapsed();

            let rtf = (audio.len() as f32 / source_sr as f32) / elapsed.as_secs_f32();
            log::info!("  RTF: {:.0}x", rtf);

            save_audio(&output_path, &clean, 16000)?;
            total_processed += 1;
            total_rtf += rtf;
        }
    }

    log::info!("Processed {} files", total_processed);
    log::info!("Average RTF: {:.0}x", total_rtf / total_processed as f32);

    Ok(())
}

fn load_audio(path: &std::path::Path) -> Result<(Vec<f32>, u32)> {
    // Use hound for WAV loading
    let reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| s.unwrap_or(0.0))
            .collect(),
        hound::SampleFormat::Int => {
            let max_val = (1 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.unwrap_or(0) as f32 / max_val)
                .collect()
        }
    };

    // Convert to mono if stereo
    let mono = if spec.channels == 2 {
        samples
            .chunks(2)
            .map(|chunk| (chunk[0] + chunk.get(1).copied().unwrap_or(0.0)) / 2.0)
            .collect()
    } else {
        samples
    };

    Ok((mono, sample_rate))
}

fn save_audio(path: &std::path::Path, audio: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = hound::WavWriter::create(path, spec)?;
    for &sample in audio {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;

    Ok(())
}
