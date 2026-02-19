// Debug script to test marmoset phrase extraction
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{DecoderOptions, CODEC_TYPE_NULL};
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;
use technical_architecture::phrase_sequence_analyzer::PhraseSequenceAnalyzer;

fn load_flac_file(path: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let file = std::fs::File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    hint.with_extension("flac");

    let meta_opts: MetadataOptions = Default::default();
    let fmt_opts: FormatOptions = Default::default();

    let probed = symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts)?;

    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or("No valid audio track found")?;

    let mut decoder =
        symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;
    let n_channels = decoder.codec_params().channels.map_or(1, |ch| ch.count());

    let mut audio_samples = Vec::new();
    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(_) => break,
        };

        let decoded = decoder.decode(&packet)?;

        match decoded {
            AudioBufferRef::F64(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(samples.iter().map(|&s| s as f32));
                }
            }
            AudioBufferRef::F32(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend_from_slice(samples);
                }
            }
            AudioBufferRef::S32(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(samples.iter().map(|&s| s as f32 / i32::MAX as f32));
                }
            }
            AudioBufferRef::S24(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(
                        samples
                            .iter()
                            .map(|&s| s.inner() as f32 / (i32::MAX >> 8) as f32),
                    );
                }
            }
            AudioBufferRef::S16(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(samples.iter().map(|&s| s as f32 / i16::MAX as f32));
                }
            }
            AudioBufferRef::S8(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(samples.iter().map(|&s| s as f32 / i8::MAX as f32));
                }
            }
            AudioBufferRef::U8(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(samples.iter().map(|&s| (s as f32 - 128.0) / 128.0));
                }
            }
            AudioBufferRef::U16(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(samples.iter().map(|&s| (s as f32 - 32768.0) / 32768.0));
                }
            }
            AudioBufferRef::U24(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(
                        samples
                            .iter()
                            .map(|&s| (s.inner() as f32 - 8388608.0) / 8388608.0),
                    );
                }
            }
            AudioBufferRef::U32(buf) => {
                for ch in 0..n_channels {
                    let samples = buf.chan(ch);
                    audio_samples.extend(
                        samples
                            .iter()
                            .map(|&s| (s as f32 - 2147483648.0) / 2147483648.0),
                    );
                }
            }
        }
    }

    Ok(audio_samples)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let test_file = "/home/sheel/birdsong_analysis/data/Vocalizations/2021_1_0/Phee_525714.flac";

    println!("Loading marmoset FLAC file: {}", test_file);
    let audio = load_flac_file(test_file)?;
    println!("Loaded {} samples at 96kHz", audio.len());
    println!("Duration: {:.2} seconds", audio.len() as f64 / 96000.0);

    // Test phrase extraction with different thresholds
    let analyzer = PhraseSequenceAnalyzer::with_threshold(0.2);
    let phrases = analyzer.extract_phrases(&audio, 96000)?;
    println!("\nThreshold 0.2: Extracted {} phrases", phrases.len());

    let analyzer2 = PhraseSequenceAnalyzer::with_threshold(0.1);
    let phrases2 = analyzer2.extract_phrases(&audio, 96000)?;
    println!("Threshold 0.1: Extracted {} phrases", phrases2.len());

    let analyzer3 = PhraseSequenceAnalyzer::with_threshold(0.05);
    let phrases3 = analyzer3.extract_phrases(&audio, 96000)?;
    println!("Threshold 0.05: Extracted {} phrases", phrases3.len());

    let analyzer4 = PhraseSequenceAnalyzer::with_threshold(0.01);
    let phrases4 = analyzer4.extract_phrases(&audio, 96000)?;
    println!("Threshold 0.01: Extracted {} phrases", phrases4.len());

    // Show some details about the first phrase if any were extracted
    if !phrases4.is_empty() {
        let first = &phrases4[0];
        println!("\nFirst phrase details:");
        println!("  Start: {:.2} ms", first.start_ms);
        println!("  Duration: {:.2} ms", first.duration_ms);
        println!("  End: {:.2} ms", first.start_ms + first.duration_ms);
    }

    Ok(())
}
