use std::fs::File;
use std::path::Path;
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::DecoderOptions;
use symphonia::core::formats::FormatOptions;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

fn main() {
    let base = "/home/sheel/birdsong_analysis/data/orcas/audio";
    
    for i in [0, 1, 10, 100, 200, 300, 400, 500].iter() {
        let path_str = format!("{}/{}.wav", base, i);
        let path = std::path::Path::new(&path_str);
        
        if !path.exists() {
            println!("{}: File not found", i);
            continue;
        }
        
        match load_audio(path) {
            Ok((samples, sr)) => {
                let rms = (samples.iter().map(|x| (*x as f32).powi(2)).sum::<f32>() / samples.len() as f32).sqrt();
                println!("{}: {} samples, SR={}, RMS={:.4}", i, samples.len(), sr, rms);
            }
            Err(e) => {
                println!("{}: Error - {}", i, e);
            }
        }
    }
}

fn load_audio(path: &Path) -> Result<(Vec<f32>, u32), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mss = MediaSourceStream::new(Box::new(file), Default::default());

    let mut hint = Hint::new();
    if let Some(ext) = path.extension() {
        hint.with_extension(ext.to_string_lossy().as_ref());
    }

    let probed = symphonia::default::get_probe().format(&hint, mss, &FormatOptions::default(), &MetadataOptions::default())?;
    let mut format = probed.format;

    let track = format.default_track().ok_or("No track")?;
    let track_id = track.id;
    let sample_rate = track.codec_params.sample_rate.unwrap_or(48000);

    let mut decoder = symphonia::default::get_codecs().make(&track.codec_params, &DecoderOptions::default())?;

    let mut samples = Vec::new();
    let mut sample_buf = None;

    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(_) => break,
        };

        if packet.track_id() != track_id { continue; }

        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("Decode error: {}", e);
                continue;
            }
        };

        if sample_buf.is_none() {
            let spec = *decoded.spec();
            let duration = decoded.capacity() as u64;
            sample_buf = Some(SampleBuffer::<f32>::new(duration, spec));
        }

        if let Some(ref mut buf) = sample_buf {
            buf.copy_interleaved_ref(decoded);
            for chunk in buf.samples().chunks(2) {
                let mono: f32 = chunk.iter().sum::<f32>() / chunk.len() as f32;
                samples.push(mono);
            }
        }
    }

    Ok((samples, sample_rate))
}
