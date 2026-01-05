use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use reflex_arc::{AudioProcessor, SafetySystem};
use std::time::Duration;

fn fft_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT Performance");

    // Test different buffer sizes
    for buffer_size in [128, 512, 1024, 2048].iter() {
        group.bench_with_input(BenchmarkId::new("fft", buffer_size), buffer_size, |b, &size| {
            let processor = AudioProcessor::new(48000, size);
            let signal: Vec<f32> = (0..size).map(|i| (i as f32 / size as f32 * 2.0 * std::f32::consts::PI).sin()).collect();

            b.iter(|| {
                black_box(processor.compute_stft(&signal));
            });
        });
    }

    group.finish();
}

fn safety_system_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Safety System");

    // Test with different signal levels
    for signal_level in [0.1, 0.5, 1.0].iter() {
        group.bench_with_input(BenchmarkId::new("spl_check", signal_level), signal_level, |b, &level| {
            let mut safety = SafetySystem::new(90.0);
            let signal = vec![level; 512];

            b.iter(|| {
                black_box(safety.check_spl(&signal));
            });
        });
    }

    group.finish();
}

fn full_processing_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Full Audio Processing");

    // Test the full pipeline
    for buffer_size in [256, 512, 1024].iter() {
        group.bench_with_input(BenchmarkId::new("full_pipeline", buffer_size), buffer_size, |b, &size| {
            let mut processor = AudioProcessor::new(48000, size);
            let signal: Vec<f32> = (0..size).map(|i| (i as f32 / size as f32 * 2.0 * std::f32::consts::PI).sin()).collect();

            b.iter(|| {
                black_box(processor.process_buffer(&signal));
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    fft_benchmark,
    safety_system_benchmark,
    full_processing_benchmark
);
criterion_main!(benches);