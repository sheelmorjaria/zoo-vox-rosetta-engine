// Export for Synthesis: JSON Metadata and Audio Segments
//
// This example demonstrates exporting clustered phrases for synthesis methods:
// - **Metadata-driven synthesis**: Feature-based phrase selection from JSON
// - **Granular synthesis**: Grain-based audio manipulation using metadata
// - **Concatenative synthesis**: Unit selection using extracted audio segments
//
// Usage:
//   cargo run --example export_for_synthesis --release --features parallel-extraction

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use std::path::Path;
use technical_architecture::{batch_process_and_cluster, export_phrases_for_synthesis};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                 Export for Synthesis: Full Pipeline                      ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    // Configuration
    let audio_dir = Path::new("/mnt/c/Users/sheel/Desktop/data/egyptian_fruit_bats/audio");
    let checkpoint_dir = Path::new("/tmp/bat_synthesis_checkpoint");
    let output_dir = Path::new("/tmp/bat_synthesis_output");
    let batch_size = 1000;
    let dbscan_eps = 0.5;
    let min_samples = 5;

    println!("📂 Configuration:");
    println!("   Audio directory: {}", audio_dir.display());
    println!("   Checkpoint directory: {}", checkpoint_dir.display());
    println!("   Output directory: {}", output_dir.display());
    println!("   Batch size: {}", batch_size);
    println!("   DBSCAN eps: {}", dbscan_eps);
    println!("   Min samples: {}", min_samples);
    println!();

    // Run batch processing and clustering
    println!("🔄 Running extraction and clustering...");
    let start = std::time::Instant::now();

    let (clustered_phrases, vocalization_results) = batch_process_and_cluster(
        audio_dir,
        batch_size,
        dbscan_eps,
        min_samples,
        checkpoint_dir,
        None, // Process all files
    )?;

    let duration = start.elapsed();

    println!();
    println!("✅ Extraction completed in {:.2}s", duration.as_secs_f64());
    println!();

    // Count unique clusters
    let unique_clusters: std::collections::HashSet<i32> = clustered_phrases.iter().map(|cp| cp.cluster_id).collect();

    println!("📊 Statistics:");
    println!("   Total clustered phrases: {}", clustered_phrases.len());
    println!("   Unique phrase types: {}", unique_clusters.len());
    println!("   Vocalization results: {} sentences", vocalization_results.len());
    println!();

    // Export JSON metadata for synthesis
    println!("📝 Exporting JSON metadata for synthesis...");
    let json_path = output_dir.join("phrases_metadata.json");
    export_phrases_for_synthesis(&clustered_phrases, &json_path)?;
    println!("   ✅ JSON exported to: {}", json_path.display());
    println!();

    // Extract audio segments for concatenative synthesis
    // Note: This functionality has been moved to the audio_segmenter module
    // Use AudioSegmenter from the synthesis_pipeline module for this
    println!("🎵 Audio segment extraction now available in audio_segmenter module");
    println!("   Use the SynthesisPipeline for comprehensive synthesis asset generation");
    println!();

    println!("╔═══════════════════════════════════════════════════════════════════════════╗");
    println!("║                    EXPORT COMPLETE                                       ║");
    println!("╚═══════════════════════════════════════════════════════════════════════════╝");
    println!();

    println!("📂 Output Structure:");
    println!("   {}/", output_dir.display());
    println!("   ├── phrases_metadata.json  # JSON for metadata-driven & granular synthesis");
    println!("   ├── phrases/               # Audio segments organized by cluster");
    println!("   │   ├── cluster_0/         # Individual phrase WAV files");
    println!("   │   ├── cluster_1/");
    println!("   │   └── ...");
    println!("   ├── metadata.json          # Complete phrase metadata");
    println!("   └── cluster_info.json      # Cluster statistics");
    println!();

    println!("🔧 Usage Examples:");
    println!();
    println!("   1. Metadata-Driven Synthesis (Python):");
    println!("      ```python");
    println!("      import json");
    println!("      with open('phrases_metadata.json') as f:");
    println!("          data = json.load(f)");
    println!();
    println!("      # Select phrases by features");
    println!("      target_features = [0.1, 0.2, ...]  # 30D target vector");
    println!("      similar = [p for p in data['phrases']");
    println!("                if cosine_similarity(p['features'], target_features) > 0.8]");
    println!("      ```");
    println!();
    println!("   2. Granular Synthesis:");
    println!("      ```python");
    println!("      # Use grain metadata for time-stretching");
    println!("      phrase = similar_phrases[0]");
    println!("      grain_size_ms = phrase['duration_ms'] / 10  # 10 grains");
    println!("      grains = extract_grains(phrase['file_name'], grain_size_ms)");
    println!("      ```");
    println!();
    println!("   3. Concatenative Synthesis:");
    println!("      ```python");
    println!("      # Select phrases by cluster");
    println!("      import soundfile as sf");
    println!();
    println!("      cluster_id = 42");
    println!("      phrases_dir = f'phrases/cluster_{{cluster_id}}'");
    println!("      phrase_files = sorted(Path(phrases_dir).glob('phrase_*.wav'))");
    println!();
    println!("      # Concatenate phrases");
    println!("      output = np.concatenate([sf.read(f)[0] for f in phrase_files])");
    println!("      sf.write('output.wav', output, 250000)");
    println!("      ```");
    println!();

    Ok(())
}
