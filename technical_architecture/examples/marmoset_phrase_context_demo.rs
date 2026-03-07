// Marmoset Phrase-Context Analysis Demo
//
// A demonstration of how to extract phrases from within vocalizations
// and test if phrases appear across different call types (contexts)
//
// Usage: cargo run --release --example marmoset_phrase_context_demo

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use std::collections::HashMap;
use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("╔════════════════════════════════════════════╗");
    println!("║    Marmoset Phrase-Context Analysis Demo                           ║");
    println!("╠════════════════════════════════════════╣");
    println!("║                                                                   ║");
    println!("║  Two-Step Process:                                                ║");
    println!("║                                                                   ║");
    println!("╚═════════════════════════════════════════════════╝");
    println!();

    println!("Step 1: Extract Phrases from Within Vocalizations");
    println!("────────────────────────────────────────────────────");
    println!();
    println!("  For true within-vocalization phrase extraction, you need:");
    println!("  - WithinVocalizationAnalyzer for phrase boundary detection");
    println!("  - MicroDynamicsExtractor for 15D feature extraction");
    println!("  - HdbscanClustering for vocabulary discovery");
    println!();
    println!("  This requires analyzing the actual acoustic content of FLAC files");
    println!("  in the marmoset Vocalizations directory:");
    println!("    /home/sheel/birdsong_analysis/data/Vocalizations");
    println!();
    println!("  This example demonstrates the methodology.");
    println!("  The analysis would test whether phrases are reused across");
    println!("  different call types (contexts), indicating combinatorial syntax.");
    println!();
    println!("Step 2: Build Phrase-Context Matrix");
    println!("────────────────────────────────────────────────────");
    println!();
    println!("  Count occurrences of each phrase in each context");
    println!("  Calculate generality score (contexts used / total contexts)");
    println!("  Calculate Shannon entropy (distribution evenness)");
    println!();
    println!("  Generality score near 1.0 = phrase appears in all contexts");
    println!("  Shannon entropy near 1.0 = evenly distributed across contexts");
    println!();
    println!("╔══════════════════════════════════════════╗");
    println!("║                    ANALYSIS COMPLETE                          ║");
    println!("╠════════════════════════════════════╣");
    println!("║                                                                   ║");
    println!("║  For production implementation, see these examples:                      ║");
    println!("║  - phrase_context_analysis_bat_generality.rs (bat example)             ║");
    println!("║  - phrase_context_analysis_marmoset_simple_minimal.rs             ║");
    println!("╚═════════════════════════════════════════════╝");
    println!();

    Ok(())
}
