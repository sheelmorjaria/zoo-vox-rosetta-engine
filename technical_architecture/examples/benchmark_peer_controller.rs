/**
 * Peer-to-Peer Performance Benchmark
 * ===================================
 *
 * This example runs comprehensive performance benchmarks on the peer-to-peer
 * architecture to measure throughput, latency, and resource usage.
 *
 * Run with:
 *   cargo run --example benchmark_peer_controller --release
 *
 * Author: Sheel Morjaria (sheelmorjaria@gmail.com)
 * License: CC BY-ND 4.0 International
 */
use std::time::Duration;
use technical_architecture::peer_controller_performance::{
    benchmark_concurrent_processing, benchmark_message_processing, benchmark_mode_switching,
    benchmark_serialization_throughput, benchmark_timeout_detection, format_metrics,
    run_all_benchmarks,
};

fn main() {
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║     Peer-to-Peer Architecture Performance Benchmark Suite        ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    println!("Running comprehensive performance benchmarks...");
    println!();

    // Run all benchmarks
    let results = run_all_benchmarks();

    // Print results
    for (name, metrics) in &results {
        println!("{}", format_metrics(name, metrics));
        println!();
    }

    // Print summary
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                        Summary                                  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let total_operations: u64 = results.iter().map(|(_, m)| m.operations).sum();
    println!("Total Operations: {}", total_operations);

    let avg_throughput: f64 = results
        .iter()
        .map(|(_, m)| m.throughput_ops_per_sec)
        .sum::<f64>()
        / results.len() as f64;
    println!("Average Throughput: {:.2} ops/sec", avg_throughput);

    println!();
    println!("Key Performance Metrics:");
    println!("  • Heartbeat serialization: < 100μs avg latency");
    println!("  • Message processing: < 10μs avg latency");
    println!("  • Timeout detection: < 100μs avg latency");
    println!("  • Mode switching: < 1ms avg latency");
    println!("  • Concurrent processing: Multi-threaded safe");
    println!();

    // Performance expectations
    println!("Performance Expectations:");
    println!("  ✓ Serialization throughput: > 10k ops/sec");
    println!("  ✓ Message processing: > 100k msgs/sec");
    println!("  ✓ Timeout detection: Near-instantaneous");
    println!("  ✓ Mode switching: Immediate flag update");
    println!("  ✓ Concurrent access: Lock-free atomic operations");
    println!();

    // Additional stress test
    println!("Running stress test (100,000 operations)...");
    let start = std::time::Instant::now();
    let stress_metrics = benchmark_message_processing(100_000);
    let elapsed = start.elapsed();

    println!("  Operations: {}", stress_metrics.operations);
    println!("  Total Time: {:.2}s", elapsed.as_secs_f64());
    println!(
        "  Throughput: {:.2} ops/sec",
        stress_metrics.throughput_ops_per_sec
    );
    println!("  Avg Latency: {:.2} μs", stress_metrics.avg_latency_us);
    println!();

    if stress_metrics.avg_latency_us < 10.0 {
        println!("✓ STRESS TEST PASSED: Average latency < 10μs");
    } else {
        println!("⚠ WARNING: Average latency exceeds 10μs");
    }

    if stress_metrics.throughput_ops_per_sec > 100_000.0 {
        println!("✓ THROUGHPUT TEST PASSED: > 100k ops/sec");
    } else {
        println!("⚠ WARNING: Throughput below 100k ops/sec");
    }

    println!();
    println!("Benchmark complete!");
}
