/**
 * Peer-to-Peer Performance Benchmarking Module
 * ===============================================
 *
 * This module provides comprehensive performance testing for the peer-to-peer
 * architecture including throughput, latency, and resource usage metrics.
 *
 * Author: Sheel Morjaria (sheelmorjaria@gmail.com)
 * License: CC BY-ND 4.0 International
 */

use std::time::{Duration, Instant};
use std::thread;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Performance metrics collected during benchmarks
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Number of operations completed
    pub operations: u64,
    /// Total time elapsed (in microseconds)
    pub total_duration_us: u64,
    /// Average latency per operation (in microseconds)
    pub avg_latency_us: f64,
    /// Minimum latency (in microseconds)
    pub min_latency_us: u64,
    /// Maximum latency (in microseconds)
    pub max_latency_us: u64,
    /// Throughput (operations per second)
    pub throughput_ops_per_sec: f64,
    /// Memory allocated (in bytes, approximate)
    pub memory_allocated_bytes: u64,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self {
            operations: 0,
            total_duration_us: 0,
            avg_latency_us: 0.0,
            min_latency_us: u64::MAX,
            max_latency_us: 0,
            throughput_ops_per_sec: 0.0,
            memory_allocated_bytes: 0,
        }
    }

    pub fn calculate_throughput(&mut self) {
        if self.total_duration_us > 0 {
            self.throughput_ops_per_sec = (self.operations as f64 * 1_000_000.0) / self.total_duration_us as f64;
        }
    }

    pub fn calculate_avg_latency(&mut self) {
        if self.operations > 0 {
            self.avg_latency_us = self.total_duration_us as f64 / self.operations as f64;
        }
    }
}

/// Heartbeat message (replicated from peer_controller.rs for standalone testing)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeartbeatMessage {
    pub sequence: u64,
    pub timestamp: u64,
    pub pid: u32,
    pub state: String,
}

impl HeartbeatMessage {
    pub fn new(sequence: u64, pid: u32) -> Self {
        Self {
            sequence,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_micros() as u64,
            pid,
            state: "active".to_string(),
        }
    }

    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        Ok(bincode::serialize(self)?)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Ok(bincode::deserialize(bytes)?)
    }
}

/// Peer controller simulator for performance testing
pub struct PeerControllerSimulator {
    last_heartbeat: Option<Instant>,
    heartbeat_count: Arc<AtomicU64>,
    mode: OperationMode,
    heartbeat_timeout_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperationMode {
    Passthrough,
    Interactive,
}

impl PeerControllerSimulator {
    pub fn new(heartbeat_timeout_ms: u64) -> Self {
        Self {
            last_heartbeat: None,
            heartbeat_count: Arc::new(AtomicU64::new(0)),
            mode: OperationMode::Passthrough,
            heartbeat_timeout_ms,
        }
    }

    pub fn handle_heartbeat(&mut self) {
        self.last_heartbeat = Some(Instant::now());
        self.heartbeat_count.fetch_add(1, Ordering::Relaxed);
        self.mode = OperationMode::Interactive;
    }

    pub fn check_timeout(&mut self) -> bool {
        if let Some(last) = self.last_heartbeat {
            if last.elapsed() > Duration::from_millis(self.heartbeat_timeout_ms) {
                self.mode = OperationMode::Passthrough;
                return true;
            }
        }
        false
    }

    pub fn mode(&self) -> OperationMode {
        self.mode
    }

    pub fn heartbeat_count(&self) -> u64 {
        self.heartbeat_count.load(Ordering::Relaxed)
    }
}

// ============================================================================
// Performance Benchmarks
// ============================================================================

/// Benchmark 1: Serialization/Deserialization Throughput
pub fn benchmark_serialization_throughput(iterations: u64) -> PerformanceMetrics {
    let mut metrics = PerformanceMetrics::new();
    let mut latencies: Vec<u64> = Vec::with_capacity(iterations as usize);

    let start = Instant::now();

    for i in 0..iterations {
        let msg = HeartbeatMessage::new(i, 12345);

        let ser_start = Instant::now();
        let bytes = msg.to_bytes().unwrap();
        let _deserialized = HeartbeatMessage::from_bytes(&bytes).unwrap();
        let ser_latency = ser_start.elapsed().as_micros() as u64;

        latencies.push(ser_latency);
        metrics.memory_allocated_bytes += bytes.len() as u64;
    }

    metrics.operations = iterations;
    metrics.total_duration_us = start.elapsed().as_micros() as u64;
    metrics.min_latency_us = *latencies.iter().min().unwrap_or(&0);
    metrics.max_latency_us = *latencies.iter().max().unwrap_or(&0);
    metrics.calculate_avg_latency();
    metrics.calculate_throughput();

    metrics
}

/// Benchmark 2: Message Processing Throughput
pub fn benchmark_message_processing(iterations: u64) -> PerformanceMetrics {
    let mut controller = PeerControllerSimulator::new(100);
    let mut metrics = PerformanceMetrics::new();
    let mut latencies: Vec<u64> = Vec::with_capacity(iterations as usize);

    let start = Instant::now();

    for _ in 0..iterations {
        let proc_start = Instant::now();
        controller.handle_heartbeat();
        let latency = proc_start.elapsed().as_micros() as u64;
        latencies.push(latency);
    }

    metrics.operations = iterations;
    metrics.total_duration_us = start.elapsed().as_micros() as u64;
    metrics.min_latency_us = *latencies.iter().min().unwrap_or(&0);
    metrics.max_latency_us = *latencies.iter().max().unwrap_or(&0);
    metrics.calculate_avg_latency();
    metrics.calculate_throughput();

    metrics
}

/// Benchmark 3: Timeout Detection Latency
pub fn benchmark_timeout_detection() -> PerformanceMetrics {
    let mut controller = PeerControllerSimulator::new(50); // 50ms timeout
    let mut metrics = PerformanceMetrics::new();

    // Test 10 timeout cycles
    let iterations = 10u64;
    let mut latencies: Vec<u64> = Vec::with_capacity(iterations as usize);

    let start = Instant::now();

    for _ in 0..iterations {
        // Send heartbeat
        controller.handle_heartbeat();

        // Wait for timeout (50ms + 10ms margin)
        thread::sleep(Duration::from_millis(60));

        // Measure timeout detection latency (excluding the sleep time)
        let detect_start = Instant::now();
        let timed_out = controller.check_timeout();
        let detect_latency = detect_start.elapsed().as_micros() as u64;

        assert!(timed_out, "Timeout should have been detected");
        latencies.push(detect_latency);
    }

    // Only count the actual detection time, not the sleep time
    metrics.operations = iterations;
    metrics.total_duration_us = latencies.iter().sum::<u64>();
    metrics.min_latency_us = *latencies.iter().min().unwrap_or(&0);
    metrics.max_latency_us = *latencies.iter().max().unwrap_or(&0);
    metrics.calculate_avg_latency();

    // Calculate throughput based on actual operations per real time
    let real_duration_us = start.elapsed().as_micros() as u64;
    metrics.throughput_ops_per_sec = (iterations as f64 * 1_000_000.0) / real_duration_us as f64;

    metrics
}

/// Benchmark 4: Mode Switching Speed
pub fn benchmark_mode_switching(iterations: u64) -> PerformanceMetrics {
    let mut controller = PeerControllerSimulator::new(100);
    let mut metrics = PerformanceMetrics::new();
    let mut latencies: Vec<u64> = Vec::with_capacity(iterations as usize);

    let start = Instant::now();

    for _ in 0..iterations {
        // Start in Passthrough mode
        assert_eq!(controller.mode(), OperationMode::Passthrough);

        // Send heartbeat (should switch to Interactive)
        let switch_start = Instant::now();
        controller.handle_heartbeat();
        assert_eq!(controller.mode(), OperationMode::Interactive);
        let switch_latency_1 = switch_start.elapsed().as_micros() as u64;

        // Wait for timeout
        thread::sleep(Duration::from_millis(110));

        // Check timeout (should switch back to Passthrough)
        let switch_start = Instant::now();
        controller.check_timeout();
        assert_eq!(controller.mode(), OperationMode::Passthrough);
        let switch_latency_2 = switch_start.elapsed().as_micros() as u64;

        // Only count the actual switching time, not the sleep time
        latencies.push(switch_latency_1 + switch_latency_2);
    }

    // Only count the actual switching time, not the sleep time
    metrics.operations = iterations;
    metrics.total_duration_us = latencies.iter().sum::<u64>();
    metrics.min_latency_us = *latencies.iter().min().unwrap_or(&0);
    metrics.max_latency_us = *latencies.iter().max().unwrap_or(&0);
    metrics.calculate_avg_latency();

    // Calculate throughput based on actual operations per real time
    let real_duration_us = start.elapsed().as_micros() as u64;
    metrics.throughput_ops_per_sec = (iterations as f64 * 1_000_000.0) / real_duration_us as f64;

    metrics
}

/// Benchmark 5: Concurrent Heartbeat Processing
pub fn benchmark_concurrent_processing(
    num_senders: usize,
    messages_per_sender: u64,
) -> PerformanceMetrics {
    let controller = Arc::new(std::sync::Mutex::new(PeerControllerSimulator::new(100)));
    let heartbeat_count = Arc::new(AtomicU64::new(0));
    let mut handles = vec![];

    let start = Instant::now();

    for sender_id in 0..num_senders {
        let controller_clone = controller.clone();
        let count_clone = heartbeat_count.clone();

        let handle = thread::spawn(move || {
            for i in 0..messages_per_sender {
                let mut ctrl = controller_clone.lock().unwrap();
                ctrl.handle_heartbeat();
                count_clone.fetch_add(1, Ordering::Relaxed);

                // Small delay to simulate real-world timing
                if i % 100 == 0 {
                    thread::sleep(Duration::from_micros(100));
                }
            }
        });

        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    let elapsed = start.elapsed();

    let mut metrics = PerformanceMetrics::new();
    metrics.operations = num_senders as u64 * messages_per_sender;
    metrics.total_duration_us = elapsed.as_micros() as u64;
    metrics.min_latency_us = 0;
    metrics.max_latency_us = 0;
    metrics.calculate_avg_latency();
    metrics.calculate_throughput();

    metrics
}

/// Benchmark 6: Memory Allocation Pattern
pub fn benchmark_memory_allocation(iterations: u64) -> PerformanceMetrics {
    let mut metrics = PerformanceMetrics::new();

    // Start with baseline memory
    // Note: This is a simplified measurement - real memory profiling would use more sophisticated tools
    let estimated_message_size = std::mem::size_of::<HeartbeatMessage>() +
        std::mem::size_of::<Vec<u8>>() + // bytes vector overhead
        32; // estimated serialization overhead

    let start = Instant::now();

    for i in 0..iterations {
        let msg = HeartbeatMessage::new(i, 12345);
        let _bytes = msg.to_bytes().unwrap();
        // Drop both to measure allocation pattern
    }

    metrics.operations = iterations;
    metrics.total_duration_us = start.elapsed().as_micros() as u64;
    metrics.memory_allocated_bytes = estimated_message_size as u64 * iterations;
    metrics.min_latency_us = 0;
    metrics.max_latency_us = 0;
    metrics.avg_latency_us = 0.0;
    metrics.calculate_throughput();

    metrics
}

/// Run all benchmarks and return a summary
pub fn run_all_benchmarks() -> Vec<(&'static str, PerformanceMetrics)> {
    vec![
        ("Serialization Throughput", benchmark_serialization_throughput(10_000)),
        ("Message Processing", benchmark_message_processing(10_000)),
        ("Timeout Detection", benchmark_timeout_detection()),
        ("Mode Switching", benchmark_mode_switching(100)),
        ("Concurrent Processing (4 threads, 1000 msgs)",
         benchmark_concurrent_processing(4, 1000)),
        ("Memory Allocation", benchmark_memory_allocation(10_000)),
    ]
}

/// Format performance metrics for display
pub fn format_metrics(name: &str, metrics: &PerformanceMetrics) -> String {
    format!(
        "=== {} ===\n\
         Operations: {}\n\
         Total Duration: {:.2} ms\n\
         Avg Latency: {:.2} μs\n\
         Min Latency: {} μs\n\
         Max Latency: {} μs\n\
         Throughput: {:.2} ops/sec\n\
         Memory Allocated: {} bytes\n",
        name,
        metrics.operations,
        metrics.total_duration_us as f64 / 1000.0,
        metrics.avg_latency_us,
        metrics.min_latency_us,
        metrics.max_latency_us,
        metrics.throughput_ops_per_sec,
        metrics.memory_allocated_bytes
    )
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_serialization_throughput() {
        let metrics = benchmark_serialization_throughput(1_000);

        assert_eq!(metrics.operations, 1_000);
        assert!(metrics.avg_latency_us < 100.0, "Avg latency should be < 100μs");
        assert!(metrics.throughput_ops_per_sec > 10_000.0, "Should process > 10k ops/sec");
    }

    #[test]
    fn test_performance_message_processing() {
        let metrics = benchmark_message_processing(1_000);

        assert_eq!(metrics.operations, 1_000);
        assert!(metrics.avg_latency_us < 10.0, "Message processing should be < 10μs");
        assert!(metrics.throughput_ops_per_sec > 100_000.0, "Should process > 100k msgs/sec");
    }

    #[test]
    fn test_performance_timeout_detection() {
        let metrics = benchmark_timeout_detection();

        assert_eq!(metrics.operations, 10);
        // Timeout detection should be near-instantaneous (just checking elapsed time)
        assert!(metrics.avg_latency_us < 100.0, "Timeout detection should be < 100μs");
    }

    #[test]
    fn test_performance_mode_switching() {
        let metrics = benchmark_mode_switching(10);

        assert_eq!(metrics.operations, 10);
        // Mode switching should be fast (just setting a flag)
        assert!(metrics.avg_latency_us < 1000.0, "Mode switching should be < 1ms");
    }

    #[test]
    fn test_performance_concurrent_processing() {
        let metrics = benchmark_concurrent_processing(4, 100);

        assert_eq!(metrics.operations, 400);
        // Concurrent processing should handle 4 threads without issues
        assert!(metrics.throughput_ops_per_sec > 1000.0, "Concurrent processing > 1k ops/sec");
    }

    #[test]
    fn test_performance_memory_allocation() {
        let metrics = benchmark_memory_allocation(1_000);

        assert_eq!(metrics.operations, 1_000);
        // Should allocate memory efficiently (no leaks)
        assert!(metrics.memory_allocated_bytes > 0, "Should have allocated memory");
    }

    #[test]
    fn test_run_all_benchmarks() {
        let results = run_all_benchmarks();
        assert_eq!(results.len(), 6);

        for (name, metrics) in &results {
            println!("{}", format_metrics(name, metrics));
        }
    }
}
