//! Peer-to-Peer Integration Test
//!
//! This example demonstrates the peer controller heartbeat monitoring system.
//! Run this example, then in another terminal run the Python heartbeat client.

use anyhow::Result;
use std::time::Duration;
use technical_architecture::{OperationMode, PeerController, PeerControllerConfig};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    println!("===========================================================");
    println!("Peer-to-Peer Integration Test");
    println!("===========================================================");
    println!();
    println!("This test demonstrates the ZeroMQ heartbeat monitoring.");
    println!();
    println!("In another terminal, run:");
    println!("  python3 technical_architecture/deployment/python_heartbeat_client.py");
    println!();
    println!("Expected behavior:");
    println!("  - Starts in Passthrough Mode (safe default)");
    println!("  - Detects Python heartbeat → switches to Interactive Mode");
    println!("  - Python stops → switches back to Passthrough Mode");
    println!();
    println!("Press Ctrl+C to stop");
    println!("===========================================================");
    println!();

    // Create peer controller with default configuration
    let config = PeerControllerConfig::default();
    let endpoint = config.heartbeat_endpoint.clone();
    let timeout = config.heartbeat_timeout_ms;
    let mut controller = PeerController::new(config)?;

    println!("✓ Peer Controller initialized");
    println!("  Endpoint: {}", endpoint);
    println!("  Timeout: {}ms", timeout);
    println!("  Interval: 20ms (Python)");
    println!();

    // Main loop
    let mut tick_count = 0u64;
    let mut last_mode = OperationMode::Passthrough;

    loop {
        tick_count += 1;

        // Poll for heartbeat and update mode
        let mode = controller.tick()?;

        // Log mode changes
        if mode != last_mode {
            match mode {
                OperationMode::Passthrough => {
                    println!("❌ Mode: PASSTHROUGH (Python disconnected or stopped)");
                }
                OperationMode::Interactive => {
                    println!("⚡ Mode: INTERACTIVE (Python connected and sending heartbeats)");
                }
            }
            last_mode = mode;
        }

        // Log every 5 seconds (250 ticks at 20ms interval)
        if tick_count % 250 == 0 {
            println!("Tick {} - Mode: {:?}", tick_count, mode);
        }

        // Small sleep to prevent busy-waiting
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}
