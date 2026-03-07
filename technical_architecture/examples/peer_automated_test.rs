//! Automated Peer-to-Peer Test
//!
//! This example tests the peer controller without requiring manual interaction.

#![allow(clippy::all, dead_code, unused_imports, unused_variables)]
use anyhow::Result;
use serde_json;
use std::thread;
use std::time::Duration;
use technical_architecture::{HeartbeatMessage, OperationMode, PeerController, PeerControllerConfig};
use zmq::{Context, SocketType};

fn main() -> Result<()> {
    println!("===========================================================");
    println!("Automated Peer-to-Peer Test");
    println!("===========================================================");

    // Step 1: Create peer controller (Rust side)
    println!("\n[1/5] Creating Peer Controller...");
    let config = PeerControllerConfig::default();
    let mut controller = PeerController::new(config)?;

    println!("✓ Peer Controller created");
    println!("  Endpoint: ipc:///tmp/cognitive_heartbeat.ipc");

    // Step 2: Verify starts in Passthrough Mode
    println!("\n[2/5] Checking initial state...");
    let mode = controller.tick()?;
    assert_eq!(mode, OperationMode::Passthrough, "Should start in Passthrough Mode");
    println!("✓ Initial mode: PASSTHROUGH (correct)");

    // Step 3: Create Python heartbeat client (simulate)
    println!("\n[3/5] Creating Python heartbeat client...");
    let ctx = Context::new();
    let heartbeat_pub = ctx.socket(SocketType::PUB)?;
    heartbeat_pub.connect("ipc:///tmp/cognitive_heartbeat.ipc")?;
    println!("✓ Heartbeat publisher connected");

    // Give the connection time to establish
    thread::sleep(Duration::from_millis(100));

    // Step 4: Send heartbeats and verify mode switch
    println!("\n[4/5] Sending heartbeats...");
    let mut switched = false;
    for i in 0..10 {
        let heartbeat = HeartbeatMessage {
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_millis() as u64,
            sequence: i,
            pid: std::process::id(),
            state: "active".to_string(),
        };

        let message = serde_json::to_vec(&heartbeat)?;
        heartbeat_pub.send(&message, 0)?;

        // Small delay to ensure message is delivered
        thread::sleep(Duration::from_millis(5));

        // Poll after each heartbeat
        let mode = controller.tick()?;

        // Should switch to Interactive after first heartbeat
        if !switched && mode == OperationMode::Interactive {
            println!("✓ Mode switched to INTERACTIVE after heartbeat #{}", i + 1);
            switched = true;
        }

        thread::sleep(Duration::from_millis(20));
    }

    // Verify we're in Interactive Mode
    let final_mode = controller.tick()?;
    assert_eq!(final_mode, OperationMode::Interactive, "Should be in Interactive Mode");
    println!("✓ Sent 10 heartbeats, verified in Interactive Mode");

    // Step 5: Stop heartbeats and verify timeout
    println!("\n[5/5] Testing heartbeat timeout...");
    println!(
        "  Waiting for timeout ({}ms)...",
        controller.get_config().heartbeat_timeout_ms
    );

    let start = std::time::Instant::now();
    let mut mode = OperationMode::Interactive;

    while start.elapsed() < Duration::from_millis(controller.get_config().heartbeat_timeout_ms + 100) {
        mode = controller.tick()?;
        if mode == OperationMode::Passthrough {
            break;
        }
        thread::sleep(Duration::from_millis(10));
    }

    assert_eq!(
        mode,
        OperationMode::Passthrough,
        "Should timeout back to Passthrough Mode"
    );
    println!("✓ Mode switched back to PASSTHROUGH after timeout");

    println!("\n===========================================================");
    println!("All tests PASSED! ✓");
    println!("===========================================================");
    println!("\nSummary:");
    println!("  ✓ Peer Controller starts in Passthrough Mode");
    println!("  ✓ Switches to Interactive Mode when heartbeats received");
    println!("  ✓ Times out back to Passthrough Mode when heartbeats stop");
    println!("  ✓ ZeroMQ communication working correctly");

    Ok(())
}
