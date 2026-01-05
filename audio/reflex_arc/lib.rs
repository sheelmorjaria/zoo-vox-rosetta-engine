// Empty lib.rs to satisfy Cargo.toml requirements
pub mod audio_processor;
pub mod safety_system;
pub mod watchdog_timer;

pub use audio_processor::*;
pub use safety_system::*;
pub use watchdog_timer::*;