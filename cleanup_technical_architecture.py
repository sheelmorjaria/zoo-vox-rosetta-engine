#!/usr/bin/env python3
"""
Cleanup script to move remaining important files from technical_architecture
"""

import shutil
from pathlib import Path

# Important files to keep/move
important_files = {
    'adaptive_audio_processor.py': 'realtime/adaptive_audio_processor.py',
    'adaptive_context_switcher.py': 'realtime/adaptive_context_switcher.py',
    'adaptive_resonance.py': 'realtime/adaptive_resonance.py',
    'advanced_acoustic_features.py': 'realtime/advanced_acoustic_features.py',
    'advanced_technical_enhancements.py': 'realtime/advanced_technical_enhancements.py',
    'deep_reinforcement_learning.py': 'realtime/deep_reinforcement_learning.py',
    'environmental_convolution.py': 'realtime/environmental_convolution.py',
    'fpga_jetson_acceleration.py': 'realtime/fpga_jetson_acceleration.py',
    'gil_handler.py': 'realtime/gil_handler.py',
    'ieee_1588_ptp.py': 'realtime/ieee_1588_ptp.py',
    'parametric_morphing.py': 'realtime/parametric_morphing.py',
    'shared_memory_ipc.py': 'realtime/shared_memory_ipc.py',
    'thermal_throttling_prevention.py': 'realtime/thermal_throttling_prevention.py',
    'zero_copy_rust.py': 'realtime/zero_copy_rust.py',
    'rust_zero_copy.rs': 'realtime/rust_zero_copy.rs',
    'Cargo.toml': 'realtime/Cargo.toml',
    'Cargo.lock': 'realtime/Cargo.lock',
    'requirements.txt': 'realtime/requirements.txt',
    'setup.py': 'realtime/setup.py',
    'README.md': 'realtime/README.md'
}

# Test files to move
test_files_to_move = [
    'test_adaptive_resonance.py',
    'test_advanced_granular_synthesis.py',
    'test_deep_reinforcement_learning.py',
    'test_deterministic_provenance_logging.py',
    'test_enhanced_microharmonic_synthesizer.py',
    'test_environmental_convolution.py',
    'test_evolutionary_synthesis.py',
    'test_fpga_jetson_acceleration.py',
    'test_gil_handler.py',
    'test_ieee_1588_ptp.py',
    'test_parametric_morphing.py',
    'test_shared_memory_ipc.py',
    'test_thermal_throttling_prevention.py',
    'test_zero_copy_rust.py'
]

def move_files():
    technical_arch_path = Path('/mnt/c/Users/sheel/Desktop/src/technical_architecture')
    Path('/mnt/c/Users/sheel/Desktop/src/realtime')
    tests_path = Path('/mnt/c/Users/sheel/Desktop/src/tests')

    # Move important files
    for src_file, dest_file in important_files.items():
        src = technical_arch_path / src_file
        dest = Path('/mnt/c/Users/sheel/Desktop/src') / dest_file

        if src.exists():
            print(f"Moving {src_file} to {dest_file}")
            shutil.move(str(src), str(dest))
        else:
            print(f"Source file not found: {src_file}")

    # Move test files
    for test_file in test_files_to_move:
        src = technical_arch_path / 'tests' / test_file
        dest = tests_path / test_file

        if src.exists():
            print(f"Moving tests/{test_file} to tests/{test_file}")
            shutil.move(str(src), str(dest))
        else:
            print(f"Test file not found: tests/{test_file}")

    # Clean up empty directories
    try:
        if (technical_arch_path / 'tests').exists() and not (technical_arch_path / 'tests').iterdir():
            shutil.rmtree(technical_arch_path / 'tests')
            print("Removed empty tests directory")
    except:
        pass

    # Check if technical_architecture directory is empty
    if technical_arch_path.exists() and not any(technical_arch_path.iterdir()):
        shutil.rmtree(technical_arch_path)
        print("Removed empty technical_architecture directory")
    else:
        print("technical_architecture directory still contains files:")
        for item in technical_arch_path.iterdir():
            print(f"  - {item.name}")

if __name__ == "__main__":
    move_files()
