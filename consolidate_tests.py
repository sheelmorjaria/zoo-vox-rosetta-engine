#!/usr/bin/env python3
"""
Script to consolidate all test files into a single tests directory
"""

import os
import shutil
from pathlib import Path


def consolidate_test_files():
    # Source directories containing test files
    source_dirs = [
        Path('/mnt/c/Users/sheel/Desktop/src/analysis/rosetta_stone'),
        Path('/mnt/c/Users/sheel/Desktop/src/analysis/rosetta_stone/tests'),
        Path('/mnt/c/Users/sheel/Desktop/src/cognition/field_deployment/tests'),
        Path('/mnt/c/Users/sheel/Desktop/src/cognition/fpga_spinal_cord/tests'),
        Path('/mnt/c/Users/sheel/Desktop/src/realtime'),
        Path('/mnt/c/Users/sheel/Desktop/src/realtime/tests'),
        Path('/mnt/c/Users/sheel/Desktop/src/scientific_validation')
    ]

    # Target directory
    target_dir = Path('/mnt/c/Users/sheel/Desktop/src/tests')
    target_dir.mkdir(exist_ok=True)

    # Files to exclude from consolidation (already in tests or not test files)
    exclude_files = {
        '__init__.py',
        'conftest.py',
        'README.md',
        '.gitignore'
    }

    # Files to move with special handling
    special_files = {
        '/mnt/c/Users/sheel/Desktop/src/realtime/simple_enhanced_test.py': 'test_simple_enhanced.py',
        '/mnt/c/Users/sheel/Desktop/src/realtime/test_combined_microharmonic.py': 'test_combined_microharmonic.py',
        '/mnt/c/Users/sheel/Desktop/src/realtime/test_enhanced_microharmonic.py': 'test_enhanced_microharmonic.py',
        '/mnt/c/Users/sheel/Desktop/src/realtime/test_enhanced_microharmonic_fixed.py': 'test_enhanced_microharmonic_fixed.py',
        '/mnt/c/Users/sheel/Desktop/src/realtime/test_mixed_microharmonic.py': 'test_mixed_microharmonic.py',
        '/mnt/c/Users/sheel/Desktop/src/realtime/test_realtime_dependencies.py': 'test_realtime_dependencies.py',
        '/mnt/c/Users/sheel/Desktop/src/realtime/test_simple_mixed_microharmonic.py': 'test_simple_mixed_microharmonic.py',
    }

    # Move special files first
    for src_file, dest_name in special_files.items():
        src = Path(src_file)
        if src.exists():
            dest = target_dir / dest_name
            print(f"Moving {src_file} to {dest}")
            shutil.move(str(src), str(dest))

    # Find and move all test files from source directories
    for source_dir in source_dirs:
        if not source_dir.exists():
            continue

        for item in source_dir.rglob('*'):
            if item.is_file() and item.name.startswith('test_') and item.suffix == '.py':
                # Skip files that are already in the main tests directory
                if str(item).startswith(str(target_dir)):
                    continue

                # Skip excluded files
                if item.name in exclude_files:
                    continue

                # Move the file
                dest = target_dir / item.name
                print(f"Moving {item} to {dest}")

                # Handle name conflicts
                if dest.exists():
                    counter = 1
                    stem = item.stem
                    while dest.exists():
                        dest = target_dir / f"{stem}_{counter}.py"
                        counter += 1

                shutil.move(str(item), str(dest))

    # Create __init__.py in tests directory if it doesn't exist
    init_file = target_dir / '__init__.py'
    if not init_file.exists():
        init_file.write_text('')
        print("Created __init__.py in tests directory")

    # Clean up empty directories
    print("\nCleaning up empty directories...")
    for source_dir in source_dirs:
        if source_dir.exists():
            try:
                # Remove empty directories recursively
                for dirpath in sorted(Path(root) for root, dirs, files in os.walk(str(source_dir), topdown=False) if not dirs and not files):
                    if dirpath != source_dir:
                        dirpath.rmdir()
                        print(f"Removed empty directory: {dirpath}")
            except Exception as e:
                print(f"Error cleaning up {source_dir}: {e}")

    print("\nTest file consolidation complete!")

if __name__ == "__main__":
    consolidate_test_files()
