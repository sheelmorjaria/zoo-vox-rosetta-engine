# Justfile for Zoo Vox Rosetta Engine CI
# Usage: just ci

# List available recipes
default:
    just --list

# =================================================#
#                DEPENDENCIES                      #
# =================================================#

# Install necessary tools and libraries (run once)
install-ci-tools:
    python3 -m pip install --upgrade pip --break-system-packages
    pip install maturin[patchelf] ruff flake8 pytest --break-system-packages
    pip install -e ".[vision]" --break-system-packages

# =================================================#
#                LINTING                           #
# =================================================#

# Run Rust formatting and clippy checks
lint-rust:
    cd technical_architecture && cargo fmt -- --check
    cd technical_architecture && cargo clippy -- -D warnings

# Run Python linting (Ruff + Flake8)
lint-python:
    ruff format --check .
    ruff check .
    flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

# =================================================#
#                BUILDING                          #
# =================================================#

# Build Python bindings (Release mode)
build-bindings:
    #!/usr/bin/env bash
    set -euo pipefail
    cd technical_architecture
    # Build the wheel
    maturin build --release --features python-bindings --strip
    # Install the wheel (forces reinstall to ensure updates are picked up)
    pip install --force-reinstall target/wheels/technical_architecture*.whl || \
    pip install --force-reinstall target/wheels/technical_architecture*.whl --break-system-packages
    cd ..

# Fast build for development (uses maturin develop instead of wheel)
dev-build:
    cd technical_architecture && maturin develop --release --features python-bindings

# =================================================#
#                TESTING                           #
# =================================================#

# Run Rust Tests (Debug + Release as per CI)
test-rust:
    cd technical_architecture && cargo test --lib --verbose
    cd technical_architecture && cargo test --lib --release --verbose

# Run Python Unit Tests (Respects CI ignore lists)
test-python:
    python3 -m pytest tests/ -v --tb=short \
        --ignore=tests/test_shared_memory_ipc.py \
        --ignore=tests/test_realtime_dependencies.py \
        --ignore=tests/test_17d_metadata_synthesis.py \
        --ignore=tests/archive_experimental/ \
        --ignore=tests/archive/

# Run Integration Tests
test-integration:
    python3 -m pytest tests/test_rust_python_integration.py -v

# =================================================#
#                MAIN PIPELINE                     #
# =================================================#

# Run the FULL CI Pipeline locally
ci: lint-rust lint-python test-rust build-bindings test-python test-integration
    @echo "✅ Local CI Passed! All checks succeeded."