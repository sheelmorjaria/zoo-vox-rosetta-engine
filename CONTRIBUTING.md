# Contributing to Animal Vocalization Analysis Framework

Thank you for your interest in contributing to the Animal Vocalization Analysis Framework! This hybrid Python/Rust project enables cross-species communication research using the Universal Rosetta Stone methodology.

## Overview

This framework follows a **hybrid architecture**:
- **Rust (Execution Layer)**: Time-critical operations, signal processing, safety (technical_architecture/)
- **Python (Logic Layer)**: Cognitive intelligence, decision making, learning

## Project Structure

```
src/
├── technical_architecture/     # Rust Execution Layer (415 tests)
├── realtime/                    # Python Logic Layer
├── query_interface/             # High-performance queries
├── semiotics/                   # Cognitive analysis
├── synthesis/                   # Audio synthesis
├── data_import/                 # Data management
├── cognitive_intelligence/      # ML/AI components
├── scientific_validation/       # Validation & testing
├── tests/                       # Comprehensive test suite
└── vocalization_database.json   # Main database (2,882 phrases)
```

## Development Setup

### Prerequisites

**Python:**
- Python 3.9 or later
- pip / conda
- pytest

**Rust:**
- Rust 1.70 or later
- Cargo

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd src

# Install Python dependencies
pip install -r requirements.txt  # (if requirements.txt exists)

# Import vocalization data
python data_import/import_vocalization_data.py

# Run tests
python -m pytest tests/ -v

# For Rust components
cd technical_architecture
cargo build
cargo test
```

## Code Style

### Python

Follow PEP 8:
```bash
# Format code
black .
isort .

# Check style
flake8
pylint
```

### Rust

```bash
cd technical_architecture

# Format code
cargo fmt

# Check style
cargo fmt --check
cargo clippy
```

## Testing

### Python Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_rosetta_stone_base.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

### Rust Tests

```bash
cd technical_architecture

# Run all tests (415 tests)
cargo test

# Run specific module
cargo test module_name

# Run benchmarks
cargo run --example benchmark_peer_controller --release
```

## Test Coverage

**Rust (technical_architecture/):**
- 415 tests passing
- 21 source modules
- Core: 179 tests
- Production: 142 tests
- Field: 187 tests

**Python:**
- 50+ test files
- Comprehensive coverage of all modules

## Adding New Features

### 1. Python Logic Layer Features

For cognitive intelligence, analysis, or query features:
- Write tests first (TDD approach)
- Follow existing patterns in similar modules
- Document with docstrings
- Include type hints
- Add to tests/ directory

### 2. Rust Execution Layer Features

For signal processing, safety, or field deployment:
- Follow TDD_PLAN_PRODUCTION_FEATURES.md or TDD_PLAN_FIELD_FEATURES.md
- Write comprehensive tests (aim for 20+ tests per module)
- Use zero-copy patterns for audio
- Document safety constraints
- Include performance benchmarks

### 3. Cross-Language Integration

For Python-Rust integration via PyO3:
- Update technical_architecture/src/lib.rs
- Add Python wrapper functions
- Test both Python and Rust sides
- Document API changes in CLAUDE.md

## Submitting Changes

1. Fork the repository
2. Create a feature branch:
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. Make your changes
4. Add/update tests
5. Ensure all tests pass:
   ```bash
   python -m pytest tests/ -v
   cd technical_architecture && cargo test
   ```
6. Format code:
   ```bash
   black .
   isort .
   cd technical_architecture && cargo fmt
   ```
7. Commit changes:
   ```bash
   git commit -m "feat: add amazing feature"
   ```
8. Push and open Pull Request

## Commit Message Format

```
feat: add new feature
fix: correct bug in module
docs: update documentation
test: add tests for feature
refactor: restructure code
perf: improve performance
```

## Documentation

- Update README.md for user-facing changes
- Update CLAUDE.md for API changes
- Add docstrings to Python functions
- Add Rust documentation (/// or //!)
- Keep examples up to date

## Scientific Considerations

This is a **scientific research framework** for animal communication:

- **Ethical Compliance**: All features must respect IACUC guidelines
- **Safety First**: Rust layer enforces safety constraints
- **Data Integrity**: Maintain provenance trails
- **Reproducibility**: Document methods and parameters
- **Species Respect**: Follow ethical guidelines for each species

## Species-Specific Guidelines

| Species | Considerations |
|---------|----------------|
| Marmoset | Harmonic communication, small primate ethics |
| Egyptian Fruit Bat | FM sweep, nocturnal species |
| Dolphin | Whistle communication, marine mammal ethics |
| Chimpanzee | Mixed communication, great ape ethics |
| Zebra Finch | Songbird, passerine ethics |
| Sperm Whale | Low frequency, marine mammal ethics |

## Performance Guidelines

### Python
- Use vectorization (NumPy/Pandas)
- Avoid loops in hot paths
- Profile before optimizing
- Use caching for expensive operations

### Rust
- Use zero-copy patterns for audio
- Prefer channels over mutexes
- Profile with criterion for benchmarks
- Avoid allocations in hot paths

## Safety Considerations

This is a **safety-critical system** for animal research:

- **Fail Safe**: Always default to safe state (muted audio)
- **IACUC Compliance**: Never bypass legal requirements
- **Validation**: Validate all inputs
- **Logging**: Log all safety-relevant events
- **Testing**: Comprehensive test coverage required

## Questions?

- Open an issue for bugs or feature requests
- Check CLAUDE.md for detailed API documentation
- Review existing code for patterns
- Contact: sheelmorjaria@gmail.com

## License

By contributing, you agree that your contributions will be licensed under **CC BY-ND 4.0 International**.

---

Thank you for contributing to animal communication research!
