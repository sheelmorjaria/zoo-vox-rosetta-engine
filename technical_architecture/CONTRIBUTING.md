# Contributing to Technical Architecture

Thank you for your interest in contributing to the Animal Vocalization Analysis Framework's Rust Execution Layer!

## Development Setup

### Prerequisites

- Rust 1.70 or later ([Install Rust](https://rustup.rs/))
- Cargo (comes with Rust)
- Git

### Building

```bash
# Clone the repository
git clone <repository-url>
cd technical_architecture

# Build in debug mode
cargo build

# Build in release mode (optimized)
cargo build --release

# Run tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific test
cargo test test_name
```

## Code Style

This project follows standard Rust conventions:

```bash
# Format code
cargo fmt

# Check code style
cargo fmt --check

# Run linter
cargo clippy

# Fix linter warnings
cargo clippy --fix
```

## Testing

We maintain high test coverage (415 tests). All new features must include tests:

```bash
# Run all tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run specific module tests
cargo test module_name

# Run doc tests
cargo test --doc
```

### Test Organization

- Unit tests go in the same file as the code, in a `#[cfg(test)]` module
- Integration tests go in the `tests/` directory
- Performance benchmarks go in `examples/benchmark_*.rs`

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`cargo test`)
6. Format your code (`cargo fmt`)
7. Commit your changes (`git commit -m 'Add amazing feature'`)
8. Push to the branch (`git push origin feature/amazing-feature`)
9. Open a Pull Request

## Commit Message Format

Follow conventional commits:

```
feat: add new feature
fix: correct bug in module
docs: update documentation
test: add tests for feature
refactor: restructure code
perf: improve performance
```

## Adding New Features

### 1. Core Modules (synthesis, source_separation, etc.)

For signal processing and audio modules:
- Write tests first (TDD approach)
- Ensure zero-copy patterns where possible
- Document safety constraints
- Include performance benchmarks

### 2. Field Deployment Modules

For environment, power, wildlife monitoring:
- Add comprehensive tests (aim for 20+ tests per module)
- Test environmental condition handling
- Include power budget calculations
- Document failure modes

### 3. Production Features

For IACUC, calibration, dashboard, etc.:
- Follow TDD_PLAN_PRODUCTION_FEATURES.md
- Ensure legal compliance (IACUC)
- Add audit trails
- Include security considerations

## Documentation

- Update README.md for user-facing changes
- Update inline documentation (`///` or `//!`)
- Update CLAUDE.md for developer-facing changes
- Keep examples up to date

## Performance Guidelines

- **Audio processing**: Use zero-copy patterns, avoid allocations
- **Message passing**: Prefer channels over mutexes
- **Concurrency**: Use async/await with tokio
- **Memory**: Reuse buffers, avoid copies

Run benchmarks before and after optimization:

```bash
cargo run --example benchmark_peer_controller --release
```

## Safety Considerations

This is a **safety-critical system**:

- Always validate inputs
- Use types to prevent invalid states
- Fail safe (muting audio is safer than playing)
- Log all safety-relevant events
- Never bypass IACUC compliance

## Questions?

- Open an issue for bugs or feature requests
- Check CLAUDE.md for detailed API documentation
- Review existing code for patterns
- Contact: sheelmorjaria@gmail.com

## License

By contributing, you agree that your contributions will be licensed under the **CC BY-ND 4.0 International** license.
