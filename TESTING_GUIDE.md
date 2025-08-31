# GGUF Library Testing Guide

This document describes the comprehensive testing infrastructure for the GGUF library, designed to achieve >90% code coverage.

## Test Structure

The test suite is organized into several categories:

### 1. Unit Tests (`tests/unit/`)
- **Format Tests** (`format_tests.rs`): Tests for GGUF format components
  - Constants validation
  - Header serialization/deserialization
  - Metadata operations
  - Alignment calculations
  - Endian conversion
  - Type system validation

- **Tensor Tests** (`tensor_tests.rs`): Tests for tensor operations
  - Tensor type validation
  - Shape calculations
  - Data type conversions
  - Quantization handling
  - Size calculations

- **Reader Tests** (`reader_tests.rs`): Tests for file reading
  - File reader creation and validation
  - Stream reading operations
  - Tensor data loading
  - Error handling
  - Memory-mapped file access

- **Writer Tests** (`writer_tests.rs`): Tests for file writing
  - File writer operations
  - Stream writing
  - Tensor data writing
  - Alignment and padding
  - Round-trip validation

- **Builder Tests** (`builder_tests.rs`): Tests for high-level builders
  - GGUF builder operations
  - Metadata builder
  - Tensor builder
  - Validation logic
  - Complex model creation

- **Error Tests** (`error_tests.rs`): Comprehensive error handling
  - All error types and variants
  - Error propagation
  - Edge cases and boundary conditions
  - Concurrent access patterns

### 2. Integration Tests (`tests/integration/`)
- **End-to-End Tests** (`end_to_end_tests.rs`): Complete workflows
  - Complex model creation
  - Round-trip data integrity
  - Mixed tensor types
  - Comprehensive metadata

- **Format Conversion Tests** (`format_conversion_tests.rs`): Format handling
  - Version compatibility
  - Endianness handling
  - Alignment scenarios
  - String encoding
  - Large metadata handling

- **Quantization Tests** (`quantization_tests.rs`): Quantization features
  - Quantized tensor handling
  - Block size validation
  - Size calculations

- **Large File Tests** (`large_file_tests.rs`): Performance and scalability
  - Large tensor handling
  - Many small tensors
  - Large metadata sets

- **Compatibility Tests** (`compatibility_tests.rs`): Edge cases
  - Empty files
  - Special characters
  - Unicode handling
  - Zero-dimensional tensors

### 3. Property-Based Tests (`tests/property_based/`)
- **Tensor Properties** (`tensor_property_tests.rs`): Random tensor validation
  - Random shapes and data
  - Round-trip properties
  - Size calculations
  - Quantization alignment

- **Metadata Properties** (`metadata_property_tests.rs`): Metadata validation
  - Random metadata generation
  - Type mixing
  - Size limits
  - Key variations

- **Alignment Properties** (`alignment_property_tests.rs`): Alignment validation
  - Alignment calculations
  - Padding properties
  - Edge cases

### 4. Test Fixtures (`tests/fixtures/`)
- Pre-built test data for various scenarios
- Valid and invalid GGUF files
- Edge case data
- Performance test data

## Running Tests

### Quick Tests
```bash
./scripts/run_quick_tests.sh
```

### Full Test Suite with Coverage
```bash
./scripts/run_tests_with_coverage.sh
```

### Individual Test Categories
```bash
# Unit tests only
cargo test unit

# Integration tests only
cargo test integration

# Property-based tests
cargo test property_based

# Library tests
cargo test --lib
```

## Coverage Measurement

The project uses `cargo-tarpaulin` for coverage measurement:

1. **Installation**: The coverage script automatically installs `cargo-tarpaulin` if needed
2. **Reports**: Generated in HTML and XML formats in `target/coverage/`
3. **Target**: >90% line coverage goal
4. **Exclusions**: Test files, examples, and benchmarks are excluded from coverage

### Coverage Reports
- **HTML Report**: `target/coverage/tarpaulin-report.html`
- **XML Report**: `target/coverage/cobertura.xml`

## Test Configuration

### Cargo.toml Dependencies
```toml
[dev-dependencies]
proptest = "1.4"
quickcheck = "1.0"
quickcheck_macros = "1.0"
criterion = "0.5"
tempfile = "3.8"
tokio-test = "0.4"
```

### Property Test Configuration
- Default: 100 test cases per property
- Reduced cases for complex tests: 10 cases
- Configurable via `PROPTEST_CASES` environment variable

## Test Coverage Strategy

### Code Paths Covered
1. **Happy Paths**: Normal operation scenarios
2. **Error Paths**: All error conditions and edge cases
3. **Boundary Conditions**: Limits and edge values
4. **Concurrent Access**: Thread safety validation
5. **Format Compatibility**: Version and format variations

### Areas of Focus
1. **Data Integrity**: Round-trip validation for all operations
2. **Error Handling**: Comprehensive error scenarios
3. **Performance**: Large data handling
4. **Compatibility**: Format variations and edge cases
5. **Type Safety**: All tensor and metadata types

## Achieving >90% Coverage

### Key Strategies
1. **Comprehensive Unit Tests**: Test every public function and method
2. **Error Path Testing**: Test all error conditions
3. **Integration Testing**: End-to-end workflows
4. **Property-Based Testing**: Random input validation
5. **Edge Case Testing**: Boundary conditions and special cases

### Coverage Gaps to Address
1. **Async Code**: Ensure async paths are tested with `tokio-test`
2. **Error Recovery**: Test error recovery scenarios
3. **Platform-Specific Code**: Test OS-specific code paths
4. **Optimization Paths**: Test performance optimizations

### Monitoring Coverage
1. **Automated Reports**: Coverage reports generated with each test run
2. **Threshold Checking**: Scripts check for >90% coverage goal
3. **CI Integration**: Coverage measurement in continuous integration
4. **Trend Tracking**: Monitor coverage changes over time

## Best Practices

### Test Organization
- One test file per module
- Logical grouping of related tests
- Clear, descriptive test names
- Comprehensive documentation

### Test Quality
- **Independence**: Tests don't depend on each other
- **Repeatability**: Tests produce consistent results
- **Speed**: Tests run quickly for rapid feedback
- **Clarity**: Tests are easy to understand and maintain

### Maintenance
- **Regular Updates**: Keep tests updated with code changes
- **Coverage Monitoring**: Regular coverage analysis
- **Performance Testing**: Ensure tests don't become too slow
- **Documentation**: Keep testing guide updated

## Troubleshooting

### Common Issues
1. **Slow Tests**: Use `cargo test --release` for performance tests
2. **Flaky Tests**: Check for timing dependencies and race conditions
3. **Coverage Gaps**: Use coverage reports to identify untested code
4. **Memory Issues**: Monitor memory usage in large file tests

### Debug Tools
- **Test Output**: Use `-- --nocapture` for detailed test output
- **Selective Testing**: Run specific test patterns
- **Timing**: Use `--time-threshold` to identify slow tests
- **Parallel Control**: Use `--test-threads` to control parallelism

## Future Improvements

### Planned Enhancements
1. **Benchmark Tests**: Performance regression testing
2. **Fuzz Testing**: Random input fuzzing for robustness
3. **Cross-Platform Testing**: Ensure compatibility across platforms
4. **Memory Safety**: Additional memory leak detection
5. **Load Testing**: Stress testing with very large files

This comprehensive testing infrastructure ensures high code quality, reliability, and maintainability of the GGUF library while achieving the >90% coverage goal.