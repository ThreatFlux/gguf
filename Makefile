.PHONY: all build release check fmt format lint test test-features doc examples \
	docs-check package-check coverage audit clean help ci-local hooks-install workflow-check

CARGO ?= cargo
ACTIONLINT ?= actionlint
YAMLLINT ?= yamllint

all: check

ci-local: check workflow-check
	$(CARGO) build --locked -p gguf-rs-lib --examples --all-features
	$(CARGO) package --locked --allow-dirty -p gguf-rs-lib

hooks-install:
	scripts/install-hooks.sh

workflow-check:
	$(ACTIONLINT)
	$(YAMLLINT) -c .yamllint.yml .github

build:
	$(CARGO) build --locked --workspace --all-features

release:
	$(CARGO) build --locked --workspace --release --all-features

check: fmt lint test test-features doc docs-check package-check

fmt:
	$(CARGO) fmt --all -- --check

format:
	$(CARGO) fmt --all

lint:
	$(CARGO) clippy --locked --workspace --all-targets --all-features -- -D warnings

test:
	$(CARGO) test --locked --workspace --all-features

test-features:
	$(CARGO) check --locked -p gguf-rs-lib
	$(CARGO) check --locked -p gguf-rs-lib --all-features
	$(CARGO) check --locked -p gguf-rs-lib --no-default-features --features alloc
	$(CARGO) test --locked -p gguf-rs-lib --no-default-features --features alloc --test alloc_only
	$(CARGO) check --locked -p gguf-cli --all-features

doc:
	$(CARGO) test --locked -p gguf-rs-lib --doc --all-features
	$(CARGO) doc --locked --workspace --all-features --no-deps

examples:
	$(CARGO) run --locked --example roundtrip_test
	$(CARGO) run --locked --example create_test_gguf
	$(CARGO) run --locked --example inspect_gguf -- target/examples/test-model.gguf

docs-check:
	python3 scripts/check_docs.py

package-check:
	scripts/check_package.sh

coverage:
	scripts/run_tests_with_coverage.sh

audit:
	$(CARGO) audit --locked

clean:
	$(CARGO) clean

help:
	@echo "Targets:"
	@echo "  check          full contributor checks"
	@echo "  ci-local       contributor checks, workflows, examples and archive verification"
	@echo "  hooks-install  install this worktree's pre-push validation hook"
	@echo "  build          build the workspace"
	@echo "  release        build the release profile"
	@echo "  fmt            check formatting"
	@echo "  format         apply formatting"
	@echo "  lint           run Clippy with warnings denied"
	@echo "  test           run workspace tests"
	@echo "  test-features  check supported feature combinations"
	@echo "  doc            test and build API documentation"
	@echo "  examples       run self-contained examples"
	@echo "  docs-check     validate Markdown links and stale patterns"
	@echo "  package-check  verify published crate contents"
	@echo "  coverage       write reports below target/coverage"
	@echo "  audit          run cargo-audit (must be installed)"
	@echo "  clean          remove Cargo build output"
