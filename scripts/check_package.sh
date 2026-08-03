#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

for generated_path in lcov.info test_model.gguf test_roundtrip.gguf; do
    if [[ -e "$generated_path" ]]; then
        echo "generated root artifact must not be tracked: $generated_path" >&2
        exit 1
    fi
done

package_list="$(
    cargo package \
        --locked \
        --allow-dirty \
        --list \
        -p gguf-rs-lib
)"

unexpected=()
while IFS= read -r entry; do
    case "$entry" in
        .cargo_vcs_info.json | Cargo.lock | Cargo.toml | Cargo.toml.orig | CHANGELOG.md | LICENSE | README.md | SECURITY.md)
            ;;
        benches/* | docs/* | examples/* | src/* | tests/*)
            ;;
        *)
            unexpected+=("$entry")
            ;;
    esac
done <<< "$package_list"

if ((${#unexpected[@]} > 0)); then
    echo "unexpected files in gguf-rs-lib package:" >&2
    printf '  %s\n' "${unexpected[@]}" >&2
    exit 1
fi

for required in Cargo.toml CHANGELOG.md LICENSE README.md SECURITY.md src/lib.rs; do
    if ! grep -Fxq "$required" <<< "$package_list"; then
        echo "required package file is missing: $required" >&2
        exit 1
    fi
done

if grep -Eq '(^|/)(lcov\.info|test_model\.gguf|test_roundtrip\.gguf)$' <<< "$package_list"; then
    echo "generated artifacts leaked into gguf-rs-lib package" >&2
    exit 1
fi

entry_count="$(wc -l <<< "$package_list" | tr -d ' ')"
echo "Package checks passed ($entry_count files)."
