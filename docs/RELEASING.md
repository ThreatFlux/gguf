# Releasing GGUF

This runbook covers releases of the `gguf-rs-lib` crate from this repository.
The `gguf-cli` workspace package is not published, and the unrelated crates.io
package named `gguf` is not owned by this project.

## Release contract

- `Cargo.toml` is the source of truth for the library version.
- `gguf-cli/Cargo.toml` must carry the same version for workspace consistency.
- Primary release verification uses Rust 1.97.1, and CI separately enforces the
  declared Rust 1.87 minimum.
- Release tags are annotated or signed and use exactly `v<package-version>`.
- Only `gguf-rs-lib` is published to crates.io.
- The release workflow does not publish `gguf-cli` or attach example binaries.
- A GitHub release is created only after the registry publication succeeds.
- The workflow never changes a manifest, commits to `main`, or creates a tag.
- crates.io versions are immutable. A defective version can only be yanked and
  superseded.

## Required repository and registry access

Create a protected GitHub environment named `crates-io`. Restrict deployments
to `v*` tags and require approval from a maintainer who did not initiate the
release. The organization-level `CARGO_REGISTRY_TOKEN` Actions secret must be
available to this repository and belong to an appropriately scoped crates.io
automation identity. The workflow exposes it only to the `cargo publish` step;
never print, persist, or pass it to third-party actions.

Protect `v*` tags with a repository ruleset. Ordinary writers must not be able
to create, move, or delete release tags. Configure the release maintainer or
approved automation as the narrow bypass needed to create a new tag.

Protect `main` with required pull requests, review, and the stable CI and
security checks. Set the repository's default workflow token permission to
read-only and prevent GitHub Actions from approving pull requests; individual
jobs in this repository declare the narrower write permissions they need.

On crates.io, ensure `gguf-rs-lib` has at least two accountable owners or an
appropriate organization team. Rotate the registry credential on personnel or
ownership changes and review repository access to the organization secret
regularly.

## Prepare a release

1. Choose the version from the public API, file-format behavior, and serialized
   compatibility—not only commit labels.
2. Update `CHANGELOG.md` with the release date, user-visible changes, migration
   notes, and comparison link.
3. Set the same version in both workspace manifests and refresh the tracked
   `Cargo.lock`.
4. Confirm the README, crate metadata, repository URLs, and examples describe
   the package as `gguf-rs-lib` and do not suggest installing the unrelated
   `gguf` crate.
5. Run the release checks from the repository root:

   ```bash
   cargo fmt --all -- --check
   cargo clippy --locked --workspace --all-targets --all-features -- -D warnings
   cargo test --locked --workspace --all-features
   cargo test --locked -p gguf-rs-lib --doc --all-features
   RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --all-features --no-deps
   python3 scripts/check_docs.py
   bash scripts/check_package.sh
   cargo deny check
   cargo audit --deny warnings
   cargo package --locked -p gguf-rs-lib --list
   cargo package --locked -p gguf-rs-lib
   cargo publish --locked -p gguf-rs-lib --dry-run
   ```

6. Inspect the package list for credentials, local paths, generated reports,
   fixtures, large model files, and other unintended content.
7. Merge the release-preparation pull request to `main` only after required CI
   and review succeed.

## Create the release

From the verified commit on `main`, create one annotated or signed tag and push
it normally:

```bash
git switch main
git pull --ff-only
git tag -s v0.3.0 -m "Release v0.3.0"
git push origin v0.3.0
```

If signed tags are not part of the project's established key-management
process, use `git tag -a` rather than inventing an unverifiable signing
identity.

The tag push starts `.github/workflows/release.yml`, which:

1. requires the tag, requested version, library version, and CLI version to
   agree;
2. requires an annotated tag whose commit is reachable from `origin/main`;
3. tests the workspace and verifies the exact `gguf-rs-lib` package;
4. confirms the version is absent from crates.io and performs a dry run;
5. waits for approval in the protected `crates-io` environment;
6. rechecks that the remote tag object has not changed and publishes only
   `gguf-rs-lib` with the protected organization registry credential;
7. rechecks the tag again and creates the GitHub release.

Use one trigger per release. A normal human-pushed tag starts the workflow; do
not dispatch a duplicate run. If GitHub did not create a tag-triggered run,
first confirm that no release run is active and that the version is still
unpublished. Only then dispatch the workflow at the tag with the unprefixed
version input:

```bash
gh workflow run release.yml --ref v0.3.0 --field version=0.3.0
```

A dispatch on a branch performs verification only. Publish and GitHub-release
jobs are tag-gated.

## Current version reconciliation

As of 2026-08-03, crates.io and the latest GitHub release are at `0.2.5`. The
workspace manifests are staged at `0.3.0`. The repository also contains an
annotated public `v0.2.6` tag from the retired auto-version workflow, but there
is no `0.2.6` crate or GitHub release. That version is burned: do not move,
delete, or reuse the tag, and do not publish a crate under it. The next release
candidate is `0.3.0`; create `v0.3.0` only from the reviewed release commit on
`main` after all preparation checks pass. The minor-version increase is
required because correcting public `#[repr(u32)]` tensor-type discriminants is
a breaking change; do not ship these changes as `0.2.7`.

## After publication

1. Confirm crates.io shows the expected owner, version, checksum, license,
   repository, README, and features for `gguf-rs-lib`.
2. Confirm docs.rs successfully built the same version and renders the public
   API.
3. Verify the GitHub release points to the immutable tag and contains accurate
   generated notes.
4. Test the published dependency from a clean project using the documented
   feature combinations.
5. Announce only behavior present in the published source.

## Failure and recovery

### Verification fails before publication

Fix the source in a new commit. If the failed tag is already public, do not move
it; bump the version and create a new tag after the fix is merged.

### crates.io publication fails

The GitHub release is not created because it depends on successful registry
publication. Check crates.io before retrying. If the version exists, never try
to upload different source under the same version.

### The published crate is defective

Assess whether users need an advisory or workaround, yank the affected version
when continued selection is harmful, and publish a corrected patch release.
Keep the original tag and release record immutable so the shipped source stays
auditable.
