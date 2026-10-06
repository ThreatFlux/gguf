# Releasing GGUF

This runbook covers releases of the `gguf-rs-lib` crate from this repository.
The `gguf-cli` workspace package is not published, and the unrelated crates.io
package named `gguf` is not owned by this project.

## Release contract

- `Cargo.toml` is the source of truth for the library version.
- `gguf-cli/Cargo.toml` must carry the same version for workspace consistency.
- Primary release verification uses Rust 1.99.0, and CI separately enforces the
  declared Rust 1.87 minimum.
- Release tags are annotated or signed and use exactly `v<package-version>`.
- Only `gguf-rs-lib` is published to crates.io. `gguf-cli` stays
  `publish = false`; its prebuilt archives are attached to the GitHub release.
- Release assets (the `gguf-cli` archives, their SHA-256 files, and CycloneDX
  SBOMs for both packages) are attached only after the registry publication
  succeeds or the version is confirmed to be on crates.io already.
- `release.yml` never changes a manifest, commits to `main`, or creates a tag.
  Version commits and tags come from `auto-release.yml` or a maintainer.
- crates.io versions are immutable. A defective version can only be yanked and
  superseded.

## Required repository and registry access

`gguf-rs-lib` uses crates.io trusted publishing. Its crates.io trusted
publisher is the `ThreatFlux/gguf` repository, the `release.yml` workflow, and
the `crates-io` environment, so the workflow file name and the environment name
must not change. The publish job requests a GitHub OIDC token (`id-token:
write`), exchanges it with `rust-lang/crates-io-auth-action` for a short-lived
crates.io token that is revoked when the job ends, and reads no registry secret.
Do not add a `CARGO_REGISTRY_TOKEN` secret back to the workflow.

Keep the GitHub environment named `crates-io`. Restricting its deployments to
`v*` tags, or requiring a reviewer, adds a second gate in front of every
publication.

Protect `v*` tags with a repository ruleset. Ordinary writers must not be able
to create, move, or delete release tags. Configure the release maintainer or
approved automation as the narrow bypass needed to create a new tag.

Protect `main` with required pull requests, review, and the stable CI and
security checks. Set the repository's default workflow token permission to
read-only and prevent GitHub Actions from approving pull requests; individual
jobs in this repository declare the narrower write permissions they need.

On crates.io, ensure `gguf-rs-lib` has at least two accountable owners or an
appropriate organization team, and review the trusted-publisher configuration
on personnel or ownership changes.

## Automatic releases

`auto-release.yml` runs after `CI` and `Security` succeed for a push to `main`.
It calls the ThreatFlux reusable auto-release workflow, which reads the
Conventional Commit subjects since the last tag: `feat:` and `fix:` (or a
breaking change) cut a release, while `ci:`, `build:`, `chore:`, `docs:` and
`test:` do not. A release writes the version commit, the annotated `v<version>`
tag and the GitHub release as the `threatflux-automation` GitHub App, so the
tag push starts `release.yml` by itself. If the App credentials are not
available, the reusable workflow falls back to the workflow token and dispatches
`release.yml` for the new tag instead.

Rehearse the next automatic release without writing anything:

```bash
gh workflow run auto-release.yml --ref main --field dry_run=true
```

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
git tag -s v0.3.1 -m "Release v0.3.1"
git push origin v0.3.1
```

If signed tags are not part of the project's established key-management
process, use `git tag -a` rather than inventing an unverifiable signing
identity.

The tag push starts `.github/workflows/release.yml`, which:

1. requires the tag, requested version, library version, and CLI version to
   agree;
2. requires an annotated tag whose commit is reachable from `origin/main`;
3. tests the workspace and verifies the exact `gguf-rs-lib` package;
4. builds and runs `gguf-cli` for Linux (x86_64 glibc and musl, arm64), macOS
   (arm64, x86_64), and Windows (x86_64), and generates CycloneDX SBOMs;
5. runs in the `crates-io` environment, rechecks that the remote tag object has
   not changed, and publishes only `gguf-rs-lib` through trusted publishing.
   If the version is already on crates.io, it skips the upload only when the
   registry checksum matches the crate this tag packages, and fails otherwise;
6. rechecks the tag again, creates the GitHub release if the tag has none yet,
   and attaches the archives, checksums, and SBOMs. A re-run keeps every asset
   that is already attached and uploads only the missing ones.

Use one trigger per release. A normal human-pushed tag starts the workflow; do
not dispatch a duplicate run. If GitHub did not create a tag-triggered run,
first confirm that no release run is active and that the version is still
unpublished. Only then dispatch the workflow at the tag with the unprefixed
version input:

```bash
gh workflow run release.yml --ref v0.3.1 --field version=0.3.1
```

A dispatch on a branch performs verification only. Publish and GitHub-release
jobs are tag-gated.

Rehearse the whole pipeline on any ref with `dry_run`. A dry run builds every
target, generates the SBOMs, and runs `cargo publish --dry-run`, but it never
enters the `crates-io` environment, publishes, or changes a tag or release. It
warns instead of failing when the requested version differs from the
manifests:

```bash
gh workflow run release.yml --ref main --field version=0.3.0 --field dry_run=true
```

## Version history notes

`0.3.0` is the latest `gguf-rs-lib` release on crates.io and GitHub, and both
workspace manifests are at `0.3.0`, so the next automatic release is `0.3.1` or
later. The repository also contains an annotated public `v0.2.6` tag from the
retired auto-version workflow; a GitHub release record was later attached to it,
but no `0.2.6` crate exists. That version is burned: do not move, delete, or
reuse the tag, and do not publish a crate under it. `0.3.0` was a minor-version
increase because correcting public `#[repr(u32)]` tensor-type discriminants is a
breaking change.

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

Release assets are not attached because that job depends on the publish job.
Check crates.io, fix the cause (for example the trusted-publisher
configuration), and re-run the failed jobs. The publish job skips a version
that is already on crates.io with the same checksum, so a re-run never tries to
upload different source under the same version; a checksum mismatch stops the
release before any asset is attached.

### The published crate is defective

Assess whether users need an advisory or workaround, yank the affected version
when continued selection is harmful, and publish a corrected patch release.
Keep the original tag and release record immutable so the shipped source stays
auditable.
