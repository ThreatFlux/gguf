# Security policy

## Supported versions

Security fixes are developed on the default branch, currently staged for
`0.3.0`. Until 0.3.0 is published, `0.2.5` remains the latest published crate
and receives security fixes. Earlier patch releases are not maintained.

| Version | Supported |
| --- | --- |
| Default branch (`0.3.0` prerelease) | Yes |
| `0.2.5` (latest published) | Yes |
| `0.2.0`–`0.2.4` | No; upgrade to `0.2.5` |
| Earlier versions | No |

## Report a vulnerability

Use
[GitHub private vulnerability reporting](https://github.com/ThreatFlux/gguf/security/advisories/new).
Do not open a public issue for an undisclosed vulnerability.

Include, when available:

- the affected crate/CLI version, features, target, and Rust version;
- a minimal reproducer or carefully minimized GGUF file;
- the expected and observed behavior;
- impact and realistic attack preconditions;
- sanitizer, Miri, backtrace, or crash output;
- whether the issue has been disclosed anywhere else.

Do not include private model weights or third-party secrets. If a reproducer is
sensitive, describe it first so maintainers can arrange an appropriate
transfer.

## Relevant security boundaries

Reports are especially useful for:

- memory-safety or soundness problems, including the optional memory-map path;
- parser panics, hangs, unchecked allocation, or arithmetic errors caused by
  untrusted files;
- validation bypasses that allow out-of-bounds or overlapping tensor ranges;
- malicious archive or path handling in the CLI;
- dependency or release-process compromises.

A file being unsupported by this crate, semantically invalid for a model
architecture, or numerically poor is not by itself a vulnerability. The crate
does not execute models, authenticate publishers, or provide a cryptographic
integrity format.

See [safety and validation](docs/safety-and-validation.md) for the checks the
current parser performs and the guarantees it does not provide.

## Disclosure

Please allow maintainers time to investigate and prepare a fix before public
disclosure. Maintainers will coordinate advisories and credit with reporters
when practical.
