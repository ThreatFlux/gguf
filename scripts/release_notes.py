#!/usr/bin/env python3
"""Write the GitHub release notes for one gguf-rs-lib version.

The curated part comes from CHANGELOG.md: the ``## [<version>]`` section when
the changelog already names the version, otherwise the ``## [Unreleased]``
section of the tagged commit, which lists the changes that tag adds. GitHub's
generated pull-request list and comparison link follow when they are given.

The notes end with a marker comment so release.yml can recognise notes it has
already written and leave them (and any later manual edits) unchanged.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

MARKER = "<!-- gguf-release-notes -->"
SECTION_RE = re.compile(r"^## \[(?P<name>[^\]]+)\](?:\s+-\s+.*)?\s*$")
LINK_DEFINITION_RE = re.compile(r"^\[[^\]]+\]:\s+\S+")
VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+(?:-[0-9A-Za-z.-]+)?(?:\+[0-9A-Za-z.-]+)?$")


def changelog_section(text: str, name: str) -> str:
    """Return the body of the ``## [name]`` section, without link definitions."""
    body: list[str] | None = None
    for line in text.splitlines():
        match = SECTION_RE.match(line)
        if match:
            if body is not None:
                break
            if match.group("name").casefold() == name.casefold():
                body = []
            continue
        if body is not None and not LINK_DEFINITION_RE.match(line):
            body.append(line)
    return "\n".join(body or []).strip()


def demote_headings(markdown: str) -> str:
    """Nest GitHub's level-2 generated headings below the changelog headings."""
    return re.sub(r"^## ", "### ", markdown, flags=re.MULTILINE)


def build_notes(version: str, changelog: str, generated: str) -> str:
    curated = changelog_section(changelog, version)
    source = f"[{version}]"
    if not curated:
        curated = changelog_section(changelog, "Unreleased")
        source = "[Unreleased]"

    parts = []
    if curated:
        parts.append(f"{curated}\n\n_From the {source} section of CHANGELOG.md._")
    generated = generated.strip()
    if generated:
        parts.append(demote_headings(generated))
    if not parts:
        parts.append("No changelog entries or pull requests were recorded for this release.")
    parts.append(MARKER)
    return "\n\n".join(parts) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", required=True, help="release version without the v prefix")
    parser.add_argument("--changelog", type=Path, default=Path("CHANGELOG.md"))
    parser.add_argument(
        "--generated",
        type=Path,
        help="GitHub's generated release notes (Markdown) to append",
    )
    args = parser.parse_args()

    if not VERSION_RE.match(args.version):
        parser.error(f"not a semantic version: {args.version!r}")
    changelog = args.changelog.read_text(encoding="utf-8") if args.changelog.is_file() else ""
    generated = args.generated.read_text(encoding="utf-8") if args.generated else ""
    sys.stdout.write(build_notes(args.version, changelog, generated))
    return 0


if __name__ == "__main__":
    sys.exit(main())
