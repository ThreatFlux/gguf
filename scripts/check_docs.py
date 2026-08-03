#!/usr/bin/env python3
"""Check repository Markdown links and retired documentation patterns."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parent.parent
ROOT_DOCS = [
    ROOT / "README.md",
    ROOT / "CHANGELOG.md",
    ROOT / "CONTRIBUTING.md",
    ROOT / "SECURITY.md",
    ROOT / "TESTING_GUIDE.md",
]
MARKDOWN_FILES = ROOT_DOCS + sorted((ROOT / "docs").rglob("*.md"))
LINK_RE = re.compile(r"!?\[[^\]]*]\(([^)]+)\)")
HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)

RETIRED_PATTERNS = {
    "retired repository URL/name": re.compile(r"ThreatFlux/gguf_rs|cd\s+gguf_rs\b"),
    "nonexistent high-level type": re.compile(
        r"\bGGUFFile::|\buse\s+gguf_rs_lib::\{GGUFFile"
    ),
    "unrelated crates.io install": re.compile(r"\bcargo\s+install\s+gguf(?:\s|$)"),
    "stale exact dependency example": re.compile(r'gguf-rs-lib\s*=\s*"0\.2\.[0-9]+"'),
}


def github_slug(heading: str) -> str:
    """Approximate GitHub's heading slug for the headings used in these docs."""
    heading = re.sub(r"<[^>]+>", "", heading)
    heading = heading.replace("`", "").strip().lower()
    heading = re.sub(r"[^\w\- ]", "", heading)
    return re.sub(r"\s+", "-", heading)


def anchors(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    result: set[str] = set()
    seen: dict[str, int] = {}
    for heading in HEADING_RE.findall(text):
        base = github_slug(heading)
        count = seen.get(base, 0)
        seen[base] = count + 1
        result.add(base if count == 0 else f"{base}-{count}")
    return result


def local_target(raw_target: str, source: Path) -> tuple[Path, str] | None:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    target = target.split(maxsplit=1)[0]
    if target.startswith(("http://", "https://", "mailto:", "tel:")):
        return None

    path_part, separator, fragment = target.partition("#")
    destination = (
        source if not path_part else (source.parent / unquote(path_part)).resolve()
    )
    return destination, unquote(fragment) if separator else ""


def main() -> int:
    errors: list[str] = []

    for path in MARKDOWN_FILES:
        if not path.is_file():
            errors.append(f"missing documentation file: {path.relative_to(ROOT)}")
            continue

        text = path.read_text(encoding="utf-8")
        relative = path.relative_to(ROOT)

        for description, pattern in RETIRED_PATTERNS.items():
            match = pattern.search(text)
            if match:
                line = text.count("\n", 0, match.start()) + 1
                errors.append(f"{relative}:{line}: {description}: {match.group(0)!r}")

        for match in LINK_RE.finditer(text):
            resolved = local_target(match.group(1), path)
            if resolved is None:
                continue

            destination, fragment = resolved
            line = text.count("\n", 0, match.start()) + 1
            try:
                display = destination.relative_to(ROOT)
            except ValueError:
                errors.append(
                    f"{relative}:{line}: link escapes repository: {match.group(1)}"
                )
                continue

            if not destination.exists():
                errors.append(f"{relative}:{line}: missing link target: {display}")
                continue

            if (
                fragment
                and destination.is_file()
                and destination.suffix.lower() == ".md"
            ):
                if fragment not in anchors(destination):
                    errors.append(
                        f"{relative}:{line}: missing anchor #{fragment} in {display}"
                    )

    if errors:
        print("Documentation checks failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print(f"Documentation checks passed ({len(MARKDOWN_FILES)} files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
