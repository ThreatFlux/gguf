#!/usr/bin/env python3
"""Inspect GGUF header prefixes without claiming full-file validation."""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path


HEADER = struct.Struct("<4sIQQ")


def discover(paths: list[Path]) -> list[Path]:
    files: set[Path] = set()
    for path in paths:
        if path.is_dir():
            files.update(
                candidate for candidate in path.rglob("*.gguf") if candidate.is_file()
            )
        else:
            files.add(path)
    return sorted(files)


def read_header(path: Path) -> tuple[dict[str, int | str] | None, str | None]:
    try:
        with path.open("rb") as stream:
            raw = stream.read(HEADER.size)
            file_size = os.fstat(stream.fileno()).st_size
    except OSError as error:
        return None, str(error)

    if len(raw) != HEADER.size:
        return None, f"truncated header: expected {HEADER.size} bytes, found {len(raw)}"

    magic, version, tensor_count, metadata_count = HEADER.unpack(raw)
    if magic != b"GGUF":
        return None, f"invalid magic: {magic!r}"

    return {
        "version": version,
        "tensor_count": tensor_count,
        "metadata_count": metadata_count,
        "file_size": file_size,
    }, None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect GGUF magic/version/count fields; does not validate the full file"
    )
    parser.add_argument("paths", nargs="+", type=Path, help="GGUF files or directories")
    args = parser.parse_args()

    files = discover(args.paths)
    if not files:
        parser.error("no .gguf files found")

    failed = False
    for path in files:
        header, error = read_header(path)
        if error:
            failed = True
            print(f"{path}: ERROR: {error}", file=sys.stderr)
            continue

        if header is None:
            failed = True
            print(f"{path}: ERROR: header parser returned no result", file=sys.stderr)
            continue
        support = (
            "supported" if header["version"] == 3 else "unsupported by gguf-rs-lib"
        )
        print(
            f"{path}: GGUF v{header['version']} ({support}), "
            f"{header['tensor_count']} tensors, "
            f"{header['metadata_count']} metadata entries, "
            f"{header['file_size']} bytes"
        )

    print(
        "Header inspection only; payloads and semantic compatibility were not checked."
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
