#!/usr/bin/env bash
set -euo pipefail
repository_root="$(git rev-parse --show-toplevel)"
git config extensions.worktreeConfig true
git config --worktree core.hooksPath "$repository_root/.githooks"
echo "Installed hooks for this worktree."
