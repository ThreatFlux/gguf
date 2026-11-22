"""Require an untagged version to come from the exact release proposal merge."""

import argparse
import base64
import json
import subprocess
import tomllib


def github_api(path, *, paginate=False):
    command = ["gh", "api", path]
    if paginate:
        command.extend(("--paginate", "--slurp"))
    return json.loads(subprocess.check_output(command, text=True, encoding="utf-8"))


def verify_release_provenance(repository, target_sha):
    repo = f"repos/{repository}"
    manifest = github_api(f"{repo}/contents/Cargo.toml?ref={target_sha}")
    version = tomllib.loads(base64.b64decode(manifest["content"]).decode())["package"]["version"]

    # Normal proposal generation starts at an already-tagged version.
    tag_ref = f"refs/tags/v{version}"
    tag_matches = github_api(f"{repo}/git/matching-refs/tags/v{version}")
    if any(tag["ref"] == tag_ref for tag in tag_matches):
        return

    pages = github_api(f"{repo}/commits/{target_sha}/pulls?per_page=100", paginate=True)
    proposals = [
        pr
        for page in pages
        for pr in page
        if pr["merged_at"]
        and pr["merge_commit_sha"] == target_sha
        and pr["user"]["login"] == "threatflux-automation[bot]"
        and pr["head"]["ref"] == "automation/release"
        and (pr["head"].get("repo") or {}).get("full_name") == repository
        and pr["base"]["ref"] == "main"
        and pr["base"]["repo"]["full_name"] == repository
    ]
    if len(proposals) != 1:
        raise ValueError(
            "An untagged version can be published automatically only at the exact "
            "merge commit of its release proposal. Check the release runbook."
        )

    target = github_api(f"{repo}/commits/{target_sha}")
    proposal_sha = proposals[0]["head"]["sha"]
    proposal = github_api(f"{repo}/commits/{proposal_sha}")
    # The App stages one commit directly on its tested base. The merge must
    # preserve its tree and its first-parent base, including for squash merges.
    if (
        not target["parents"]
        or [parent["sha"] for parent in proposal["parents"]] != [target["parents"][0]["sha"]]
        or proposal["commit"]["tree"]["sha"] != target["commit"]["tree"]["sha"]
    ):
        raise ValueError(
            "The release proposal did not land on its tested base with the proposed tree. "
            "Check the release runbook before tagging."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository")
    parser.add_argument("target_sha")
    args = parser.parse_args()
    verify_release_provenance(args.repository, args.target_sha)


if __name__ == "__main__":
    main()
