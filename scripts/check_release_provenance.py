"""Require an untagged version to come from the exact release proposal merge."""

import argparse
import base64
import http.client
import json
import os
import re
import tomllib


def github_api(path):
    connection = http.client.HTTPSConnection("api.github.com", timeout=30)
    try:
        connection.request(
            "GET",
            f"/{path}",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
                "User-Agent": "gguf-release-provenance",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with connection.getresponse() as response:
            if response.status != 200:
                raise RuntimeError(f"GitHub returned HTTP {response.status} for {path}")
            return json.load(response)
    finally:
        connection.close()


def is_release_proposal(pr, repository, target_sha):
    return (
        pr["merged_at"] is not None
        and pr["merge_commit_sha"] == target_sha
        and pr["user"]["login"] == "threatflux-automation[bot]"
        and pr["head"]["ref"] == "automation/release"
        and (pr["head"].get("repo") or {}).get("full_name") == repository
        and pr["base"]["ref"] == "main"
        and pr["base"]["repo"]["full_name"] == repository
    )


def proposal_parents_match_base(parents, base_sha):
    """Accept a first proposal [base] or a refreshed proposal [base, previous].

    Since github_actions v0.7.15 the release runtime keeps the previous proposal
    as a second parent when it refreshes automation/release, so the branch update
    is a fast-forward. The tree is still built from the tested base alone.
    """
    if not parents or parents[0] != base_sha:
        return False
    return len(parents) == 1 or (len(parents) == 2 and parents[1] != base_sha)


def verify_release_provenance(repository, target_sha):
    repo = f"repos/{repository}"
    manifest = github_api(f"{repo}/contents/Cargo.toml?ref={target_sha}")
    version = tomllib.loads(base64.b64decode(manifest["content"]).decode())["package"]["version"]

    # Normal proposal generation starts at an already-tagged version.
    tag_ref = f"refs/tags/v{version}"
    tag_matches = github_api(f"{repo}/git/matching-refs/tags/v{version}")
    if any(tag["ref"] == tag_ref for tag in tag_matches):
        return

    associated_prs = github_api(f"{repo}/commits/{target_sha}/pulls?per_page=100")
    if len(associated_prs) >= 100:
        raise ValueError("The associated pull request list may be incomplete.")
    proposals = [
        pr for pr in associated_prs if is_release_proposal(pr, repository, target_sha)
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
        or not proposal_parents_match_base(
            [parent["sha"] for parent in proposal["parents"]],
            target["parents"][0]["sha"],
        )
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
    if not re.fullmatch(r"[0-9a-f]{40}", args.target_sha):
        parser.error("target_sha must be a full commit SHA")
    verify_release_provenance(args.repository, args.target_sha)


if __name__ == "__main__":
    main()
