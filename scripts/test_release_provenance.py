import base64
import unittest
from unittest.mock import patch

from check_release_provenance import verify_release_provenance


class ReleaseProvenanceTest(unittest.TestCase):
    repository = "ThreatFlux/gguf"
    repo = f"repos/{repository}"

    def setUp(self):
        self.pr = {
            "merged_at": "2026-01-01T00:00:00Z",
            "merge_commit_sha": "target",
            "user": {"login": "threatflux-automation[bot]"},
            "head": {
                "ref": "automation/release",
                "repo": {"full_name": self.repository},
                "sha": "proposal",
            },
            "base": {"ref": "main", "repo": {"full_name": self.repository}},
        }
        manifest = b'[package]\nversion = "0.3.10"\n'
        self.responses = {
            f"{self.repo}/contents/Cargo.toml?ref=target": {
                "content": base64.b64encode(manifest).decode(),
            },
            f"{self.repo}/git/matching-refs/tags/v0.3.10": [],
            f"{self.repo}/commits/target/pulls?per_page=100": [self.pr],
            f"{self.repo}/commits/target": self.commit("base", "tree", "proposal"),
            f"{self.repo}/commits/proposal": self.commit("base", "tree"),
        }

    @staticmethod
    def commit(base, tree, *other_parents):
        return {
            "parents": [{"sha": sha} for sha in (base, *other_parents)],
            "commit": {"tree": {"sha": tree}},
        }

    def validate(self):
        with patch(
            "check_release_provenance.github_api",
            side_effect=self.responses.__getitem__,
        ):
            verify_release_provenance(self.repository, "target")

    def test_existing_tag_permits_new_proposal(self):
        self.responses[f"{self.repo}/git/matching-refs/tags/v0.3.10"] = [
            {"ref": "refs/tags/v0.3.10"},
        ]
        del self.responses[f"{self.repo}/commits/target/pulls?per_page=100"]
        self.validate()

    def test_fresh_proposal_permits_merge_or_squash(self):
        self.validate()
        self.responses[f"{self.repo}/commits/target"] = self.commit("base", "tree")
        self.validate()

    def test_refreshed_proposal_permits_merge_or_squash(self):
        # v0.7.15 refreshes keep the previous proposal as a second parent.
        self.responses[f"{self.repo}/commits/proposal"] = self.commit(
            "base", "tree", "previous-proposal"
        )
        self.validate()
        self.responses[f"{self.repo}/commits/target"] = self.commit("base", "tree")
        self.validate()

    def test_refreshed_proposal_on_stale_base_is_rejected(self):
        self.responses[f"{self.repo}/commits/proposal"] = self.commit(
            "stale", "tree", "base"
        )
        with self.assertRaisesRegex(ValueError, "tested base"):
            self.validate()

    def test_refreshed_proposal_with_extra_parents_is_rejected(self):
        self.responses[f"{self.repo}/commits/proposal"] = self.commit(
            "base", "tree", "previous-proposal", "unrelated"
        )
        with self.assertRaisesRegex(ValueError, "tested base"):
            self.validate()

    def test_refreshed_proposal_with_changed_tree_is_rejected(self):
        self.responses[f"{self.repo}/commits/proposal"] = self.commit(
            "base", "tree", "previous-proposal"
        )
        self.responses[f"{self.repo}/commits/target"] = self.commit("base", "changed")
        with self.assertRaisesRegex(ValueError, "proposed tree"):
            self.validate()

    def test_later_main_commit_cannot_inherit_untagged_version(self):
        self.pr["merge_commit_sha"] = "earlier"
        self.responses[f"{self.repo}/git/matching-refs/tags/v0.3.10"] = [
            {"ref": "refs/tags/v0.3.100"},
        ]
        with self.assertRaisesRegex(ValueError, "exact merge commit"):
            self.validate()

    def test_stale_proposal_base_is_rejected_even_with_same_tree(self):
        self.responses[f"{self.repo}/commits/proposal"] = self.commit("stale", "tree")
        with self.assertRaisesRegex(ValueError, "tested base"):
            self.validate()

    def test_changed_merge_tree_is_rejected_even_with_fresh_base(self):
        self.responses[f"{self.repo}/commits/target"] = self.commit("base", "changed")
        with self.assertRaisesRegex(ValueError, "proposed tree"):
            self.validate()
