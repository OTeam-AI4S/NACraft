from __future__ import annotations

import hashlib
import sys
import unittest
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(EXPERIMENT_DIR / "scripts"))

from oga_pipeline import extract_oga_target  # noqa: E402


class OgaTargetTests(unittest.TestCase):
    def test_extracts_complete_504_residue_entity_from_authoritative_cif(self):
        cif_path = (
            REPO_ROOT
            / "RNA适配体设计需求"
            / "需求1 OGA"
            / "Human O-GlcNAcase 5VVO.cif"
        )

        target = extract_oga_target(cif_path)

        self.assertEqual(target["sequence_length"], 504)
        self.assertEqual(len(target["sequence"]), 504)
        self.assertEqual(target["protein_copy_count"], 2)
        self.assertEqual(target["protein_asym_ids"], ["A", "B"])
        self.assertEqual(target["resolved_residue_counts"], {"A": 437, "B": 429})
        self.assertEqual(
            target["sequence_sha256"],
            hashlib.sha256(target["sequence"].encode()).hexdigest(),
        )
        self.assertIn("GGGGSGGGGS", target["sequence"])
        self.assertEqual(target["mutation"], "D175N")


if __name__ == "__main__":
    unittest.main()
