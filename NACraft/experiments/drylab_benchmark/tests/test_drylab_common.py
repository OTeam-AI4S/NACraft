import json
import math
import tempfile
import unittest
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import drylab_common as dc


class DrylabCommonTests(unittest.TestCase):
    def test_select_length_stratified_targets_excludes_visible_targets(self):
        rows = [
            {"target_id": "7WKP", "polymer_type": "rna", "na_length": "20", "release_date": "2024-01-01"},
            {"target_id": "R1", "polymer_type": "rna", "na_length": "20", "release_date": "2024-01-01"},
            {"target_id": "R2", "polymer_type": "rna", "na_length": "35", "release_date": "2024-01-01"},
            {"target_id": "R3", "polymer_type": "rna", "na_length": "65", "release_date": "2024-01-01"},
        ]
        bins = [(10, 30, 1), (31, 50, 1), (51, 75, 1)]

        selected = dc.select_length_stratified_targets(
            rows,
            polymer_type="rna",
            bins=bins,
            exclude_target_ids={"7WKP"},
        )

        self.assertEqual([row["target_id"] for row in selected], ["R1", "R2", "R3"])
        self.assertTrue(all(row["selection_reason"].startswith("length_bin_") for row in selected))

    def test_select_length_stratified_targets_filters_large_and_duplicate_proteins(self):
        rows = [
            {"target_id": "A", "polymer_type": "rna", "na_length": "20", "release_date": "2024-01-01", "protein_sequence": "M" * 401},
            {"target_id": "B", "polymer_type": "rna", "na_length": "21", "release_date": "2024-01-02", "protein_sequence": "K" * 100},
            {"target_id": "C", "polymer_type": "rna", "na_length": "22", "release_date": "2024-01-03", "protein_sequence": "K" * 100},
            {"target_id": "D", "polymer_type": "rna", "na_length": "23", "release_date": "2024-01-04", "protein_sequence": "L" * 100},
        ]

        selected = dc.select_length_stratified_targets(
            rows,
            polymer_type="rna",
            bins=[(10, 30, 2)],
            max_protein_length=400,
            unique_protein_sequence=True,
        )

        self.assertEqual([row["target_id"] for row in selected], ["B", "D"])

    def test_filter_manifest_candidates_removes_large_complexes(self):
        rows = [
            {"target_id": "ok", "protein_sequence": "MKT", "protein_chains": json.dumps(["A"]), "na_chains": json.dumps(["B"]), "na_length": "30"},
            {"target_id": "large", "protein_sequence": "M" * 2001, "protein_chains": json.dumps(["A"]), "na_chains": json.dumps(["B"]), "na_length": "30"},
            {"target_id": "many_chains", "protein_sequence": "M:M:M:M:M", "protein_chains": json.dumps(["A", "B", "C", "D", "E"]), "na_chains": json.dumps(["F"]), "na_length": "30"},
            {"target_id": "ambiguous", "polymer_type": "dna", "native_na_sequence": "ACGX", "protein_sequence": "MKT", "protein_chains": json.dumps(["A"]), "na_chains": json.dumps(["B"]), "na_length": "4"},
        ]

        filtered = dc.filter_manifest_candidates(rows, max_protein_length=2000, max_protein_chains=4, max_na_chains=1)

        self.assertEqual([row["target_id"] for row in filtered], ["ok"])

    def test_compute_hotspots_returns_top_contacts_then_distance(self):
        protein_atoms = [
            dc.AtomRecord("T", "protein", "A", 1, "ALA", "CA", 0.0, 0.0, 0.0),
            dc.AtomRecord("T", "protein", "A", 2, "LYS", "CA", 1.0, 0.0, 0.0),
            dc.AtomRecord("T", "protein", "A", 3, "GLY", "CA", 3.5, 0.0, 0.0),
            dc.AtomRecord("T", "protein", "A", 4, "SER", "CA", 4.5, 0.0, 0.0),
            dc.AtomRecord("T", "protein", "A", 5, "TYR", "CA", 7.0, 0.0, 0.0),
        ]
        na_atoms = [
            dc.AtomRecord("T", "rna", "B", 1, "A", "P", 0.4, 0.0, 0.0),
            dc.AtomRecord("T", "rna", "B", 2, "C", "P", 1.4, 0.0, 0.0),
        ]

        patch, hotspots = dc.compute_hotspots(protein_atoms, na_atoms, cutoff=5.0, top_n=4)

        self.assertEqual(len(patch), 4)
        self.assertEqual([h["residue_key"] for h in hotspots], ["A:2:LYS", "A:1:ALA", "A:3:GLY", "A:4:SER"])

    def test_build_nacraft_config_uses_protocol_terms(self):
        target = {
            "target_id": "R1",
            "polymer_type": "rna",
            "na_length": "42",
            "protein_sequence": "MKT",
            "native_na_sequence": "ACGU",
            "hotspots": json.dumps(["A:12:LYS", "A:34:ARG"]),
        }

        denovo = dc.build_nacraft_config(target, mode="denovo")
        guided = dc.build_nacraft_config(target, mode="similarity_guided", sequence_guidance_weight=0.2)
        ablation = dc.build_nacraft_config(target, mode="antibind_loss")

        self.assertEqual(denovo["predictor"], "boltz")
        self.assertIn("LigandContactLoss", denovo["loss_types"])
        self.assertNotIn("SequenceSimilarityLoss", denovo["loss_types"])
        self.assertEqual(guided["init_seq"], "ACGU")
        self.assertIn("SequenceSimilarityLoss", guided["loss_types"])
        self.assertNotIn("LigandContactLoss", ablation["loss_types"])
        self.assertIn("AntiLigandContactLoss", ablation["loss_types"])
        self.assertFalse(ablation["contact_loss"])

    def test_ranking_score_and_best_of_k_do_not_use_rmsd(self):
        rows = [
            {"target_id": "T", "method": "A", "candidate_id": "a1", "iptm": "0.4", "plddt_aptamer": "70", "ipae": "5", "rmsd": "100"},
            {"target_id": "T", "method": "A", "candidate_id": "a2", "iptm": "0.5", "plddt_aptamer": "80", "ipae": "5", "rmsd": "0.1"},
            {"target_id": "T", "method": "B", "candidate_id": "b1", "iptm": "0.6", "plddt_aptamer": "60", "ipae": "", "rmsd": "0.1"},
        ]

        self.assertAlmostEqual(dc.ranking_score(rows[0]), 0.85)
        best = dc.best_by_method_target(rows)

        self.assertEqual(best[("T", "A")]["candidate_id"], "a2")
        self.assertTrue(math.isclose(dc.ranking_score(best[("T", "B")]), 1.2))

    def test_empty_losses_yaml_roundtrips_as_empty_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            dc.write_simple_yaml(path, {"losses": [], "motifs": [], "contact_loss": False})
            parsed = yaml.safe_load(path.read_text())

        self.assertEqual(parsed["losses"], [])
        self.assertEqual(parsed["motifs"], [])
        self.assertFalse(parsed["contact_loss"])

    def test_manifest_qc_summary_reports_hotspot_counts(self):
        rows = [
            {
                "target_id": "T1",
                "polymer_type": "rna",
                "na_length": "26",
                "selected_hotspots": json.dumps(["A:1:LYS", "A:2:ARG"]),
                "interface_patch": json.dumps(["A:1:LYS", "A:2:ARG", "A:3:SER"]),
                "structure_path": "/missing/T1.pdb",
            }
        ]

        summary = dc.manifest_qc_summary(rows)

        self.assertEqual(summary[0]["hotspot_count"], 2)
        self.assertEqual(summary[0]["interface_patch_size"], 3)
        self.assertFalse(summary[0]["structure_exists"])


if __name__ == "__main__":
    unittest.main()
