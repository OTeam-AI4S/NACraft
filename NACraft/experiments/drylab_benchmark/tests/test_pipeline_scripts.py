import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import drylab_common as dc
from scripts import query_pdb_post2023_na_targets as query_pdb
from scripts import submit_nacraft_designs
from scripts import submit_af3_eval
from scripts import submit_odesign_runs
from scripts import make_slurm_array
from scripts import generate_nacraft_configs


PDB_FIXTURE = """\
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00 90.00           C
ATOM      2  CA  LYS A   2       1.000   0.000   0.000  1.00 90.00           C
HETATM    3  P     A B   1       0.400   0.000   0.000  1.00 80.00           P
HETATM    4  P     C B   2       1.400   0.000   0.000  1.00 80.00           P
END
"""


class PipelineScriptTests(unittest.TestCase):
    def test_parse_pdb_atoms_classifies_protein_and_rna(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "toy.pdb"
            path.write_text(PDB_FIXTURE)

            atoms = dc.parse_structure_atoms(path, target_id="toy")

        self.assertEqual([atom.molecule_type for atom in atoms], ["protein", "protein", "rna", "rna"])

    def test_build_target_manifest_from_structures_reports_lengths_and_chains(self):
        with tempfile.TemporaryDirectory() as tmp:
            structure = Path(tmp) / "toy.pdb"
            structure.write_text(PDB_FIXTURE)
            rows = dc.build_target_manifest_from_structures([structure], release_date="2024-01-01")

        self.assertEqual(rows[0]["target_id"], "toy")
        self.assertEqual(rows[0]["polymer_type"], "rna")
        self.assertEqual(rows[0]["na_length"], 2)
        self.assertEqual(rows[0]["protein_chains"], json.dumps(["A"]))
        self.assertEqual(rows[0]["na_chains"], json.dumps(["B"]))

    def test_collect_af3_summary_json_prefers_chain_plddt(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "T__M__C_summary.json"
            summary.write_text(
                json.dumps(
                    {
                        "iptm": 0.66,
                        "chain_iptm": [0.7, 0.6],
                        "chain_plddt": [88.0, 76.0],
                        "chain_pair_pae_min": [[0.0, 6.0], [5.0, 0.0]],
                    }
                )
            )

            rows = dc.collect_af3_metrics(tmp)

        self.assertEqual(rows[0]["target_id"], "T")
        self.assertEqual(rows[0]["method"], "M")
        self.assertEqual(rows[0]["candidate_id"], "C")
        self.assertEqual(rows[0]["iptm"], 0.66)
        self.assertEqual(rows[0]["plddt_aptamer"], 76.0)
        self.assertEqual(rows[0]["ipae"], 5.5)

    def test_kabsch_rmsd_aligns_translated_phosphate_atoms(self):
        native_protein = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        pred_protein = [(10, 5, 2), (11, 5, 2), (10, 6, 2)]
        native_na = [(0, 0, 1), (1, 0, 1)]
        pred_na = [(10, 5, 3), (11, 5, 3)]

        rmsd = dc.protein_aligned_rmsd(native_protein, pred_protein, native_na, pred_na)

        self.assertLess(rmsd, 1e-6)

    def test_rcsb_query_does_not_mix_return_all_hits_and_paginate(self):
        query = query_pdb.build_query("rna", "2023-01-13")
        options = query["request_options"]

        self.assertTrue(options["return_all_hits"])
        self.assertNotIn("paginate", options)

    def test_rcsb_entries_to_manifest_candidates_extracts_na_metadata(self):
        payload = {
            "data": {
                "entries": [
                    {
                        "rcsb_id": "7WKP",
                        "rcsb_accession_info": {"initial_release_date": "2023-04-26T00:00:00Z"},
                        "polymer_entities": [
                            {
                                "entity_poly": {
                                    "rcsb_entity_polymer_type": "Protein",
                                    "pdbx_seq_one_letter_code_can": "MKT",
                                },
                                "rcsb_polymer_entity_container_identifiers": {"auth_asym_ids": ["A"]},
                            },
                            {
                                "entity_poly": {
                                    "rcsb_entity_polymer_type": "RNA",
                                    "pdbx_seq_one_letter_code_can": "ACGU",
                                },
                                "rcsb_polymer_entity_container_identifiers": {"auth_asym_ids": ["B"]},
                            },
                        ],
                    }
                ]
            }
        }

        rows = dc.rcsb_entries_to_manifest_candidates(payload, polymer_type="rna", structure_root="/tmp/pdb")

        self.assertEqual(rows[0]["target_id"], "7WKP")
        self.assertEqual(rows[0]["na_length"], 4)
        self.assertEqual(rows[0]["native_na_sequence"], "ACGU")
        self.assertEqual(rows[0]["protein_sequence"], "MKT")
        self.assertEqual(rows[0]["structure_path"], "/tmp/pdb/7WKP.pdb")

    def test_nacraft_command_uses_real_main_arguments_and_init_seq(self):
        command = submit_nacraft_designs.build_nacraft_command(
            config_path=Path("/tmp/sequence_guided/T1.yaml"),
            out_root="/tmp/out",
            candidates=400,
            nacraft_entry="NACraft/main.py",
            python_executable="/env/bin/python",
            num_workers=4,
            worker_id=2,
            init_seq="ACGU",
        )

        self.assertIn("/env/bin/python NACraft/main.py", command)
        self.assertIn("--num_designs 400", command)
        self.assertIn("--num_workers 4", command)
        self.assertIn("--worker_id 2", command)
        self.assertIn("-o /tmp/out/sequence_guided/T1", command)
        self.assertIn("--ligandmpnn_seqs 2", command)
        self.assertIn("--init-seq ACGU", command)
        self.assertNotIn("--num-samples", command)

    def test_af3_command_uses_real_predict_only_entry(self):
        command = submit_af3_eval.build_af3_command(
            candidate={
                "target_id": "T1",
                "method": "nacraft_denovo",
                "candidate_id": "cand1",
                "sequence": "ACGU",
            },
            target={"target_id": "T1", "polymer_type": "rna"},
            manifest="/tmp/manifest.csv",
            af3_entry="NACraft/experiments/drylab_benchmark/scripts/run_af3_predict_only.py",
            python_executable="/env/bin/python3",
            out_root="/tmp/af3",
            num_samples=5,
        )

        self.assertIn("NACRAFT_DIR=", command)
        self.assertIn("/env/bin/python3", command)
        self.assertIn("run_af3_predict_only.py", command)
        self.assertIn("--manifest /tmp/manifest.csv", command)
        self.assertIn("--sequence ACGU", command)
        self.assertIn("--polymer-type rna", command)
        self.assertIn("--num-samples 5", command)
        self.assertNotIn("run_af3_predict_only.sh", command)

    def test_slurm_array_script_uses_command_file_and_array_bounds(self):
        script = make_slurm_array.render_slurm_array(
            job_name="nacraft",
            commands_file="/tmp/commands.sh",
            num_commands=80,
            partition="gpu",
            gres="gpu:1",
            time_limit="12:00:00",
            cpus_per_task=8,
            mem="64G",
            log_dir="/tmp/logs",
            concurrency=10,
            workdir="/repo",
        )

        self.assertIn("#SBATCH --array=1-80", script)
        self.assertNotIn("%10", script)
        self.assertIn("#SBATCH --gres=gpu:1", script)
        self.assertIn("cd /repo", script)
        self.assertIn("sed -n \"${SLURM_ARRAY_TASK_ID}p\" /tmp/commands.sh", script)

    def test_odesign_command_uses_real_na_entry(self):
        command = submit_odesign_runs.build_odesign_command(
            target={"target_id": "T1", "polymer_type": "rna"},
            input_json_path="/tmp/T1.json",
            odesign_root="/opt/odesign",
            output_root="/tmp/out",
            samples=200,
            python_env_bin="/env/bin",
            seeds="[42]",
        )

        self.assertIn("cd /opt/odesign", command)
        self.assertIn("PATH=/env/bin:$PATH", command)
        self.assertIn("bash scripts/run_odesign.sh", command)
        self.assertIn("--infer_model_name odesign_base_na_rigid", command)
        self.assertIn("--design_modality rna", command)
        self.assertIn("--input_json_path /tmp/T1.json", command)
        self.assertIn("--N_sample 200", command)
        self.assertNotIn("run_na_design.sh", command)

    def test_clean_output_dir_removes_stale_yaml_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stale = root / "denovo" / "old.yaml"
            keep = root / "notes.txt"
            stale.parent.mkdir()
            stale.write_text("old")
            keep.write_text("keep")

            generate_nacraft_configs.clean_output_dir(root)

            self.assertFalse(stale.exists())
            self.assertTrue(keep.exists())


if __name__ == "__main__":
    unittest.main()
