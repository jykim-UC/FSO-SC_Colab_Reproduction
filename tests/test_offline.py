"""Source-only synthetic tests. No images, weights, network or Drive required."""
import ast
import hashlib
import json
import math
from pathlib import Path
import socket
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"tools"))
from fso_sc_holdout import build_holdout_manifest, canonical_sha256, verify_holdout_manifest
from fso_sc_bpg_codec import BPGCodec, bytes_to_bits, bits_to_bytes
from fso_sc_channel_profiles import make_channel_profile
from fso_sc_ldpc import SystematicRALDPC


def notebook_sources(name):
    book = json.loads((ROOT/"notebooks"/name).read_text(encoding="utf-8"))
    return book, {c.get("metadata", {}).get("id", c.get("id", str(i))): "".join(c["source"])
                  for i, c in enumerate(book["cells"])}


def synthetic_namespace():
    _, cells = notebook_sources("03_final_holdout.ipynb")
    module = ModuleType("fso_sc_public_offline_test")
    sys.modules[module.__name__] = module
    ns = module.__dict__
    allowed = (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign, ast.FunctionDef, ast.ClassDef)
    for name in ("qam_index_link", "paper_fso_channel_v6", "signal_estimated_receiver", "repair_dsp"):
        tree = ast.parse(cells[name])
        tree.body = [n for n in tree.body if isinstance(n, allowed)]
        exec(compile(ast.fix_missing_locations(tree), name, "exec"), ns)
    return ns


class OfflineSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.network_guard = patch.object(socket.socket, "connect", side_effect=AssertionError("Network disabled in offline tests"))
        cls.network_guard.start()
        torch.set_num_threads(2)
        cls.ns = synthetic_namespace()
        cls.ldpc = SystematicRALDPC()

    @classmethod
    def tearDownClass(cls):
        cls.network_guard.stop()

    def test_notebooks_parse_have_no_outputs_and_use_mounted_drive_only(self):
        for name in ("01_prepare_GEE.ipynb", "02_train.ipynb", "03_final_holdout.ipynb"):
            book, sources = notebook_sources(name)
            for cell in book["cells"]:
                if cell["cell_type"] == "code":
                    self.assertEqual(cell["outputs"], [])
                    self.assertIsNone(cell["execution_count"])
                    ast.parse("\n".join("pass" if line.lstrip().startswith(("!", "%")) else line
                                         for line in "".join(cell["source"]).splitlines()))
            self.assertNotIn("googleapiclient", "\n".join(sources.values()))

    def test_fixed_width_bits_and_all_qam_identity_paths(self):
        ns = self.ns
        indices = torch.arange(16384).reshape(2, 64, 128)
        self.assertTrue(torch.equal(indices, ns["bits_to_indices"](ns["indices_to_bits"](indices))))
        for modulation in ("QPSK", "8QAM", "16QAM"):
            points, labels, _ = ns["qam_constellation"](modulation, torch.device("cpu"))
            self.assertAlmostEqual(float(points.abs().square().mean()), 1., places=6)
            self.assertTrue(torch.equal(ns["qam_demodulate"](ns["qam_modulate"](labels, modulation), modulation), labels))
            received = ns["QAMIndexLink"](ns["IdentityFSOChannel"](), modulation=modulation)(indices)
            self.assertTrue(torch.equal(received, indices))

    def test_ldpc_matrix_frozen_and_clean_decode_is_exact(self):
        info = torch.randint(2, (3, 1536), dtype=torch.uint8, generator=torch.Generator().manual_seed(7))
        coded = self.ldpc.encode_bits(info)
        self.assertFalse(bool(self.ldpc.syndrome(coded).any()))
        decoded, stats = self.ldpc.decode_llr((1-2*coded.float())*25)
        self.assertTrue(torch.equal(decoded, info))
        self.assertTrue(bool(stats["converged"].all()))
        self.assertFalse(bool(stats["iterations"].any()))
        lock = json.loads((ROOT/"config/experiment_lock.json").read_text())
        self.assertEqual(self.ldpc.metadata(), lock["settings"]["ldpc"])

    def test_ldpc_corrects_single_reliable_frame_bit_error(self):
        info = torch.randint(2, (2, 1536), dtype=torch.uint8, generator=torch.Generator().manual_seed(14))
        coded = self.ldpc.encode_bits(info)
        llr = (1-2*coded.float())*20
        llr[:, 23] *= -.1
        decoded, stats = self.ldpc.decode_llr(llr)
        self.assertTrue(torch.equal(decoded, info))
        self.assertTrue(bool(stats["converged"].all()))
        self.assertTrue(bool((stats["iterations"] > 0).all()))
        with self.assertRaises(ValueError):
            self.ldpc.decode_llr(torch.full((1, 2048), float("nan")))

    def test_noise_profiles_are_explicit_and_cannot_be_relabeled(self):
        ns = self.ns
        background = make_channel_profile("paper_eq3_4_background_only", ns["PaperFSOSystemConfig"])
        coherent = make_channel_profile("coherent_receiver_extension_v6", ns["PaperFSOSystemConfig"])
        self.assertFalse(background.include_shot_noise)
        self.assertTrue(coherent.include_shot_noise)
        self.assertFalse(background.include_ase_noise)
        self.assertTrue(coherent.include_ase_noise)
        with self.assertRaises(ValueError):
            make_channel_profile("paper_eq3_4_background_only", ns["PaperFSOSystemConfig"], include_shot_noise=True)

    def test_rect8_dsp_preflight_without_tx_truth(self):
        report = self.ns["rect8_repair_self_test"]("cpu")
        self.assertTrue(report["synthetic_only"])
        self.assertEqual(len(report["conditions"]), 2)
        self.assertTrue(all(r["symbol_errors"] == 0 for r in report["conditions"]))
        metadata = self.ns["rect8_repair_metadata"]()
        self.assertFalse(metadata["uses_true_channel_phase"])
        self.assertFalse(metadata["uses_transmitted_bits_or_symbols"])

    def test_reference_manifest_integrity_and_disjointness(self):
        manifest = json.loads((ROOT/"config/holdout_manifest.json").read_text())
        lock = json.loads((ROOT/"config/experiment_lock.json").read_text())
        self.assertEqual(manifest["manifest_sha256"], lock["holdout_manifest_sha256"])
        body = {k:v for k,v in manifest.items() if k != "manifest_sha256"}
        self.assertEqual(canonical_sha256(body), manifest["manifest_sha256"])
        self.assertEqual(len(manifest["selected_file_sha256"]), 200)
        self.assertFalse(set(manifest["selected_file_sha256"]) & set(manifest["development_file_sha256"]))

    def test_split_is_result_blind_and_renaming_invariant(self):
        pool = {f"fixture{i}.png": hashlib.sha256(str(i).encode()).hexdigest() for i in range(1000)}
        dev = list(pool)[:200]
        manifest = build_holdout_manifest(pool, dev, [pool[n] for n in dev], test_zip_sha256="1"*64,
                                         previous_summary_sha256="2"*64)
        selected = verify_holdout_manifest(manifest, pool)
        renamed = {"new_"+name:digest for name,digest in reversed(list(pool.items()))}
        repeated = verify_holdout_manifest(manifest, renamed)
        self.assertEqual({pool[n] for n in selected}, {renamed[n] for n in repeated})
        self.assertEqual(len(selected), 200)
        self.assertFalse(set(selected) & set(dev))
        tampered = dict(pool)
        tampered[dev[0]] = "3"*64
        with self.assertRaises(ValueError):
            verify_holdout_manifest(manifest, tampered)

    def test_public_defaults_do_not_require_private_weights_and_never_train(self):
        _, cells = notebook_sources("03_final_holdout.ipynb")
        self.assertIn("DATA_MODE = 'own_data'", cells["bpg_setup"])
        self.assertIn("ENFORCE_REFERENCE_MODEL = False", cells["bpg_setup"])
        combined = "\n".join(cells.values())
        self.assertNotIn("train_paper_stage(", combined)
        self.assertNotIn("load_verified_sc_reuse", combined)
        self.assertIn('"development_scores_reused":False', combined)
        self.assertIn("verify_frozen_experiment(protocol)", cells["bpg_prepare"])

    def test_published_final_holdout_is_complete_frozen_and_not_development(self):
        final = json.loads((ROOT/"results/final_holdout_summary.json").read_text(encoding="utf-8"))
        development = json.loads((ROOT/"results/dev_summary.json").read_text(encoding="utf-8"))
        lock = json.loads((ROOT/"config/experiment_lock.json").read_text())
        manifest = json.loads((ROOT/"config/holdout_manifest.json").read_text())
        self.assertEqual(final["evaluation_role"], "final_holdout")
        self.assertIs(final["development_scores_reused"], False)
        protocol = final["protocol"]
        self.assertEqual(protocol["data_mode"], "reference_locked")
        self.assertIs(protocol["reference_model_enforced"], True)
        self.assertEqual(protocol["model_sha256"], lock["reference_model_sha256"])
        self.assertEqual(protocol["experiment_lock_sha256"], canonical_sha256(lock))
        self.assertEqual(protocol["holdout_manifest"], manifest)
        self.assertEqual(sorted(protocol["selected_file_sha256"]), manifest["selected_file_sha256"])
        self.assertFalse(set(protocol["selected_file_sha256"]) & set(development["protocol"]["selected_file_sha256"]))
        for key, value in lock["settings"].items():
            self.assertEqual(protocol[key], value, key)
        self.assertEqual(len(final["results"]), 86)
        self.assertEqual(sum(r["summary"]["image_channel_realizations"] for r in final["results"].values()), 17200)
        for row in final["results"].values():
            stats = row["summary"]
            self.assertEqual(stats["unique_images"], 200)
            self.assertTrue(math.isfinite(stats["lpips_mean"]))
            self.assertNotIn("per_image_lpips", stats)
            for error, total, ratio in (("bit_errors", "valid_bits", "payload_ber"),
                                        ("index_errors", "indices", "index_error_rate"),
                                        ("symbol_errors", "valid_symbols", "symbol_error_rate")):
                count, denominator = stats["counts"][error], stats["counts"][total]
                self.assertGreaterEqual(count, 0)
                self.assertLessEqual(count, denominator)
                if denominator:
                    self.assertAlmostEqual(count/denominator, stats[ratio], places=12)
        for private in ("selected_images", "source_metadata", "source_bpg_sha256"):
            self.assertNotIn(private, protocol)

    def test_training_new_resume_guards_preserve_existing_files(self):
        _, cells = notebook_sources("02_train.ipynb")
        combined = "\n".join(cells.values())
        function = None
        for source in cells.values():
            try:
                nodes = ast.parse(source).body
            except SyntaxError:
                continue
            function = next((n for n in nodes if isinstance(n, ast.FunctionDef) and n.name == "preflight_run_directory"), function)
        self.assertIsNotNone(function)
        ns = {"Path":Path}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "run_guard", "exec"), ns)
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            ns["preflight_run_directory"](directory, "new")
            with self.assertRaises(FileNotFoundError): ns["preflight_run_directory"](directory, "resume")
            evidence = directory/"existing.txt"
            evidence.write_text("preserve me")
            with self.assertRaises(FileExistsError): ns["preflight_run_directory"](directory, "new")
            ns["preflight_run_directory"](directory, "resume")
            self.assertEqual(evidence.read_text(), "preserve me")

    def test_bpg_bytes_roundtrip(self):
        raw = bytes(range(256))
        self.assertEqual(bits_to_bytes(bytes_to_bits(raw)), raw)

    def test_bpg_missing_decoder_fails_explicitly(self):
        # No surrogate codec is substituted if native executables are absent.
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(RuntimeError):
                BPGCodec(encoder=str(Path(folder)/"missing_enc"), decoder=str(Path(folder)/"missing_dec"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
