"""Opt-in CPU closure from synthetic CBCT to condition-only trained inference.

Run with DECOUPLE_TRAINING_SMOKE=1 and the preprocessing extra installed.
This test uses generated phantoms only; it does not access an external cohort.
"""
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch


@unittest.skipUnless(os.environ.get("DECOUPLE_TRAINING_SMOKE") == "1",
                     "Set DECOUPLE_TRAINING_SMOKE=1 for CPU preprocessing/training smoke")
class EndToEndPipelineTests(unittest.TestCase):
    def test_cbct_to_training_to_condition_only_inference(self):
        import numpy as np
        import SimpleITK as sitk

        from cbct_preprocess.config import load_config
        from cbct_preprocess.pipeline import process_case
        from consistent_cdm.cli import main
        from consistent_cdm.runtime import read_prepared, verify_prepared
        from consistent_cdm.utils.checkpointing import load_checkpoint_file
        import torch

        config = load_config()
        config["nnunet"]["enabled"] = False
        config["projection"].update(
            device="cpu", sad_mm=100., sid_mm=150., raw_size=[32,32],
            detector_pixel_spacing_mm=[.5,.5], step_size_mm=1.,
            chunk_size=128, output_size=[32,32], crop={"mode":"none"},
            variants=[{"name":"std","primary_angle_deg":0.},
                      {"name":"aug","primary_angle_deg":5.}],
        )
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            inputs, projections = root/"inputs", root/"artifacts/projections"
            inputs.mkdir()
            # Four independent patients, each with two correlated views. Bright
            # label-8/high-LPS-X tissue makes branch provenance observable.
            for index in range(4):
                case = f"Patient{index:02d}"
                array = np.full((9,9,9), -1000, dtype=np.int16)
                array[1:8,1:8,1:4] = 100 + 20*index
                array[1:8,1:8,4:8] = 900 + 30*index
                labels = np.zeros_like(array, dtype=np.uint8)
                labels[4,4,6], labels[4,4,2] = 8,9
                image_path, seg_path = inputs/f"{case}.nii", inputs/f"{case}_seg.nii.gz"
                sitk.WriteImage(sitk.GetImageFromArray(array), str(image_path))
                sitk.WriteImage(sitk.GetImageFromArray(labels), str(seg_path))
                manifest = process_case(image_path, seg_path, projections, config)
                self.assertEqual(manifest["status"], "complete")
                left = np.load(projections/case/"projections/std_left.npy")
                right = np.load(projections/case/"projections/std_right.npy")
                self.assertGreater(float(left.sum()), float(right.sum()))

            # Exercise public entrypoints, including prepared source-hash checks.
            prepared_dir = root/"artifacts/prepared"
            main(["prepare-data", "--data-root", str(projections), "--output", str(prepared_dir),
                  "--image-size", "32", "--train-ratio", "0.75"])
            prepared = read_prepared(prepared_dir)
            verify_prepared(prepared)
            self.assertEqual(len(prepared["split"]["train"]), 3)
            self.assertEqual(len(prepared["split"]["val"]), 1)
            self.assertFalse(set(prepared["split"]["train"]) & set(prepared["split"]["val"]))
            self.assertEqual(len(prepared["audit"]), 8)
            self.assertEqual(prepared["calibration"]["train_case_ids"], prepared["split"]["train"])
            self.assertGreater(prepared["d_normalization"]["scale"], 0)

            run_id = "20000101__phantom__resunet__smoke"
            with patch.dict(os.environ, {"CONSISTENT_CDM_RUNS_ROOT":str(root/"runs"),
                                         "OMP_NUM_THREADS":"1", "MKL_NUM_THREADS":"1"}):
                main(["train", "--family", "resunet", "--prepared", str(prepared_dir),
                      "--run-id", run_id, "--max-steps", "1", "--device", "cpu", "--tiny"])
            checkpoint = root/"runs/resunet"/run_id/"checkpoints/last.pt"
            self.assertTrue(checkpoint.is_file())
            saved = load_checkpoint_file(checkpoint, torch.device("cpu"))
            self.assertEqual(saved["step"], 1)
            self.assertEqual(saved["config"]["prepared_id"], prepared["prepared_id"])
            self.assertIn("ema_model", saved)

            # Deliberately provide just a copied Full NPY: infer cannot discover
            # corresponding targets or rely on a paired dataset interface.
            condition = root/"condition_only.npy"
            val_case = prepared["split"]["val"][0]
            shutil.copyfile(projections/val_case/"projections/std_full.npy", condition)
            output = root/"artifacts/inference"
            main(["infer", "--checkpoint", str(checkpoint), "--input", str(condition),
                  "--output", str(output), "--device", "cpu"])
            manifest = json.loads((output/"manifest.json").read_text())
            self.assertEqual(manifest["family"], "resunet")
            self.assertEqual(manifest["scope"], "condition_only_no_paired_accuracy_metrics")
            self.assertEqual(manifest["images"][0]["transform"]["mapping"], "raw")
            prediction = output/manifest["images"][0]["output"]
            full,d,left,right = [np.load(prediction/f"{name}.npy") for name in ("full","d","left","right")]
            for array in (full,d,left,right):
                self.assertEqual(array.shape, (32,32))
                self.assertEqual(array.dtype, np.float32)
                self.assertTrue(np.isfinite(array).all())
            np.testing.assert_allclose(full, np.load(condition), atol=0, rtol=0)
            np.testing.assert_allclose(left+right, full, atol=1e-6)
            np.testing.assert_allclose((left-right)/2, d, atol=1e-6)
            self.assertTrue((prediction/"preview.png").is_file())


if __name__ == "__main__":
    unittest.main()
