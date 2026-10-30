"""Fast tests for operational pretrained-checkpoint validation."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch

from consistent_cdm.factories import build_deterministic_generator
from consistent_cdm.runtime import recipe


spec = importlib.util.spec_from_file_location("validate_pretrained", Path(__file__).parents[1]/"scripts/validate_pretrained.py")
validation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validation)


class PretrainedValidationTests(unittest.TestCase):
    def _checkpoint(self, root, *, ema=True):
        cfg=recipe("resunet")
        cfg["model"].update(image_size=32,base_channels=8,channel_mults=[1,2],blocks_per_level=1,
                            attention_resolutions=[],attention_heads=1,norm_groups=4,use_se=False,dropout=0.)
        cfg["dataset"]["train"]["image_size"]=[32,32]
        cfg["dataset"]["train"]["d_only_normalization"]={"scale":.1}
        cfg["calibration"]={"raw_full_quantiles":[.1,2.,5.],"source":"train_patients_only"}
        torch.manual_seed(1234)
        model=build_deterministic_generator(cfg["model"],1,1)
        state={"step":42,"config":cfg,"model":model.state_dict(),
               "optimizer":{"state":{},"param_groups":[{"params":[]}]}}
        if ema:state["ema_model"]=model.state_dict()
        path=root/"checkpoint.pt";torch.save(state,path)
        return path

    def test_checkpoint_inference_repeat_real_mapping_and_loss(self):
        previous_threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);checkpoint=self._checkpoint(root)
                full=np.linspace(0,6,32*32,dtype=np.float32).reshape(32,32)
                source=root/"full.npy";np.save(source,full)
                real=root/"real.tiff";Image.fromarray(np.arange(32*32,dtype=np.uint16).reshape(32,32)*60).save(real)
                args=SimpleNamespace(checkpoint=checkpoint,input=source,output=root/"out",device="cpu",
                                     seed=1234,repeat=True,check_loss=True,real_input=real,calibration=None)
                report=validation.validate(args)
                self.assertEqual(report["family"],"resunet")
                self.assertEqual(report["checkpoint_step"],42)
                self.assertEqual(report["weights"],"ema_model")
                self.assertTrue(report["strict_state_dict_load"])
                self.assertTrue(report["checkpoint_optimizer"]["all_tensors_finite"])
                self.assertEqual(len(report["images"]),2)
                self.assertEqual(report["images"][0]["same_seed_repeat_max_abs_error"],0)
                self.assertEqual(report["images"][1]["transform"]["mapping"],"quantile")
                self.assertFalse(report["images"][0]["native_loss_smoke"]["optimizer_updated"])
                np.testing.assert_array_equal(np.load(root/"out/synthetic/full.npy"),full)
                self.assertTrue((root/"out/synthetic/preview.png").is_file())
                self.assertEqual(json.loads((root/"out/validation.json").read_text())["checkpoint_sha256"],report["checkpoint_sha256"])
                with self.assertRaises(FileExistsError):validation.validate(args)
        finally:
            torch.set_num_threads(previous_threads)

    def test_requires_full_ema_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);checkpoint=self._checkpoint(root,ema=False)
            source=root/"full.npy";np.save(source,np.ones((32,32),np.float32))
            args=SimpleNamespace(checkpoint=checkpoint,input=source,output=root/"out",device="cpu",
                                 seed=1234,repeat=False,check_loss=False,real_input=None,calibration=None)
            with self.assertRaisesRegex(ValueError,"EMA"):
                validation.validate(args)
            self.assertFalse((root/"out").exists())

    def test_native_resume_restores_optimizer_and_ema_without_update(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            checkpoint=self._checkpoint(root)
            state=torch.load(checkpoint,map_location='cpu',weights_only=False)
            model=build_deterministic_generator(state['config']['model'],1,1)
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
            sum(parameter.square().mean() for parameter in model.parameters()).backward()
            optimizer.step()
            state.update(model=model.state_dict(),ema_model=model.state_dict(),optimizer=optimizer.state_dict())
            torch.save(state,checkpoint)
            result=validation._resume_smoke(model,state['config'],checkpoint,'cpu')
            self.assertTrue(result['restoration_tested'])
            self.assertTrue(result['ema_restored'])
            self.assertFalse(result['optimizer_updated'])
            self.assertEqual(result['restored_step'],42)
            self.assertGreater(result['optimizer_parameter_states'],0)

    def test_rejects_nonfinite_optimizer_state(self):
        state={"optimizer":{"state":{0:{"exp_avg":torch.tensor(float("nan"))}},"param_groups":[]}}
        with self.assertRaisesRegex(ValueError,"nonfinite"):
            validation._optimizer_summary(state)


if __name__ == "__main__":
    unittest.main()
