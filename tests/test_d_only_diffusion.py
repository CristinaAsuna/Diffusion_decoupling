import torch
import unittest

from consistent_cdm.model.diffusion import GaussianConditionalDiffusion
from consistent_cdm.model.flow_matching import ConditionalFlowMatching


def _model():
    return GaussianConditionalDiffusion(
        condition_channels=1, target_channels=1, image_size=32, inner_channel=32,
        channel_mults=[1, 2], attn_res=[], res_blocks=1, dropout=0.0,
        beta_schedule={"schedule": "cosine", "n_timestep": 10}, prediction_type="v_prediction",
        target_parameterization_cfg={"type": "d_only", "difference_scale": 0.3305510432, "already_standardized": True, "clamp_decoded": False},
        branch_decoder_cfg={"enabled": False}, consistency_cfg={"enabled": False},
        x0_reconstruction_cfg={"enabled": True, "lambda": 0.05, "loss": "l1", "clamp": False},
        min_snr_cfg={"enabled": True, "gamma": 5.0}, difference_cfg={"enabled": False}, difference_edge_cfg={"enabled": False},
        data_norm_cfg={"normalize": "fixed_range_m11", "value_range": [0.0, 6.0]}, num_head_channels=32,
    )


class DOnlyDiffusionTest(unittest.TestCase):
    def test_diffusion_samplers_preserve_d_outside_unit_range(self):
        model = _model().eval()
        model.predict_start_from_model_output = lambda y_t, **kwargs: torch.full_like(y_t, 3.0)
        condition = torch.zeros(1, 1, 32, 32)
        with torch.no_grad():
            ddim = model.sample_ddim(condition, sample_steps=1)
            dpm = model.sample_dpm_solver_pp(condition, sample_steps=1)
        self.assertTrue(torch.all(ddim == 3.0))
        self.assertTrue(torch.all(dpm == 3.0))

    def test_flow_matching_is_d_only_and_fixed_latent_is_deterministic(self):
        model = ConditionalFlowMatching(
            condition_channels=1, target_channels=1, image_size=32, inner_channel=32,
            channel_mults=[1, 2], attn_res=[], res_blocks=1, dropout=0.0,
            beta_schedule={"schedule": "cosine", "n_timestep": 10}, prediction_type="v_prediction",
            target_parameterization_cfg={"type": "d_only", "difference_scale": 1.0, "already_standardized": True},
            branch_decoder_cfg={"enabled": False}, data_norm_cfg={"normalize": "fixed_range_m11", "value_range": [0.0, 6.0]},
            flow_cfg={"solver": "heun", "steps": 2},
        ).eval()
        condition = torch.zeros(1, 1, 32, 32)
        torch.manual_seed(7); first = model.sample_flow(condition, sample_steps=2)
        torch.manual_seed(7); second = model.sample_flow(condition, sample_steps=2)
        self.assertTrue(model.is_d_only)
        self.assertTrue(torch.equal(first, second))
    def test_standardized_d_only_is_single_channel_and_unclamped(self):
        model = _model()
        condition = torch.zeros(2, 1, 32, 32)
        target_d = torch.full((2, 1, 32, 32), 1.5)
        self.assertTrue(torch.equal(model.encode_target(target_d, condition), target_d))
        self.assertTrue(torch.allclose(model.raw_d_from_target(target_d), target_d * 0.3305510432))
        self.assertTrue(torch.equal(model.decode_target(target_d, condition), target_d))

    def test_d_only_rejects_branch_specific_losses_and_decoder(self):
        base = dict(
        condition_channels=1, target_channels=1, image_size=32, inner_channel=32, channel_mults=[1, 2], attn_res=[],
        res_blocks=1, dropout=0.0, beta_schedule={"schedule": "cosine", "n_timestep": 10},
        target_parameterization_cfg={"type": "d_only", "difference_scale": 1.0},
    )
        with self.assertRaisesRegex(ValueError, "branch_decoder"):
            GaussianConditionalDiffusion(**base, branch_decoder_cfg={"enabled": True})
        with self.assertRaisesRegex(ValueError, "forbids difference"):
            GaussianConditionalDiffusion(**base, branch_decoder_cfg={"enabled": False}, difference_cfg={"enabled": True})


if __name__ == "__main__":
    unittest.main()
