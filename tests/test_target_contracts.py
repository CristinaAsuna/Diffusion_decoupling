from __future__ import annotations

import unittest
import torch

from consistent_cdm.contracts import TargetContract
from consistent_cdm.factories import build_generative_model, generative_run_group
from consistent_cdm.model.flow_matching import ConditionalFlowMatching


def _raw(value: torch.Tensor) -> torch.Tensor:
    return value


def _identity(value: torch.Tensor) -> torch.Tensor:
    return value


def _full(condition: torch.Tensor) -> torch.Tensor:
    return condition[:, :1]


def test_d_only_contract_is_one_channel_and_rejects_branch_losses() -> None:
    contract = TargetContract.from_config({"type": "d_only", "difference_scale": 0.25, "already_standardized": True}, raw_value_scale=1.0)
    value = torch.ones(2, 1, 4, 4)
    assert torch.allclose(contract.raw_d_from_model_space(value), value * 0.25)
    with unittest.TestCase().assertRaisesRegex(ValueError, "forbids branch_decoder"):
        contract.validate(target_channels=1, branch_decoder=True, difference_loss=False, difference_edge_loss=False)
    with unittest.TestCase().assertRaisesRegex(ValueError, "forbids difference"):
        contract.validate(target_channels=1, branch_decoder=False, difference_loss=True, difference_edge_loss=False)


def _config(generative_type: str = "diffusion") -> dict:
    return {
        "generative": {"type": generative_type},
        "dataset": {"train": {"normalize": "none", "value_range": [0.0, 1.0]}},
        "model": {"inner_channel": 8, "channel_mults": [1, 2], "attn_res": [], "res_blocks": 1},
        "diffusion": {"beta_schedule": {"schedule": "linear", "n_timestep": 4}, "prediction_type": "v_prediction", "target_parameterization": {"type": "d_only", "difference_scale": 0.25, "already_standardized": True}},
        "flow_matching": {"sigma_min": 1e-4},
    }


def test_factory_uses_contract_without_changing_d_only_shape() -> None:
    cfg = _config()
    model, contract = build_generative_model(cfg, condition_channels=1, dataset_target_channels=2, image_size=32)
    assert contract.is_d_only
    assert model.target_channels == 1
    assert generative_run_group(cfg) == "diffusion"
    flow_model, _ = build_generative_model(_config("flow_matching"), condition_channels=1, dataset_target_channels=2, image_size=32)
    assert isinstance(flow_model, ConditionalFlowMatching)
    assert generative_run_group(_config("flow_matching")) == "flow_matching"


class ContractTests(unittest.TestCase):
    def test_contract(self):
        test_d_only_contract_is_one_channel_and_rejects_branch_losses()

    def test_factory(self):
        test_factory_uses_contract_without_changing_d_only_shape()

    def test_legacy_factory_rejected(self):
        cfg = _config()
        cfg["diffusion"]["target_parameterization"]["type"] = "left_right"
        with self.assertRaisesRegex(ValueError, "Only d_only"):
            build_generative_model(cfg, condition_channels=1, dataset_target_channels=2, image_size=32)
