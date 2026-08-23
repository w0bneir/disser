import json
import tempfile
import unittest
from pathlib import Path

import torch

from run_audiox_experiments import _enable_sequential_cfg, _validate_smoke_gate


class _DummyTransformer:
    def __init__(self):
        self.batch_sizes = []

    def _forward(self, x, _t, cross_attn_cond=None, **_kwargs):
        self.batch_sizes.append(x.shape[0])
        if cross_attn_cond is None:
            delta = 0.0
        else:
            delta = float(cross_attn_cond.mean())
        return x + delta

    def forward(self, x, t, **_kwargs):
        return self._forward(x, t)


class SequentialCfgTests(unittest.TestCase):
    def test_cfg_uses_two_single_batch_forwards(self):
        transformer = _DummyTransformer()
        _enable_sequential_cfg(transformer)
        x = torch.ones((1, 2, 3))
        condition = torch.full((1, 4, 2), 2.0)

        output = transformer.forward(
            x,
            torch.ones(1),
            cross_attn_cond=condition,
            cfg_scale=2.0,
        )

        self.assertEqual(transformer.batch_sizes, [1, 1])
        self.assertTrue(torch.equal(output, torch.full_like(x, 5.0)))

    def test_cfg_one_preserves_original_path(self):
        transformer = _DummyTransformer()
        _enable_sequential_cfg(transformer)
        x = torch.ones((1, 2, 3))

        output = transformer.forward(
            x,
            torch.ones(1),
            cross_attn_cond=torch.full((1, 4, 2), 2.0),
            cfg_scale=1.0,
        )

        self.assertEqual(transformer.batch_sizes, [1])
        self.assertTrue(torch.equal(output, x))


class SmokeGateTests(unittest.TestCase):
    def test_valid_metadata_unlocks_one_full_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model.ckpt"
            checkpoint.write_bytes(b"checkpoint")
            metadata = {
                "protocol_id": "audiox_reference_variation_smoke_v1",
                "steps": 2,
                "checkpoint_bytes": checkpoint.stat().st_size,
                "sequential_cfg": True,
                "deterministic_vae_mean": True,
                "max_cuda_reserved_mib": 4_784.0,
            }
            (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            self.assertEqual(
                _validate_smoke_gate(root, checkpoint)["steps"],
                2,
            )

    def test_high_vram_smoke_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "model.ckpt"
            checkpoint.write_bytes(b"checkpoint")
            metadata = {
                "protocol_id": "audiox_reference_variation_smoke_v1",
                "steps": 2,
                "checkpoint_bytes": checkpoint.stat().st_size,
                "sequential_cfg": True,
                "deterministic_vae_mean": True,
                "max_cuda_reserved_mib": 8_001.0,
            }
            (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            with self.assertRaises(RuntimeError):
                _validate_smoke_gate(root, checkpoint)


if __name__ == "__main__":
    unittest.main()
