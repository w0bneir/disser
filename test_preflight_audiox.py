import copy
import json
import unittest
from pathlib import Path

from tools.preflight_audiox import (
    GpuInfo,
    PreflightError,
    parse_nvidia_smi_line,
    validate_config,
)


class AudioXPreflightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config_path = Path("artifacts/AudioX_source/config.json")
        if config_path.is_file():
            cls.config = json.loads(config_path.read_text(encoding="utf-8"))
        else:
            cls.config = {
                "model_type": "diffusion_cond",
                "sample_rate": 44_100,
                "sample_size": 485_100,
                "audio_channels": 2,
                "model": {
                    "pretransform": {
                        "config": {"latent_dim": 64, "downsampling_ratio": 2_048}
                    },
                    "conditioning": {"configs": [{"id": "audio_prompt"}]},
                    "diffusion": {
                        "config": {
                            "io_channels": 64,
                            "embed_dim": 1_536,
                            "depth": 24,
                            "num_heads": 24,
                        }
                    },
                },
            }

    def test_audited_config_passes(self):
        validate_config(copy.deepcopy(self.config))

    def test_changed_depth_fails_closed(self):
        changed = copy.deepcopy(self.config)
        changed["model"]["diffusion"]["config"]["depth"] = 25
        with self.assertRaises(PreflightError):
            validate_config(changed)

    def test_missing_audio_conditioner_fails_closed(self):
        changed = copy.deepcopy(self.config)
        changed["model"]["conditioning"]["configs"] = []
        with self.assertRaises(PreflightError):
            validate_config(changed)

    def test_nvidia_smi_parser(self):
        self.assertEqual(
            parse_nvidia_smi_line("NVIDIA GeForce RTX 5070, 595.79, 12227, 11035"),
            GpuInfo("NVIDIA GeForce RTX 5070", "595.79", 12_227, 11_035),
        )


if __name__ == "__main__":
    unittest.main()
