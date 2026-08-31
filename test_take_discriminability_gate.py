from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

from run_take_discriminability_gate import select_discriminability_pairs, verify_gate


PROJECT_DIR = Path(__file__).resolve().parent


class TakeDiscriminabilityGateTests(unittest.TestCase):
    def test_pair_selection_returns_same_near_middle_and_far(self) -> None:
        matrix = np.asarray(
            [
                [0.0, 0.1, 0.7, 1.2],
                [0.1, 0.0, 0.4, 0.9],
                [0.7, 0.4, 0.0, 0.6],
                [1.2, 0.9, 0.6, 0.0],
            ]
        )
        pairs = select_discriminability_pairs([0, 1, 2, 3], matrix, medoid=1)
        self.assertEqual(pairs["same_control"], (1, 1))
        self.assertEqual(pairs["near_pair"], (0, 1))
        self.assertEqual(pairs["far_pair"], (0, 3))
        self.assertNotIn(pairs["median_pair"], {pairs["near_pair"], pairs["far_pair"]})

    def test_runner_builds_verified_blind_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "gate"
            input_dir.mkdir()
            sample_rate = 8_000
            frames = int(1.2 * sample_rate)
            for index in range(1, 6):
                rng = np.random.default_rng(200 + index)
                time = np.arange(frames, dtype=np.float64) / sample_rate
                signal = (
                    np.sin(2.0 * np.pi * (300.0 + 40.0 * index) * time)
                    + 0.2 * rng.standard_normal(frames)
                ) * np.exp(-4.0 * time)
                signal /= max(float(np.max(np.abs(signal))), 1e-12)
                stereo = np.column_stack((0.7 * signal, 0.68 * signal))
                sf.write(input_dir / f"SHOT 1.{index}.wav", stereo, sample_rate, subtype="PCM_24")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_DIR / "run_take_discriminability_gate.py"),
                    "--input-dir",
                    str(input_dir),
                    "--group",
                    "1",
                    "--events",
                    "6",
                    "--interval-ms",
                    "500",
                    "--clip-seconds",
                    "1.0",
                    "--results-dir",
                    str(output_dir),
                ],
                cwd=PROJECT_DIR,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            report = verify_gate(output_dir)
            self.assertTrue(report["passed"], report)
            public = json.loads(
                (output_dir / "experiment" / "manifest_public.json").read_text(encoding="utf-8")
            )
            self.assertEqual(len(public["direct_pairs"]), 4)
            self.assertEqual(len(public["loop_pairs"]), 3)
            self.assertEqual(len(list((output_dir / "experiment").glob("*.wav"))), 12)
            key = json.loads(
                (output_dir / "private_do_not_open_before_scoring" / "blind_key.json").read_text(
                    encoding="utf-8"
                )
            )
            same_codes = [
                blind_id
                for blind_id, value in key["blind_mapping"].items()
                if value.startswith("direct_same_control_")
            ]
            self.assertEqual(len(same_codes), 2)
            first = (output_dir / "experiment" / f"{same_codes[0]}.wav").read_bytes()
            second = (output_dir / "experiment" / f"{same_codes[1]}.wav").read_bytes()
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
