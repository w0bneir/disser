import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import signal
import soundfile as sf

from run_perceptual_variation_draft import build_package, verify_package


class PerceptualVariationDraftTests(unittest.TestCase):
    def test_builds_verified_blind_package(self):
        sample_rate = 44_100
        frames = int(2.5 * sample_rate)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            for index in range(4):
                rng = np.random.default_rng(index + 10)
                time = np.arange(frames) / sample_rate
                onset = int(0.02 * sample_rate)
                impulse = np.zeros(frames)
                impulse[onset] = 0.75
                decay = np.where(time >= 0.02, np.exp(-(3.5 + index * 0.3) * (time - 0.02)), 0.0)
                noise = signal.lfilter([1.0], [1.0, -0.92 + index * 0.005], rng.standard_normal(frames))
                mono = impulse + noise * decay * 0.02
                audio = np.stack((mono + 0.08 * np.roll(mono, 6 + index), mono - 0.08 * np.roll(mono, 6 + index)), axis=1)
                sf.write(source / f"SHOT 1.{index + 1}.wav", audio, sample_rate, subtype="PCM_24")
            target = root / "result"
            result = build_package(source, target, group="1", events=5, seed=7)
            self.assertTrue(result["verification"]["passed"])
            self.assertTrue((target / "experiment" / "blind_test.html").is_file())
            self.assertEqual(len(list((target / "experiment").glob("*.wav"))), 12)
            public = json.loads((target / "experiment" / "manifest_public.json").read_text(encoding="utf-8"))
            self.assertEqual(len(public["direct_pairs"]), 5)
            self.assertEqual(set(public["loop_pair"]), {"A", "B"})
            self.assertTrue(verify_package(target)["passed"])

    def test_refuses_existing_result_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(FileExistsError):
                build_package(root, root, group="1")


if __name__ == "__main__":
    unittest.main()
