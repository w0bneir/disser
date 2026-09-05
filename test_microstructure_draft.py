import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import signal
import soundfile as sf

from run_microstructure_draft import PROTOCOL, build_package


SR = 44_100


def _write_take(path: Path, seed: int, colour: float) -> None:
    rng = np.random.default_rng(seed)
    frames = int(round(2.7 * SR))
    time = np.arange(frames, dtype=np.float64) / SR
    onset = int(round(0.05 * SR))
    envelope = np.where(time >= 0.05, np.exp(-3.6 * (time - 0.05)), 0.0)
    noise = signal.lfilter([1.0], [1.0, -colour], rng.standard_normal(frames))
    mono = 0.025 * noise * envelope
    mono[onset : onset + 4] += np.asarray([0.72, -0.41, 0.18, -0.07])
    side = 0.04 * np.roll(mono, 6 + seed % 4)
    audio = np.stack((mono + side, mono - side), axis=1)
    sf.write(path, audio, SR, subtype="PCM_16")


class MicrostructureDraftTests(unittest.TestCase):
    def test_builds_atomic_blind_package(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "takes"
            source.mkdir()
            for group in (1, 2):
                for take in (1, 2, 3):
                    _write_take(
                        source / f"SHOT {group}.{take}.wav",
                        seed=group * 10 + take,
                        colour=0.88 + 0.01 * take + 0.015 * group,
                    )
            target = root / "result"
            result = build_package(
                source,
                target,
                experiment_group="1",
                reference_name="SHOT 1.2.wav",
                seed=123,
                events=3,
                interval_ms=500.0,
            )
            self.assertTrue(result["verification"]["passed"])
            self.assertTrue((target / "experiment" / "blind_test.html").is_file())
            self.assertTrue((target / "analysis" / "microstructure_profile.json").is_file())
            self.assertTrue((target / "private" / "blind_key.json").is_file())
            manifest = json.loads((target / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["protocol"], PROTOCOL)
            self.assertEqual(manifest["profile_summary"]["files"], 6)
            self.assertEqual(manifest["profile_summary"]["within_group_pair_count"], 6)
            self.assertEqual(len(list((target / "experiment").glob("*.wav"))), 16)
            self.assertEqual(
                manifest["candidate_metrics"]["micro_median"]["leading_attack_max_abs_error"],
                0.0,
            )

    def test_refuses_to_overwrite_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "existing"
            target.mkdir()
            with self.assertRaises(FileExistsError):
                build_package(root, target)


if __name__ == "__main__":
    unittest.main()
