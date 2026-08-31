"""CPU-проверки сборки слепого listening-пакета."""

from __future__ import annotations

import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf

from analyze_listening_test import analyze_listening_package
from prepare_listening_test import prepare_listening_package


def _write_tone(path: Path, sample_rate: int, frequency: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    time_axis = np.arange(round(sample_rate * 0.12), dtype=np.float32) / sample_rate
    mono = 0.2 * np.sin(2 * np.pi * frequency * time_axis)
    sf.write(path, np.stack((mono, mono), axis=1), sample_rate, subtype="PCM_24")


class PrepareListeningTestTests(unittest.TestCase):
    def test_package_is_blind_balanced_and_reproducible(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "reference.wav"
            dsp_dir = root / "dsp"
            generation_dir = root / "generation"
            seeds = [17, 42]
            _write_tone(reference, 48_000, 440)
            for index, seed in enumerate(seeds):
                _write_tone(
                    dsp_dir / "case" / f"seed_{seed}" / "dsp.wav",
                    44_100,
                    450 + index * 10,
                )
                _write_tone(
                    generation_dir / "case" / f"seed_{seed}" / "guided.wav",
                    44_100,
                    470 + index * 10,
                )

            outputs = [root / "output_a", root / "output_b"]
            for output in outputs:
                prepare_listening_package(
                    case_id="case",
                    reference_path=reference,
                    dsp_results_dir=dsp_dir,
                    generation_results_dir=generation_dir,
                    output_dir=output,
                    seeds=seeds,
                    randomization_seed=1234,
                    loop_repetitions=2,
                    silence_seconds=0.01,
                )

            public = outputs[0] / "public"
            private = outputs[0] / "private_do_not_open_before_scoring"
            self.assertEqual(len(list((public / "stimuli").glob("*.wav"))), 4)
            package_paths = list((public / "packages").glob("*.wav"))
            self.assertEqual(len(package_paths), 3)
            package_lengths = [sf.info(path).frames for path in package_paths]
            self.assertEqual(len(set(package_lengths)), 1)
            self.assertEqual(sf.info(public / "reference.wav").samplerate, 44_100)

            with (private / "individual_answer_key.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as input_file:
                rows = list(csv.DictReader(input_file))
            self.assertEqual({row["method"] for row in rows}, {"dsp", "reference_sde"})
            self.assertTrue(all(row["stimulus_id"].startswith("S") for row in rows))

            first_key = (private / "individual_answer_key.csv").read_bytes()
            second_key = (
                outputs[1]
                / "private_do_not_open_before_scoring"
                / "individual_answer_key.csv"
            ).read_bytes()
            self.assertEqual(first_key, second_key)

            with self.assertRaises(ValueError):
                analyze_listening_package(outputs[0])

            for score_name in ("individual_scores.csv", "package_scores.csv"):
                score_path = public / score_name
                with score_path.open("r", encoding="utf-8-sig", newline="") as input_file:
                    score_rows = list(csv.DictReader(input_file))
                    fieldnames = list(score_rows[0])
                for row in score_rows:
                    row["listener_id"] = "test_listener"
                    for column in fieldnames:
                        if column.endswith("_1_5"):
                            row[column] = "4"
                with score_path.open("w", encoding="utf-8-sig", newline="") as output_file:
                    writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(score_rows)

            analyze_listening_package(outputs[0])
            with (outputs[0] / "analysis" / "individual_method_summary.csv").open(
                "r", encoding="utf-8-sig", newline=""
            ) as input_file:
                summaries = list(csv.DictReader(input_file))
            self.assertEqual(
                {row["method"] for row in summaries},
                {"dsp", "reference_sde"},
            )

    def test_nonempty_output_directory_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "output"
            output.mkdir()
            (output / "existing.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                prepare_listening_package(
                    case_id="case",
                    reference_path=Path("missing.wav"),
                    dsp_results_dir=Path("dsp"),
                    generation_results_dir=Path("generation"),
                    output_dir=output,
                    seeds=[17],
                    randomization_seed=1,
                    loop_repetitions=1,
                    silence_seconds=0,
                )


if __name__ == "__main__":
    unittest.main()
