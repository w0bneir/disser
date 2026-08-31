from __future__ import annotations

import json
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf

from analyze_natural_pool_ratings import decode_ratings
from run_natural_pool_pilot import _pairwise_html
from verify_natural_pool_package import verify


PROJECT_DIR = Path(__file__).resolve().parent


class NaturalPoolPipelineIntegrationTests(unittest.TestCase):
    def _write_group(self, directory: Path, *, mixed_sample_rate: bool = False) -> None:
        directory.mkdir(parents=True)
        for index in range(1, 6):
            sample_rate = 11_025 if mixed_sample_rate and index == 5 else 8_000
            frames = int(round(1.35 * sample_rate))
            onset = int(round(0.04 * sample_rate))
            time = np.arange(frames - onset, dtype=np.float64) / sample_rate
            rng = np.random.default_rng(100 + index)
            event = (
                0.6 * rng.standard_normal(time.size)
                + 0.4 * np.sin(2.0 * np.pi * (420.0 + 55.0 * index) * time)
            ) * np.exp(-(5.0 + index * 0.4) * time)
            event[0] += 2.0
            event /= max(float(np.max(np.abs(event))), 1e-12)
            audio = np.zeros((frames, 2), dtype=np.float32)
            audio[onset:, 0] = 0.75 * event
            audio[onset:, 1] = 0.72 * event + 0.02 * np.roll(event, index)
            sf.write(directory / f"SHOT 1.{index}.wav", audio, sample_rate, subtype="PCM_24")

    def _run(
        self,
        input_dir: Path,
        output_dir: Path,
        *,
        pool_size: int = 3,
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(PROJECT_DIR / "run_natural_pool_pilot.py"),
                "--input-dir",
                str(input_dir),
                "--experiment-group",
                "1",
                "--events",
                "10",
                "--interval-ms",
                "400",
                "--clip-seconds",
                "1.0",
                "--pool-size",
                str(pool_size),
                "--results-dir",
                str(output_dir),
            ],
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            check=False,
        )

    def _build_valid_package(self, root: Path) -> Path:
        input_dir = root / "input"
        output_dir = root / "valid"
        self._write_group(input_dir)
        completed = self._run(input_dir, output_dir)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        self.assertTrue(verify(output_dir, require_external=True)["passed"])
        return output_dir

    def _copy_and_mutate(self, source: Path, destination: Path, mutate) -> dict[str, object]:
        shutil.copytree(source, destination)
        mutate(destination)
        return verify(destination, require_external=True)

    @staticmethod
    def _load_json(path: Path) -> dict[str, object]:
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _dump_json(path: Path, value: object) -> None:
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

    def test_runner_publishes_verified_package_and_ratings_decode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "pilot"
            self._write_group(input_dir)
            completed = self._run(input_dir, output_dir)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertTrue(output_dir.is_dir())
            self.assertEqual(list(root.glob(".pilot.staging-*")), [])
            verification = verify(output_dir, require_external=True)
            self.assertTrue(verification["passed"], verification)

            key = json.loads(
                (output_dir / "private_do_not_open_before_scoring" / "blind_key.json").read_text(
                    encoding="utf-8"
                )
            )
            public_pairs = json.loads(
                (output_dir / "experiment" / "pairwise_manifest_public.json").read_text(
                    encoding="utf-8"
                )
            )["pairs"]
            document = {
                "protocol": "natural_pool_pairwise_v1",
                "session_id": "integration-session",
                "pairs": {
                    pair_id: {
                        "blind_A": sides["A"],
                        "blind_B": sides["B"],
                        "less_repetitive": "same",
                        "more_natural": "same",
                        "more_consistent": "same",
                        "preferred": "same",
                        "confidence": "3",
                        "comment": "",
                    }
                    for pair_id, sides in public_pairs.items()
                },
            }
            ratings = decode_ratings(key, [document])
            self.assertEqual(ratings["valid_documents"], 1)
            self.assertEqual(len(ratings["comparisons"]), 16)

    def test_failed_build_removes_staging_and_never_publishes_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "pilot"
            self._write_group(input_dir, mixed_sample_rate=True)
            completed = self._run(input_dir, output_dir)
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(output_dir.exists())
            self.assertEqual(list(root.glob(".pilot.staging-*")), [])

    def test_runner_rejects_pool_that_does_not_reduce_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "pilot"
            self._write_group(input_dir)
            completed = self._run(input_dir, output_dir, pool_size=5)
            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(output_dir.exists())
            self.assertEqual(list(root.glob(".pilot.staging-*")), [])

    def test_runner_preserves_preexisting_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "pilot"
            self._write_group(input_dir)
            output_dir.mkdir()
            marker = output_dir / "owned_by_user.txt"
            marker.write_text("keep", encoding="utf-8")
            completed = self._run(input_dir, output_dir)
            self.assertNotEqual(completed.returncode, 0)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
            self.assertEqual(list(root.glob(".pilot.staging-*")), [])

    def test_verify_rejects_tampered_public_metadata_and_empty_inventories(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = self._build_valid_package(root)

            def wrong_metadata(package: Path) -> None:
                path = package / "experiment" / "manifest_public.json"
                manifest = self._load_json(path)
                manifest["sample_rate"] = int(manifest["sample_rate"]) + 1
                self._dump_json(path, manifest)

            def empty_sources(package: Path) -> None:
                path = package / "run_manifest.json"
                manifest = self._load_json(path)
                manifest["files"] = []
                self._dump_json(path, manifest)

            def empty_pool(package: Path) -> None:
                path = next((package / "optimized_pool").glob("group_*/pool_manifest.json"))
                manifest = self._load_json(path)
                manifest["files"] = []
                self._dump_json(path, manifest)

            for name, mutate in (
                ("metadata", wrong_metadata),
                ("sources", empty_sources),
                ("pool", empty_pool),
            ):
                with self.subTest(name=name):
                    report = self._copy_and_mutate(valid, root / name, mutate)
                    self.assertFalse(report["passed"], report)

    def test_verify_rejects_blind_html_private_assets_and_duplicate_cards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = self._build_valid_package(root)

            def private_asset(package: Path) -> None:
                path = package / "experiment" / "pairwise_test.html"
                page = path.read_text(encoding="utf-8")
                page = page.replace(
                    "</main>",
                    '<a href="../private_do_not_open_before_scoring/blind_key.json">x</a></main>',
                )
                path.write_text(page, encoding="utf-8")

            def duplicate_card(package: Path) -> None:
                path = package / "experiment" / "blind_test.html"
                page = path.read_text(encoding="utf-8")
                page = page.replace('data-id="P02"', 'data-id="P01"', 1)
                path.write_text(page, encoding="utf-8")

            for name, mutate in (("private_asset", private_asset), ("duplicate_card", duplicate_card)):
                with self.subTest(name=name):
                    report = self._copy_and_mutate(valid, root / name, mutate)
                    self.assertFalse(report["passed"], report)

    def test_verify_rejects_public_form_or_script_tampering_even_with_updated_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = self._build_valid_package(root)

            def rewrite_and_register(package: Path, filename: str, old: str, new: str) -> None:
                path = package / "experiment" / filename
                page = path.read_text(encoding="utf-8")
                self.assertIn(old, page)
                path.write_text(page.replace(old, new, 1), encoding="utf-8")
                manifest_path = package / "run_manifest.json"
                manifest = self._load_json(manifest_path)
                manifest["experiment"]["public_artifact_hashes"][filename] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                self._dump_json(manifest_path, manifest)

            def iframe_escape(package: Path) -> None:
                rewrite_and_register(
                    package,
                    "pairwise_test.html",
                    "</main>",
                    '<iframe src="../private_do_not_open_before_scoring/blind_key.json"></iframe></main>',
                )

            def option_inversion(package: Path) -> None:
                rewrite_and_register(
                    package,
                    "pairwise_test.html",
                    '<option value="A">A</option>',
                    '<option value="B">A</option>',
                )

            def disabled_save(package: Path) -> None:
                rewrite_and_register(
                    package,
                    "pairwise_test.html",
                    "document.getElementById('save').addEventListener('click',()=>{",
                    "if(false)document.getElementById('save').addEventListener('click',()=>{",
                )

            def encoded_mapping_leak(package: Path) -> None:
                rewrite_and_register(
                    package,
                    "blind_test.html",
                    "</main>",
                    "<p>P01 = perceptual&#95;full</p></main>",
                )

            for name, mutate in (
                ("iframe", iframe_escape),
                ("options", option_inversion),
                ("save", disabled_save),
                ("mapping", encoded_mapping_leak),
            ):
                with self.subTest(name=name):
                    report = self._copy_and_mutate(valid, root / name, mutate)
                    self.assertFalse(report["passed"], report)

    def test_portable_verify_does_not_read_external_source_wavs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            package = self._build_valid_package(root)
            manifest = self._load_json(package / "run_manifest.json")
            source = Path(manifest["files"][0]["path"])
            with source.open("ab") as stream:
                stream.write(b"changed-after-package")
            portable = verify(package, require_external=False)
            strict = verify(package, require_external=True)
            self.assertTrue(portable["passed"], portable)
            self.assertTrue(portable["warnings"])
            self.assertFalse(strict["passed"], strict)

    def test_verify_rejects_tampered_pairwise_design(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = self._build_valid_package(root)

            def rewrite_pairwise_html(package: Path, pairs: dict[str, dict[str, str]]) -> None:
                public = self._load_json(package / "experiment" / "manifest_public.json")
                page = _pairwise_html(pairs, int(public["events"]), float(public["interval_ms"]))
                (package / "experiment" / "pairwise_test.html").write_text(
                    page,
                    encoding="utf-8",
                )

            def wrong_hypothesis(package: Path) -> None:
                path = package / "private_do_not_open_before_scoring" / "blind_key.json"
                key = self._load_json(path)
                key["pairwise_key"]["Q01"]["hypothesis"] = "wrong_hypothesis"
                self._dump_json(path, key)

            def wrong_method_pair(package: Path) -> None:
                key_path = package / "private_do_not_open_before_scoring" / "blind_key.json"
                public_path = package / "experiment" / "pairwise_manifest_public.json"
                key = self._load_json(key_path)
                public = self._load_json(public_path)
                method_to_blind = {
                    method: blind_id for blind_id, method in key["blind_mapping"].items()
                }
                methods = ("repeat_one", "random_full")
                sides = {"A": method_to_blind[methods[0]], "B": method_to_blind[methods[1]]}
                public["pairs"]["Q01"] = sides
                key["pairwise_key"]["Q01"].update(
                    {
                        "A_method": methods[0],
                        "B_method": methods[1],
                        "A_blind_id": sides["A"],
                        "B_blind_id": sides["B"],
                    }
                )
                self._dump_json(public_path, public)
                self._dump_json(key_path, key)
                rewrite_pairwise_html(package, public["pairs"])

            def wrong_pair_id(package: Path) -> None:
                key_path = package / "private_do_not_open_before_scoring" / "blind_key.json"
                public_path = package / "experiment" / "pairwise_manifest_public.json"
                key = self._load_json(key_path)
                public = self._load_json(public_path)
                public["pairs"]["Q99"] = public["pairs"].pop("Q01")
                key["pairwise_key"]["Q99"] = key["pairwise_key"].pop("Q01")
                self._dump_json(public_path, public)
                self._dump_json(key_path, key)
                rewrite_pairwise_html(package, public["pairs"])

            for name, mutate in (
                ("hypothesis", wrong_hypothesis),
                ("method_pair", wrong_method_pair),
                ("pair_id", wrong_pair_id),
            ):
                with self.subTest(name=name):
                    report = self._copy_and_mutate(valid, root / name, mutate)
                    self.assertFalse(report["passed"], report)

    def test_verify_returns_structured_failure_for_malformed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            valid = self._build_valid_package(root)

            def missing_key(package: Path) -> None:
                path = package / "experiment" / "manifest_public.json"
                manifest = self._load_json(path)
                manifest.pop("blind_ids")
                self._dump_json(path, manifest)

            def invalid_json(package: Path) -> None:
                (package / "experiment" / "manifest_public.json").write_text(
                    "{not-json",
                    encoding="utf-8",
                )

            for name, mutate in (("missing_key", missing_key), ("invalid_json", invalid_json)):
                with self.subTest(name=name):
                    report = self._copy_and_mutate(valid, root / name, mutate)
                    self.assertIsInstance(report, dict)
                    self.assertFalse(report["passed"], report)


if __name__ == "__main__":
    unittest.main()
