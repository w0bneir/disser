"""Verify a generated natural-pool pilot without revealing its blind key."""

from __future__ import annotations

import argparse
import csv
import hashlib
from html import unescape
from html.parser import HTMLParser
import json
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np
import soundfile as sf


FORBIDDEN_PUBLIC_TOKENS = (
    "repeat_one",
    "random_full",
    "shuffle_full",
    "perceptual_full",
    "shuffle_optimized",
    "perceptual_optimized",
)

EXPECTED_BLIND_IDS = {f"P{index:02d}" for index in range(1, 7)}
EXPECTED_PAIRWISE_DESIGN = {
    "Q01": (
        "H1_scheduler_vs_shuffle",
        frozenset({"perceptual_full", "shuffle_full"}),
    ),
    "Q02": (
        "H2_small_pool_exploratory",
        frozenset({"perceptual_optimized", "shuffle_full"}),
    ),
    "Q03": (
        "H1b_scheduler_vs_random",
        frozenset({"perceptual_full", "random_full"}),
    ),
    "Q04": (
        "sanity_repeat_vs_natural_pool",
        frozenset({"repeat_one", "shuffle_full"}),
    ),
}
EXPECTED_BLIND_FIELDS = {
    "mechanical_repetition",
    "useful_variation",
    "event_consistency",
    "naturalness",
    "game_usefulness",
    "context_switch",
}
EXPECTED_PAIRWISE_FIELDS = {
    "less_repetitive",
    "more_natural",
    "more_consistent",
    "preferred",
    "confidence",
}
PUBLIC_TEXT_ARTIFACTS = {
    "blind_test.html",
    "pairwise_test.html",
    "manifest_public.json",
    "pairwise_manifest_public.json",
}
URL_ATTRIBUTES = {"src", "href", "data", "action", "formaction", "poster"}


class _AssetParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.assets: list[str] = []
        self.pair_cards: dict[str, dict[str, str]] = {}
        self.card_audio: dict[str, list[str]] = {}
        self.card_fields: dict[str, list[str]] = {}
        self.duplicate_card_ids: list[str] = []
        self._current_card: str | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        for attribute in URL_ATTRIBUTES:
            if values.get(attribute):
                self.assets.append(str(values[attribute]))
        classes = str(values.get("class") or "").split()
        if tag == "section" and "card" in classes and values.get("data-id"):
            pair_id = str(values["data-id"])
            if pair_id in self.card_audio:
                self.duplicate_card_ids.append(pair_id)
            self._current_card = pair_id
            self.card_audio.setdefault(pair_id, [])
            self.card_fields.setdefault(pair_id, [])
            if values.get("data-a") and values.get("data-b"):
                self.pair_cards[pair_id] = {
                    "A": str(values["data-a"]),
                    "B": str(values["data-b"]),
                }
        if self._current_card is not None:
            if tag in {"audio", "source"} and values.get("src"):
                self.card_audio[self._current_card].append(str(values["src"]))
            if tag == "select" and values.get("data-field"):
                self.card_fields[self._current_card].append(str(values["data-field"]))

    def handle_endtag(self, tag: str) -> None:
        if tag == "section":
            self._current_card = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_html_assets(
    path: Path,
    *,
    allowed_root: Path,
    reject_external: bool,
) -> list[str]:
    parser = _AssetParser()
    parser.feed(path.read_text(encoding="utf-8"))
    issues = []
    allowed_root = allowed_root.resolve()
    for asset in parser.assets:
        parsed = urlparse(asset)
        if asset.startswith("#"):
            continue
        if parsed.scheme or parsed.netloc:
            if reject_external:
                issues.append(f"non-local asset is forbidden: {asset}")
            continue
        if not parsed.path:
            continue
        target = (path.parent / unquote(parsed.path)).resolve()
        try:
            target.relative_to(allowed_root)
        except ValueError:
            issues.append(f"asset escapes allowed directory: {asset}")
            continue
        if not target.is_file():
            issues.append(f"missing local asset: {target}")
    return issues


def _manifest_output_path(root: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("pool output path must be a non-empty string")
    portable = value.replace("\\", "/")
    return (root / Path(portable)).resolve()


def _portable_name(value: object) -> str:
    return str(value).replace("\\", "/").rsplit("/", 1)[-1]


def _is_contained_file(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return path.is_file()


def _verify(directory: Path, *, require_external: bool = False) -> dict[str, object]:
    root = Path(directory).resolve()
    failures: list[str] = []
    warnings: list[str] = []
    required = [
        root / "run_manifest.json",
        root / "analysis" / "report.html",
        root / "analysis" / "clip_metrics.csv",
        root / "analysis" / "recommendations.json",
        root / "experiment" / "blind_test.html",
        root / "experiment" / "pairwise_test.html",
        root / "experiment" / "manifest_public.json",
        root / "experiment" / "pairwise_manifest_public.json",
        root / "private_do_not_open_before_scoring" / "blind_key.json",
    ]
    for path in required:
        if not _is_contained_file(path, root):
            failures.append(f"missing or externalized required file: {path}")
    if failures:
        return {"passed": False, "failures": failures}

    for issue in _check_html_assets(
        root / "analysis" / "report.html",
        allowed_root=root,
        reject_external=False,
    ):
        failures.append(f"invalid asset in report.html: {issue}")
    experiment_dir = root / "experiment"
    for html_path in (
        experiment_dir / "blind_test.html",
        experiment_dir / "pairwise_test.html",
    ):
        for issue in _check_html_assets(
            html_path,
            allowed_root=experiment_dir,
            reject_external=True,
        ):
            failures.append(f"invalid public asset in {html_path.name}: {issue}")

    public_paths = [
        root / "experiment" / "blind_test.html",
        root / "experiment" / "pairwise_test.html",
        root / "experiment" / "manifest_public.json",
        root / "experiment" / "pairwise_manifest_public.json",
    ]
    public_text = "\n".join(path.read_text(encoding="utf-8") for path in public_paths)
    decoded_public_text = unescape(public_text).casefold()
    for token in FORBIDDEN_PUBLIC_TOKENS:
        if token.casefold() in decoded_public_text:
            failures.append(f"blind leakage in public package: {token}")

    public_manifest = _load_json(root / "experiment" / "manifest_public.json")
    if not isinstance(public_manifest, dict):
        raise TypeError("manifest_public.json must contain an object")
    if public_manifest.get("protocol") != "natural_pool_blind_pilot_v1":
        failures.append("unexpected public blind protocol")
    blind_ids = list(public_manifest["blind_ids"])
    if len(blind_ids) != 6 or len(set(blind_ids)) != 6:
        failures.append("expected six unique blind IDs")
    if set(blind_ids) != EXPECTED_BLIND_IDS:
        failures.append("blind IDs must be exactly P01..P06")
    key = _load_json(root / "private_do_not_open_before_scoring" / "blind_key.json")
    if not isinstance(key, dict):
        raise TypeError("blind_key.json must contain an object")
    mapping = key["blind_mapping"]
    if not isinstance(mapping, dict):
        raise TypeError("blind_mapping must contain an object")
    if set(blind_ids) != set(mapping):
        failures.append("public blind IDs and private mapping differ")
    if set(mapping.values()) != set(FORBIDDEN_PUBLIC_TOKENS):
        failures.append("private mapping does not contain exactly six registered methods")
    method_labels = key.get("method_labels")
    if not isinstance(method_labels, dict) or set(method_labels) != set(FORBIDDEN_PUBLIC_TOKENS):
        failures.append("private method labels differ from registered methods")
        method_labels = {}
    for label in method_labels.values():
        if isinstance(label, str) and label.casefold() in decoded_public_text:
            failures.append("human-readable method label leaks into public package")
    expected_public_wavs = {f"{blind_id}.wav" for blind_id in blind_ids}
    actual_public_wavs = {path.name for path in (root / "experiment").glob("*.wav")}
    if actual_public_wavs != expected_public_wavs:
        failures.append(
            "public experiment WAV set differs from blind stimuli: "
            f"expected={sorted(expected_public_wavs)}, actual={sorted(actual_public_wavs)}"
        )

    expected_public_files = expected_public_wavs | PUBLIC_TEXT_ARTIFACTS
    actual_public_files = {
        path.relative_to(experiment_dir).as_posix()
        for path in experiment_dir.rglob("*")
        if path.is_file()
    }
    if actual_public_files != expected_public_files:
        failures.append(
            "public experiment inventory differs from the registered listening package: "
            f"expected={sorted(expected_public_files)}, actual={sorted(actual_public_files)}"
        )
    for path in experiment_dir.rglob("*"):
        if path.is_file() and not _is_contained_file(path, experiment_dir):
            failures.append(f"public experiment file resolves outside package: {path}")

    pairwise_html = (root / "experiment" / "pairwise_test.html").read_text(encoding="utf-8")
    blind_html = (root / "experiment" / "blind_test.html").read_text(encoding="utf-8")
    for name, page in (("pairwise_test.html", pairwise_html), ("blind_test.html", blind_html)):
        if "session_id" not in page or "Math.random()" not in page:
            failures.append(f"{name} lacks per-session identity/order randomization")
    if "reference_medoid" in public_text:
        failures.append("central reference leaks into public listening package")

    blind_parser = _AssetParser()
    blind_parser.feed(blind_html)
    if blind_parser.duplicate_card_ids:
        failures.append(
            f"duplicate blind card IDs: {sorted(set(blind_parser.duplicate_card_ids))}"
        )
    if set(blind_parser.card_audio) != set(blind_ids):
        failures.append("blind HTML card IDs differ from public blind IDs")
    for blind_id in blind_ids:
        if blind_parser.card_audio.get(blind_id) != [f"{blind_id}.wav"]:
            failures.append(f"blind HTML audio differs from manifest: {blind_id}")
        fields = blind_parser.card_fields.get(blind_id, [])
        if len(fields) != len(EXPECTED_BLIND_FIELDS) or set(fields) != EXPECTED_BLIND_FIELDS:
            failures.append(f"blind HTML question fields differ from protocol: {blind_id}")

    audio_info = []
    shapes = set()
    sample_rates = set()
    peaks = []
    rms_values = []
    for blind_id in blind_ids:
        path = root / "experiment" / f"{blind_id}.wav"
        if not _is_contained_file(path, experiment_dir):
            failures.append(f"missing blind audio: {path}")
            continue
        info = sf.info(path)
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        shapes.add(tuple(audio.shape))
        sample_rates.add(int(sample_rate))
        peak = float(np.max(np.abs(audio)))
        audio_rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
        peaks.append(peak)
        rms_values.append(audio_rms)
        if not np.isfinite(audio).all():
            failures.append(f"NaN/Inf: {path.name}")
        if peak > 10.0 ** (-1.0 / 20.0) + 1e-5:
            failures.append(f"peak above registered -1 dBFS limit: {path.name}")
        if info.subtype != "PCM_16":
            failures.append(f"browser stimulus is not PCM_16: {path.name}")
        audio_info.append(
            {
                "blind_id": blind_id,
                "frames": int(info.frames),
                "channels": int(info.channels),
                "sample_rate": int(info.samplerate),
                "peak": peak,
                "rms": audio_rms,
            }
        )
        raw_audio = path.read_bytes()
        leak_tokens = list(FORBIDDEN_PUBLIC_TOKENS) + [
            label for label in method_labels.values() if isinstance(label, str)
        ]
        for token in leak_tokens:
            encodings = (token.encode("utf-8"), token.encode("utf-16-le"), token.encode("utf-16-be"))
            if any(encoded in raw_audio for encoded in encodings):
                failures.append(f"method metadata leaks into blind WAV: {path.name}")
                break
    if len(shapes) != 1:
        failures.append(f"blind audio shapes differ: {sorted(shapes)}")
    if len(sample_rates) != 1:
        failures.append(f"blind sample rates differ: {sorted(sample_rates)}")
    if rms_values and max(rms_values) - min(rms_values) > 1e-6:
        failures.append("blind sequence RMS values are not matched")

    run_manifest = _load_json(root / "run_manifest.json")
    if not isinstance(run_manifest, dict):
        raise TypeError("run_manifest.json must contain an object")
    if run_manifest.get("protocol") != "natural_pool_optimizer_v1":
        failures.append("unexpected run-manifest protocol")
    settings = run_manifest.get("settings")
    experiment = run_manifest.get("experiment")
    if not isinstance(settings, dict) or not isinstance(experiment, dict):
        raise TypeError("run manifest lacks settings or experiment object")
    if bool(settings.get("analysis_only")):
        failures.append("listening package is marked analysis-only")
    expected_public_hashes = experiment.get("public_artifact_hashes")
    if not isinstance(expected_public_hashes, dict) or set(expected_public_hashes) != PUBLIC_TEXT_ARTIFACTS:
        failures.append("public artifact hash inventory is missing or incomplete")
        expected_public_hashes = {}
    for name in PUBLIC_TEXT_ARTIFACTS:
        expected = expected_public_hashes.get(name)
        path = experiment_dir / name
        if not isinstance(expected, str) or _sha256(path) != expected:
            failures.append(f"public artifact SHA-256 mismatch: {name}")
    if len(shapes) == 1 and len(sample_rates) == 1:
        actual_frames, actual_channels = next(iter(shapes))
        actual_sample_rate = next(iter(sample_rates))
        if public_manifest.get("sample_rate") != actual_sample_rate:
            failures.append("public sample_rate differs from blind WAV files")
        if public_manifest.get("channels") != actual_channels:
            failures.append("public channels differ from blind WAV files")
        if public_manifest.get("events") != settings.get("events"):
            failures.append("public events differ from run settings")
        if not np.isclose(
            float(public_manifest.get("interval_ms")),
            float(settings.get("interval_ms")),
            rtol=0.0,
            atol=1e-9,
        ):
            failures.append("public interval differs from run settings")
        expected_frames = (
            int(round(actual_sample_rate * 0.250))
            + int(round(actual_sample_rate * float(settings["interval_ms"]) / 1000.0))
            * (int(settings["events"]) - 1)
            + int(round(actual_sample_rate * float(settings["clip_seconds"])))
        )
        if actual_frames != expected_frames:
            failures.append(
                f"blind WAV frame count differs from run design: {actual_frames} != {expected_frames}"
            )
    implementation_paths = {
        "runner": Path(__file__).with_name("run_natural_pool_pilot.py").resolve(),
        "optimizer": Path(__file__).with_name("sfx_pool_optimizer.py").resolve(),
        "verifier": Path(__file__).resolve(),
        "ratings_analyzer": Path(__file__).with_name("analyze_natural_pool_ratings.py").resolve(),
    }
    expected_implementation_hashes = run_manifest.get("implementation_sha256", {})
    for name, path in implementation_paths.items():
        expected = expected_implementation_hashes.get(name)
        if not isinstance(expected, str):
            failures.append(f"missing implementation SHA-256: {name}")
        elif require_external:
            if not path.is_file():
                failures.append(f"implementation file unavailable: {path}")
            elif _sha256(path) != expected:
                failures.append(f"implementation SHA-256 mismatch: {name}")
    if not require_external:
        warnings.append("portable mode: external implementation files were not read")
    registered_sources = run_manifest.get("files")
    if not isinstance(registered_sources, list) or not registered_sources:
        failures.append("run manifest must register a non-empty source inventory")
        registered_sources = []
    source_paths: list[str] = []
    for source in registered_sources:
        if not isinstance(source, dict):
            failures.append("invalid source inventory record")
            continue
        if not isinstance(source.get("path"), str) or not isinstance(source.get("sha256"), str):
            failures.append("source inventory record lacks path or SHA-256")
            continue
        source_paths.append(source["path"])
        path = Path(source["path"])
        if require_external:
            if not path.is_file():
                failures.append(f"registered source unavailable: {path}")
            elif _sha256(path) != source["sha256"]:
                failures.append(f"source SHA-256 mismatch: {path.name}")
    if not require_external:
        warnings.append("portable mode: external source WAVs were not read")
    if len(source_paths) != len(set(source_paths)):
        failures.append("source inventory contains duplicate paths")
    with (root / "analysis" / "clip_metrics.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        metric_rows = list(csv.DictReader(stream))
    metric_paths = [row.get("file", "") for row in metric_rows]
    if len(metric_paths) != len(source_paths) or set(metric_paths) != set(source_paths):
        failures.append("source inventory differs from analysis/clip_metrics.csv")
    selected_group = str(experiment.get("group"))
    selected_sources = [
        source
        for source in registered_sources
        if isinstance(source, dict) and str(source.get("group")) == selected_group
    ]
    full_pool_count = experiment.get("full_pool_count")
    optimized_pool_count = experiment.get("optimized_pool_count")
    if not isinstance(full_pool_count, int) or len(selected_sources) != full_pool_count:
        failures.append("registered source count for the selected group differs from full_pool_count")
    if (
        not isinstance(optimized_pool_count, int)
        or not isinstance(full_pool_count, int)
        or not 2 <= optimized_pool_count < full_pool_count
    ):
        failures.append("optimized pool must contain at least two and fewer takes than the full pool")
    else:
        if settings.get("pool_size") != optimized_pool_count:
            failures.append("run setting pool_size differs from optimized_pool_count")
        expected_reduction = 1.0 - optimized_pool_count / full_pool_count
        if not np.isclose(
            float(experiment.get("asset_reduction_fraction")),
            expected_reduction,
            rtol=0.0,
            atol=1e-12,
        ):
            failures.append("asset_reduction_fraction is inconsistent with pool counts")
    expected_hashes = run_manifest["experiment"]["blind_stimulus_hashes"]
    if set(expected_hashes) != set(blind_ids):
        failures.append("blind stimulus hash keys differ from public blind IDs")
    for blind_id, expected in expected_hashes.items():
        path = root / "experiment" / f"{blind_id}.wav"
        if not path.is_file():
            failures.append(f"missing blind stimulus registered by hash: {path.name}")
        elif _sha256(path) != expected:
            failures.append(f"SHA-256 mismatch: {path.name}")

    pairwise = _load_json(root / "experiment" / "pairwise_manifest_public.json")
    if not isinstance(pairwise, dict):
        raise TypeError("pairwise_manifest_public.json must contain an object")
    if pairwise.get("protocol") != "natural_pool_pairwise_v1":
        failures.append("unexpected pairwise protocol")
    if len(pairwise.get("pairs", {})) != 4:
        failures.append("expected four registered pairwise comparisons")
    private_pairwise = key.get("pairwise_key", {})
    if set(pairwise.get("pairs", {})) != set(private_pairwise):
        failures.append("public and private pairwise IDs differ")
    pair_parser = _AssetParser()
    pair_parser.feed(pairwise_html)
    if pair_parser.duplicate_card_ids:
        failures.append(
            f"duplicate pairwise card IDs: {sorted(set(pair_parser.duplicate_card_ids))}"
        )
    if pair_parser.pair_cards != pairwise.get("pairs", {}):
        failures.append("pairwise HTML cards differ from public pairwise manifest")
    if set(pairwise.get("pairs", {})) != set(EXPECTED_PAIRWISE_DESIGN):
        failures.append("pairwise IDs differ from the registered four-comparison design")
    if require_external:
        from run_natural_pool_pilot import _blind_html, _pairwise_html

        expected_blind_html = _blind_html(
            blind_ids,
            int(public_manifest["events"]),
            float(public_manifest["interval_ms"]),
        )
        expected_pairwise_html = _pairwise_html(
            pairwise["pairs"],
            int(public_manifest["events"]),
            float(public_manifest["interval_ms"]),
        )
        if blind_html != expected_blind_html:
            failures.append("blind HTML differs from the registered implementation template")
        if pairwise_html != expected_pairwise_html:
            failures.append("pairwise HTML differs from the registered implementation template")
    for pair_id, sides in pairwise["pairs"].items():
        if (
            set(sides) != {"A", "B"}
            or not set(sides.values()).issubset(set(blind_ids))
            or sides["A"] == sides["B"]
        ):
            failures.append(f"invalid pairwise mapping: {pair_id}")
            continue
        private = private_pairwise.get(pair_id)
        if not isinstance(private, dict):
            continue
        if private.get("A_blind_id") != sides["A"] or private.get("B_blind_id") != sides["B"]:
            failures.append(f"private key orientation differs from public pair: {pair_id}")
        expected_design = EXPECTED_PAIRWISE_DESIGN.get(pair_id)
        if expected_design is not None:
            expected_hypothesis, expected_methods = expected_design
            actual_methods = frozenset({private.get("A_method"), private.get("B_method")})
            if private.get("hypothesis") != expected_hypothesis or actual_methods != expected_methods:
                failures.append(f"private pair differs from registered hypothesis design: {pair_id}")
        expected_audio = [f"{sides['A']}.wav", f"{sides['B']}.wav"]
        if pair_parser.card_audio.get(pair_id) != expected_audio:
            failures.append(f"pairwise HTML audio differs from public pair: {pair_id}")
        fields = pair_parser.card_fields.get(pair_id, [])
        if len(fields) != len(EXPECTED_PAIRWISE_FIELDS) or set(fields) != EXPECTED_PAIRWISE_FIELDS:
            failures.append(f"pairwise HTML question fields differ from protocol: {pair_id}")
        for side in ("A", "B"):
            blind_id = private.get(f"{side}_blind_id")
            method = private.get(f"{side}_method")
            if blind_id not in mapping or mapping.get(blind_id) != method:
                failures.append(f"private method/blind mapping is inconsistent: {pair_id}.{side}")

    pool_manifests = list((root / "optimized_pool").glob("group_*/pool_manifest.json"))
    if len(pool_manifests) != 1:
        failures.append("expected exactly one optimized pool manifest")
    elif not _is_contained_file(pool_manifests[0], root):
        failures.append("optimized pool manifest resolves outside package")
    else:
        pool_manifest = _load_json(pool_manifests[0])
        if not isinstance(pool_manifest, dict):
            raise TypeError("pool_manifest.json must contain an object")
        if str(pool_manifest.get("group")) != selected_group:
            failures.append("optimized pool group differs from run manifest")
        pool_files = pool_manifest.get("files")
        if not isinstance(pool_files, list) or len(pool_files) != optimized_pool_count:
            failures.append("optimized pool inventory differs from optimized_pool_count")
            pool_files = []
        outputs: list[Path] = []
        pool_sources: list[str] = []
        selected_source_names = {_portable_name(source.get("path")) for source in selected_sources}
        for item in pool_files:
            if not isinstance(item, dict):
                failures.append("invalid optimized pool record")
                continue
            output = _manifest_output_path(root, item.get("output"))
            outputs.append(output)
            pool_sources.append(str(item.get("source")))
            try:
                output.relative_to(root)
            except ValueError:
                failures.append(f"pool output escapes result directory: {output}")
                continue
            if not output.is_file():
                failures.append(f"missing optimized take: {output}")
            else:
                if _sha256(output) != item["sha256"]:
                    failures.append(f"optimized take SHA-256 mismatch: {output.name}")
                if sf.info(output).subtype != "PCM_24":
                    failures.append(f"optimized take is not PCM_24: {output.name}")
                info = sf.info(output)
                if len(sample_rates) == 1 and info.samplerate != next(iter(sample_rates)):
                    failures.append(f"optimized take sample rate differs: {output.name}")
                if len(shapes) == 1 and info.channels != next(iter(shapes))[1]:
                    failures.append(f"optimized take channel count differs: {output.name}")
        if len(outputs) != len(set(outputs)):
            failures.append("optimized pool inventory contains duplicate outputs")
        if len(pool_sources) != len(set(pool_sources)):
            failures.append("optimized pool inventory contains duplicate sources")
        if not set(pool_sources).issubset(selected_source_names):
            failures.append("optimized pool references sources outside the selected group")
        actual_pool_wavs = {path.resolve() for path in (root / "optimized_pool").rglob("*.wav")}
        if actual_pool_wavs != set(outputs):
            failures.append("optimized pool WAV inventory differs from pool manifest")

    return {
        "passed": not failures,
        "failures": failures,
        "warnings": warnings,
        "blind_files": len(audio_info),
        "shape": list(next(iter(shapes))) if len(shapes) == 1 else None,
        "sample_rate": next(iter(sample_rates)) if len(sample_rates) == 1 else None,
        "maximum_peak": max(peaks) if peaks else None,
        "maximum_rms_delta": max(rms_values) - min(rms_values) if rms_values else None,
        "private_key_checked_without_revealing_mapping": True,
    }


def verify(directory: Path, *, require_external: bool = False) -> dict[str, object]:
    """Return a structured failure for malformed or unreadable packages."""

    try:
        return _verify(directory, require_external=require_external)
    except Exception as error:
        return {
            "passed": False,
            "failures": [f"verification could not be completed: {type(error).__name__}: {error}"],
            "warnings": [],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument(
        "--require-external",
        action="store_true",
        help="Fail when original source WAVs or current implementation files are unavailable.",
    )
    args = parser.parse_args()
    report = verify(args.results_dir, require_external=args.require_external)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
