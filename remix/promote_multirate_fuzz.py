"""Publish/load a separately validated offline finite-history DFZ bundle.

No existing bundle is overwritten. The model card is published last, with an
exclusive hard link: partially copied files are never an admitted runtime.
This does not register the experimental schema in any legacy/UI renderer.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

import torch

from .evaluate_multirate_fuzz import ARCHITECTURES, graph_runtime, load_candidate, model_from_payload
from .multirate_admission import source_hashes_unchanged, validate_evidence, validate_structure
from .precheck_multirate_fuzz import sha256
from .promote_dfz_forward import require, valid_digest, verify_source_files
from .widen_multirate_fuzz import WIDE_GEOMETRY


RUNTIME_SOURCES = ("promote_multirate_fuzz.py", "multirate_admission.py", "evaluate_multirate_fuzz.py",
                   "multirate_fuzz.py", "centered_multirate_fuzz.py", "audit_multirate_fuzz.py", "precheck_multirate_fuzz.py",
                   "promote_dfz_forward.py", "evaluate_asrnn_effect.py", "widen_multirate_fuzz.py")
EVIDENCE_NAMES = ("calibration", "pytorch", "onnx", "runtime")


def _read_card(directory):
    card = json.loads((directory / "model-card.json").read_text())
    require(card.get("schema") == 1 and card.get("accepted") is True
            and card.get("status") == "accepted-internal-noncommercial-development-pilot"
            and card.get("architecture") in ARCHITECTURES
            and card.get("geometry") == WIDE_GEOMETRY and card.get("device") == "dfz" and card.get("sample_rate") == 48000
            and card.get("control_names") == ["blend", "filter"], "admitted finite-history model card required")
    require(all(card.get(key) is False for key in ("ui_integration_allowed", "commercial_release_allowed",
                                                  "physical_audio_devices_used", "new_locked_final_audio_opened",
                                                  "unseen_control_interpolation_evaluated", "source_audio_modified")), "model-card scope differs")
    require(card.get("dataset_license") == "CC-BY-NC-4.0" and card.get("development_control_grid") == [[0., .5, 1.], [0., .5, 1.]],
            "model-card data/license scope differs")
    runtime_sources = card.get("runtime_source_sha256", {})
    require(set(runtime_sources) == set(RUNTIME_SOURCES), "runtime source closure incomplete")
    for name, digest in runtime_sources.items():
        require(valid_digest(digest) and sha256(Path(__file__).parent / name) == digest, "runtime source code changed")
    files = card.get("files", {})
    expected = {"model.pt", "model.onnx", "training-evidence.json", *[name + "-evidence.json" for name in EVIDENCE_NAMES]}
    require(set(files) == expected, "runtime artifact closure incomplete")
    for name, digest in files.items():
        require(valid_digest(digest) and sha256(directory / name) == digest, "runtime model/evidence bytes changed")
    reports = {name: json.loads((directory / f"{name}-evidence.json").read_text()) for name in EVIDENCE_NAMES}
    require(card["architecture"] == reports["calibration"].get("architecture"), "card/evidence architecture differs")
    require(card.get("quality_contract") == reports["onnx"].get("quality_policy"), "card quality contract differs from evidence")
    checks = validate_evidence(reports["calibration"], reports["pytorch"], reports["onnx"], reports["runtime"],
                               files["model.pt"], files["model.onnx"], files["calibration-evidence.json"])
    require(card.get("checks") == checks, "card does not carry the complete unchanged admission checks")
    training = json.loads((directory / "training-evidence.json").read_text())
    require(training.get("checkpoint_sha256") == files["model.pt"] and training.get("source_and_audio_reverified") is True
            and files["training-evidence.json"] == reports["calibration"]["training_report_sha256"],
            "packaged training evidence differs")
    return card


class AdmittedMultirate:
    """Offline/streaming CPU renderer; no audio-device or application UI API."""
    sample_rate, control_names = 48000, ("blend", "filter")

    def __init__(self, directory):
        self.directory = Path(directory)
        self.card = _read_card(self.directory)
        self.card_hash = sha256(self.directory / "model-card.json")
        payload = torch.load(self.directory / "model.pt", map_location="cpu", weights_only=True)
        require(payload.get("architecture") == self.card["architecture"], "card/weights architecture differs")
        model = model_from_payload(payload)
        structure = validate_structure(model, payload)
        require(structure == self.card.get("structure"), "actual finite-history model structure/bound differs")
        self.backend = graph_runtime(self.directory / "model.onnx", model, self.card["files"]["model.pt"])
        self.assert_artifacts_unchanged()

    def assert_artifacts_unchanged(self):
        require(sha256(self.directory / "model-card.json") == self.card_hash, "admission card changed")
        for name, digest in self.card["files"].items():
            require(sha256(self.directory / name) == digest, "admitted runtime artifact changed")
        for name, digest in self.card["runtime_source_sha256"].items():
            require(sha256(Path(__file__).parent / name) == digest, "admitted runtime code changed")

    @torch.inference_mode()
    def __call__(self, dry, controls, state=None):
        return self.backend(dry, controls, state)


def promote(args):
    require(not args.output.exists(), "promotion destination already exists; no overwrite allowed")
    paths = {"calibration": args.calibration, "pytorch": args.pytorch, "onnx": args.development, "runtime": args.runtime}
    reports = {name: json.loads(path.read_text()) for name, path in paths.items()}
    hashes = {name: sha256(path) for name, path in paths.items()}
    checkpoint_hash, graph_hash = sha256(args.checkpoint), sha256(args.onnx)
    checks = validate_evidence(reports["calibration"], reports["pytorch"], reports["onnx"], reports["runtime"],
                               checkpoint_hash, graph_hash, hashes["calibration"])
    for report in reports.values():
        source_hashes_unchanged(report)
    verify_source_files(reports["calibration"], reports["onnx"], args.corpus)
    model, payload, training = load_candidate(args.checkpoint)
    require(payload["architecture"] == reports["calibration"]["architecture"], "weight/evidence architecture differs")
    structure = validate_structure(model, payload)
    graph_runtime(args.onnx, model, checkpoint_hash)
    training_path = args.checkpoint.with_name("metrics.json")
    training_hash = sha256(training_path)
    require(training_hash == reports["calibration"]["training_report_sha256"], "training report changed before packaging")
    runtime_sources = {name: sha256(Path(__file__).parent / name) for name in RUNTIME_SOURCES}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".multirate-admission-", dir=args.output.parent) as temporary:
        staging = Path(temporary)
        shutil.copy2(args.checkpoint, staging / "model.pt")
        shutil.copy2(args.onnx, staging / "model.onnx")
        shutil.copy2(training_path, staging / "training-evidence.json")
        files = {"model.pt": checkpoint_hash, "model.onnx": graph_hash, "training-evidence.json": training_hash}
        for name, path in paths.items():
            filename = f"{name}-evidence.json"
            shutil.copy2(path, staging / filename)
            files[filename] = hashes[name]
        require(all(sha256(staging / name) == digest for name, digest in files.items()), "source artifacts changed while copying")
        require(all(sha256(path) == hashes[name] for name, path in paths.items()) and sha256(args.checkpoint) == checkpoint_hash
                and sha256(args.onnx) == graph_hash and sha256(training_path) == training_hash, "source evidence changed during packaging")
        verify_source_files(reports["calibration"], reports["onnx"], args.corpus)
        for report in reports.values():
            source_hashes_unchanged(report)
        card = {"schema": 1, "accepted": True, "status": "accepted-internal-noncommercial-development-pilot",
                "architecture": payload["architecture"], "geometry": WIDE_GEOMETRY, "device": "dfz", "sample_rate": 48000,
                "control_names": ["blend", "filter"], "files": files, "checks": checks, "structure": structure,
                "runtime_source_sha256": runtime_sources, "parameters": sum(value.numel() for value in model.parameters()),
                "calibration": reports["calibration"]["calibration"], "development": reports["onnx"]["official_eval"],
                "dataset_license": "CC-BY-NC-4.0", "commercial_release_allowed": False, "ui_integration_allowed": False,
                "physical_audio_devices_used": False, "source_audio_modified": False, "new_locked_final_audio_opened": False,
                "unseen_control_interpolation_evaluated": False, "development_control_grid": [[0., .5, 1.], [0., .5, 1.]],
                "quality_contract": reports["onnx"]["quality_policy"],
                "limitations": ["official eval was development evidence, not independent locked-final",
                                "only nine static Blend/Filter combinations have paired fidelity evidence",
                                "runtime dynamic-control checks do not prove physical dynamic-knob fidelity",
                                "unpaired GuitarSet/hardware-driver latency/product validation not claimed",
                                "finite-history boundedness is separate from measured waveform fidelity"]}
        (staging / "model-card.json").write_text(json.dumps(card, indent=2, allow_nan=False) + "\n")
        _read_card(staging)
        # copytree(exist_ok=False) never replaces a directory supplied by someone
        # else. The exclusive card link atomically activates our verified files.
        shutil.copytree(staging, args.output, ignore=shutil.ignore_patterns("model-card.json"), dirs_exist_ok=False)
        require(all(sha256(args.output / name) == digest for name, digest in files.items()), "copied bundle bytes changed")
        os.link(staging / "model-card.json", args.output / "model-card.json")
    runtime = AdmittedMultirate(args.output)
    runtime.assert_artifacts_unchanged()
    return card


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "onnx", "calibration", "pytorch", "development", "runtime", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("data/corpus/asrnn-physical-effects"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    card = promote(args)
    print(json.dumps({"accepted": card["accepted"], "status": card["status"], "checks": card["checks"]}), flush=True)


if __name__ == "__main__":
    main()
