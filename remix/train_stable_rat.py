#!/usr/bin/env python3
"""Train a paper-compatible asymptotically stable RAT renderer from Dry/Wet pairs.

This is an independent implementation of the constraints and training geometry
described in Kallinen, Juvela, and Sherson, "Deep Regularized RNNs for Virtual
Analog Modeling" (arXiv:2509.15622v2). It does not contain or import the GPLv3
reference training implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import scipy.signal
import torch
from torch import nn

from .asrnn_data import LICENSE, RECORD_URL, audit_asrnn, rat_files, read_rat_pair
from .evaluate_asrnn_stable import evaluate as official_evaluate
from .train_asrnn_rat_adapter import _bounded, _partition


RATE = 48_000
NORM = 0.995
GATE = 5.3
PAPER = "https://arxiv.org/abs/2509.15622v2"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _detach(state):
    return tuple((hidden.detach(), cell.detach()) for hidden, cell in state)


def _controls(values: torch.Tensor, frames: int) -> torch.Tensor:
    physical = values.clone()
    if physical.ndim == 2:
        physical[:, 1] = 1.0 - physical[:, 1]
        physical = physical.unsqueeze(1).expand(-1, frames, -1)
    elif physical.ndim == 3 and physical.shape[1:] == (frames, 3):
        physical[:, :, 1] = 1.0 - physical[:, :, 1]
    else:
        raise ValueError("stable RAT controls must be [batch,3] or [batch,time,3]")
    return physical.mul(2.0).sub(1.0)


class StableRatTrain(nn.Module):
    """Deep LSTM whose stored weights are projected into the stable set."""

    def __init__(self, hidden: int = 8, layers: int = 4, input_coef: float = 1.0):
        super().__init__()
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.input_coef = float(input_coef)
        blocks = [nn.LSTM(4, hidden, batch_first=True)]
        blocks.extend(nn.LSTM(hidden + 3, hidden, batch_first=True) for _ in range(1, layers))
        self.rnn_layers = nn.ModuleList(blocks)
        self.output_layer = nn.Linear(hidden, 1, bias=False)
        self.project_()

    @torch.no_grad()
    def project_(self) -> None:
        """Project parameters to Eqs. 7, 9, and 10 of the paper."""

        width = self.hidden
        for layer in self.rnn_layers:
            weight_in = layer.weight_ih_l0
            weight_hidden = layer.weight_hh_l0
            bias = layer.bias_ih_l0
            condition = slice(weight_in.shape[1] - 3, weight_in.shape[1])
            input_gate = slice(0, width)
            forget_gate = slice(width, 2 * width)
            cell_gate = slice(2 * width, 3 * width)

            condition_forget = 0.5 * (
                weight_in[forget_gate, condition] - weight_in[input_gate, condition]
            )
            hidden_forget = 0.5 * (
                weight_hidden[forget_gate] - weight_hidden[input_gate]
            )
            bias_forget = 0.5 * (bias[forget_gate] - bias[input_gate])
            bias_forget.clamp_(max=GATE)

            capacity = (GATE - bias_forget).clamp_min(0.0)
            magnitude = condition_forget.abs().sum(1) + hidden_forget.abs().sum(1)
            scale = torch.minimum(
                torch.ones_like(magnitude),
                capacity / magnitude.clamp_min(1.0e-12),
            )
            condition_forget.mul_(scale[:, None])
            hidden_forget.mul_(scale[:, None])

            weight_in[forget_gate, condition] = condition_forget
            weight_in[input_gate, condition] = -condition_forget
            weight_hidden[forget_gate] = hidden_forget
            weight_hidden[input_gate] = -hidden_forget
            bias[forget_gate] = bias_forget
            bias[input_gate] = -bias_forget

            weight_in[cell_gate, condition].zero_()
            bias[cell_gate].zero_()
            layer.bias_hh_l0.zero_()
            candidate = weight_hidden[cell_gate]
            norm = torch.linalg.matrix_norm(candidate, ord=float("inf"))
            candidate.mul_(
                torch.minimum(
                    torch.ones_like(norm),
                    torch.as_tensor(NORM, device=norm.device) / norm.clamp_min(1.0e-12),
                )
            )

    def forward(self, dry: torch.Tensor, controls: torch.Tensor, state=None):
        condition = _controls(controls, dry.shape[1])
        value = dry.unsqueeze(-1) * self.input_coef
        previous = (None,) * self.layers if state is None else state
        next_state = []
        for block, hidden in zip(self.rnn_layers, previous):
            value, hidden = block(torch.cat((value, condition), 2), hidden)
            next_state.append(hidden)
        return self.output_layer(value).squeeze(-1), tuple(next_state)

    @torch.no_grad()
    def audit(self) -> dict:
        rows = []
        width = self.hidden
        for index, layer in enumerate(self.rnn_layers):
            candidate = layer.weight_hh_l0[2 * width : 3 * width]
            condition = slice(layer.weight_ih_l0.shape[1] - 3, layer.weight_ih_l0.shape[1])
            input_condition = layer.weight_ih_l0[0:width, condition]
            forget_condition = layer.weight_ih_l0[width : 2 * width, condition]
            input_hidden = layer.weight_hh_l0[0:width]
            forget_hidden = layer.weight_hh_l0[width : 2 * width]
            rows.append(
                {
                    "layer": index,
                    "candidate_infinity_norm": float(
                        torch.linalg.matrix_norm(candidate, ord=float("inf"))
                    ),
                    "cell_condition_max": float(
                        layer.weight_ih_l0[2 * width : 3 * width, condition].abs().max()
                    ),
                    "cell_bias_max": float(layer.bias_ih_l0[2 * width : 3 * width].abs().max()),
                    "hidden_bias_max": float(layer.bias_hh_l0.abs().max()),
                    "condition_mirror_error": float(
                        (input_condition + forget_condition).abs().max()
                    ),
                    "hidden_mirror_error": float((input_hidden + forget_hidden).abs().max()),
                    "bias_mirror_error": float(
                        (layer.bias_ih_l0[0:width] + layer.bias_ih_l0[width : 2 * width])
                        .abs()
                        .max()
                    ),
                }
            )
        passed = all(
            row["candidate_infinity_norm"] <= NORM + 1.0e-6
            and row["cell_condition_max"] == 0.0
            and row["cell_bias_max"] == 0.0
            and row["hidden_bias_max"] == 0.0
            and row["condition_mirror_error"] == 0.0
            and row["hidden_mirror_error"] == 0.0
            and row["bias_mirror_error"] == 0.0
            for row in rows
        )
        return {"threshold": NORM, "layers": rows, "passed": passed}


def _erb(low: float = 100.0, high: float = 10_000.0) -> np.ndarray:
    ear_q, minimum = 9.26449, 24.7
    values = []
    value = low
    while value < high:
        values.append(value)
        position = ear_q * np.log1p(value / (minimum * ear_q))
        value = minimum * ear_q * (np.exp((position + 1.0) / ear_q) - 1.0)
    return np.asarray(values)


class GammatoneLoss(nn.Module):
    def __init__(self):
        super().__init__()
        filters = [
            scipy.signal.gammatone(float(frequency), "fir", fs=RATE, order=4)[0]
            for frequency in _erb()
        ]
        self.register_buffer("filters", torch.tensor(np.stack(filters), dtype=torch.float32))
        self._spectra = {}

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = target - prediction
        frames = error.shape[1]
        fft_length = 1 << (frames + self.filters.shape[1] - 2).bit_length()
        key = (frames, error.device.type, error.device.index, error.dtype)
        spectrum = self._spectra.get(key)
        if spectrum is None:
            spectrum = torch.fft.rfft(self.filters.to(error.dtype), n=fft_length)
            self._spectra[key] = spectrum
        filtered = torch.fft.irfft(
            torch.fft.rfft(error, n=fft_length).unsqueeze(1) * spectrum.unsqueeze(0),
            n=fft_length,
        )[:, :, :frames]
        residual = error - filtered.sum(1)
        return (residual.abs() + filtered.abs().sum(1)).mean()


def _training_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    auditory: GammatoneLoss,
    esr_weight: float,
    peak_weight: float,
    quiet_weight: float,
) -> torch.Tensor:
    loss = auditory(prediction, target)
    if esr_weight:
        error_energy = (prediction - target).square().mean(1)
        target_energy = target.square().mean(1).clamp_min(1.0e-6)
        loss = loss + esr_weight * (error_energy / target_energy).mean()
    if peak_weight:
        target_peak = target.abs().amax(1)
        prediction_peak = prediction.abs().amax(1)
        peak_error = (prediction_peak - target_peak).abs().mean()
        loss = loss + peak_weight * peak_error
    else:
        target_peak = target.abs().amax(1)
        prediction_peak = prediction.abs().amax(1)
    if quiet_weight:
        quiet = target_peak < 1.0e-3
        if bool(quiet.any()):
            loss = loss + quiet_weight * prediction_peak[quiet].mean()
    return loss


def _fractional_delay(audio: torch.Tensor, offset: float) -> torch.Tensor:
    count = 1023
    index = torch.arange(-511, 512, dtype=audio.dtype)
    kernel = torch.sinc(index - offset) * torch.kaiser_window(
        count, periodic=False, beta=12.0, dtype=audio.dtype
    )
    kernel = kernel.to(audio.device)
    length = audio.shape[1] + count - 1
    fft_length = 1 << math.ceil(math.log2(length))
    spectrum = torch.fft.rfft(audio, n=fft_length)
    response = torch.fft.rfft(kernel, n=fft_length)
    return torch.fft.irfft(spectrum * response, n=fft_length)[:, :length]


def _load(paths: list[Path]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dry, wet, controls = [], [], []
    for path in paths:
        source, target, setting = read_rat_pair(path)
        dry.append(torch.from_numpy(source))
        wet.append(torch.from_numpy(target))
        controls.append(torch.from_numpy(setting))
    return torch.stack(dry), torch.stack(wet), torch.stack(controls)


def _calibration_score(report: dict) -> float:
    """Rank checkpoints across every lower-is-better admission dimension."""

    return (
        report["global_esr"] / 0.05
        + report["mean_per_file_esr"] / 0.10
        + report["median_per_file_esr"] / 0.05
        + report["p95_per_file_esr"] / 0.25
        + report["absolute_peak_error_p95"] / 0.02
        + report["quiet_prediction_peak_maximum"] / 0.001
        + abs(report["peak_ratio_median"] - 1.0) / 0.25
        + abs(report["peak_ratio_p95"] - 1.0) / 0.35
    )


@torch.inference_mode()
def _calibrate(
    model: StableRatTrain,
    tensors,
    target_device: torch.device,
    batch: int = 32,
) -> dict:
    dry, wet, controls = tensors
    errors, ratios, peak_errors, quiet = [], [], [], []
    total_error = total_energy = 0.0
    model.eval()
    for start in range(0, len(dry), batch):
        source = dry[start : start + batch].to(target_device)
        target = wet[start : start + batch, 1_024:].to(target_device)
        setting = controls[start : start + batch].to(target_device)
        state = None
        chunks = []
        for offset in range(0, source.shape[1], 2_048):
            value, state = model(source[:, offset : offset + 2_048], setting, state)
            chunks.append(value)
        prediction = torch.cat(chunks, 1)[:, 1_024:]
        squared = (prediction - target).square()
        energy = target.square()
        total_error += float(squared.sum())
        total_energy += float(energy.sum())
        errors.extend(
            (squared.mean(1) / energy.mean(1).clamp_min(1.0e-8)).cpu().tolist()
        )
        target_peak = target.abs().amax(1)
        predicted_peak = prediction.abs().amax(1)
        peak_errors.extend((predicted_peak - target_peak).abs().cpu().tolist())
        audible = target_peak >= 1.0e-3
        ratios.extend((predicted_peak[audible] / target_peak[audible]).cpu().tolist())
        quiet.extend(predicted_peak[~audible].cpu().tolist())
    report = {
        "global_esr": total_error / max(total_energy, 1.0e-12),
        "mean_per_file_esr": float(np.mean(errors)),
        "median_per_file_esr": float(np.median(errors)),
        "p95_per_file_esr": float(np.quantile(errors, 0.95)),
        "peak_ratio_median": float(np.median(ratios)),
        "peak_ratio_p95": float(np.quantile(ratios, 0.95)),
        "absolute_peak_error_p95": float(np.quantile(peak_errors, 0.95)),
        "quiet_prediction_peak_maximum": max(quiet, default=0.0),
    }
    report["score"] = _calibration_score(report)
    return report


def _runtime_payload(model: StableRatTrain) -> dict:
    return {
        "schema": 1,
        "sample_rate": RATE,
        "architecture": "stable-conditioned-lstm",
        "hidden_size": model.hidden,
        "layers": model.layers,
        "input_coef": model.input_coef,
        "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()},
        "device_domain": "ASRNN ProCo RAT",
        "dataset_record": RECORD_URL,
        "dataset_license": LICENSE,
        "training_paper": PAPER,
        "training_origin": "independent Muspector implementation",
    }


def train(
    corpus: Path,
    output: Path,
    *,
    epochs: int,
    batch_size: int,
    frames: int,
    hidden: int,
    layers: int,
    seed: int,
    maximum_train_files: int | None,
    eval_every: int,
    augmentation: bool,
    final: bool,
    device_name: str,
    compile_model: bool,
    init: Path | None,
    learning_rate: float,
    esr_weight: float,
    peak_weight: float,
    quiet_weight: float,
) -> dict:
    complete = output / "complete.json"
    resumable = output / "latest.pt"
    if complete.exists():
        if not resumable.exists():
            raise ValueError(f"stable RAT round is already complete: {output}")
        prior = torch.load(resumable, map_location="cpu", weights_only=True)
        if int(prior["epoch"]) >= epochs:
            raise ValueError(f"stable RAT round is already complete: {output}")
    audit = audit_asrnn(corpus, scan_audio=True)
    fit_paths, calibrate_paths = _partition(rat_files(corpus, "train"))
    fit_paths = _bounded(fit_paths, maximum_train_files)
    fit = _load(fit_paths)
    calibrate = _load(calibrate_paths)
    derived_input_coef = 1.0 / float(fit[0].double().std(unbiased=False))
    target_device = torch.device(
        "mps"
        if device_name == "auto" and torch.backends.mps.is_available()
        else "cpu"
        if device_name == "auto"
        else device_name
    )
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = StableRatTrain(hidden, layers, derived_input_coef).to(target_device)
    initial_hash = None
    if init is not None:
        initial = torch.load(init, map_location=target_device, weights_only=True)
        if (
            initial.get("schema") != 1
            or initial.get("architecture") != "stable-conditioned-lstm"
            or initial.get("hidden_size") != hidden
            or initial.get("layers") != layers
        ):
            raise ValueError("initial stable RAT checkpoint is incompatible")
        model.load_state_dict(initial["state_dict"], strict=True)
        model.input_coef = float(initial.get("input_coef", derived_input_coef))
        model.project_()
        initial_hash = _sha256(init)
    if compile_model:
        model.compile()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = GammatoneLoss().to(target_device)
    history = []
    best_score = float("inf")
    best_state = None
    start_epoch = 0
    latest = output / "latest.pt"
    if latest.exists():
        checkpoint = torch.load(latest, map_location=target_device, weights_only=True)
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = checkpoint["history"]
        best_score = float(checkpoint["best_score"])
        best_state = checkpoint["best_state"]
        start_epoch = int(checkpoint["epoch"])
        generator_state = checkpoint.get("generator_state")
    else:
        generator_state = None
    output.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(seed)
    if generator_state is not None:
        generator.set_state(generator_state.cpu())
    elif init is not None:
        initial_calibration = _calibrate(model, calibrate, target_device)
        best_score = initial_calibration["score"]
        best_state = {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        history.append(
            {
                "epoch": 0,
                "source": "initial_checkpoint",
                "calibrate": initial_calibration,
            }
        )
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(history[-1], sort_keys=True), flush=True)
    for epoch in range(start_epoch, epochs):
        model.train()
        order = torch.randperm(len(fit[0]), generator=generator)
        losses = []
        for begin in range(0, len(order), batch_size):
            indices = order[begin : begin + batch_size]
            dry = fit[0][indices].to(target_device)
            wet = fit[1][indices].to(target_device)
            controls = fit[2][indices].to(target_device)
            if augmentation:
                offset = float(torch.rand((), generator=generator))
                dry = _fractional_delay(dry, offset)
                wet = _fractional_delay(wet, offset)
            state = None
            with torch.no_grad():
                _, state = model(dry[:, :1_024], controls, state)
            for frame in range(1_024, dry.shape[1], frames):
                stop = min(frame + frames, dry.shape[1])
                prediction, state = model(dry[:, frame:stop], controls, state)
                loss = _training_loss(
                    prediction,
                    wet[:, frame:stop],
                    loss_fn,
                    esr_weight,
                    peak_weight,
                    quiet_weight,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        torch.nan_to_num(parameter.grad, nan=0.0, posinf=1.0e5, neginf=-1.0e5, out=parameter.grad)
                optimizer.step()
                model.project_()
                state = _detach(state)
                losses.append(float(loss.detach()))
        row = {"epoch": epoch + 1, "train_gfb": float(np.mean(losses))}
        if (epoch + 1) % eval_every == 0 or epoch + 1 == epochs:
            calibration = _calibrate(model, calibrate, target_device)
            row["calibrate"] = calibration
            if calibration["score"] < best_score:
                best_score = calibration["score"]
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        history.append(row)
        torch.save(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
                "best_score": best_score,
                "best_state": best_state,
                "generator_state": generator.get_state(),
            },
            latest,
        )
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        print(json.dumps(row, sort_keys=True), flush=True)
    if best_state is None:
        raise RuntimeError("training produced no calibrated checkpoint")
    model.load_state_dict(best_state)
    model.project_()
    stability = model.audit()
    calibration = _calibrate(model, calibrate, target_device)
    candidate = output / "rat-stable.pt"
    torch.save(_runtime_payload(model), candidate)
    report = {
        "schema": 1,
        "status": "trained",
        "paper": PAPER,
        "source_implementation": "independent",
        "epochs": epochs,
        "fit_files": len(fit_paths),
        "calibrate_files": len(calibrate_paths),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "esr_weight": esr_weight,
        "peak_weight": peak_weight,
        "quiet_weight": quiet_weight,
        "tbptt_frames": frames,
        "hidden": hidden,
        "layers": layers,
        "seed": seed,
        "training_device": str(target_device),
        "compiled": compile_model,
        "augmentation": augmentation,
        "input_coef": model.input_coef,
        "derived_input_coef": derived_input_coef,
        "initial_checkpoint": str(init) if init is not None else None,
        "initial_checkpoint_sha256": initial_hash,
        "calibration": calibration,
        "stability": stability,
        "checkpoint": str(candidate),
        "checkpoint_sha256": _sha256(candidate),
        "dataset_audit": audit,
        "quality_policy": {
            "source_files_read_only": True,
            "automatic_normalization": False,
            "automatic_limiting": False,
            "lossy_reencoding": False,
            "physical_audio_devices_used": False,
        },
    }
    if final:
        report["official_eval"] = official_evaluate(candidate, corpus)
    (output / "complete.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--esr-weight", type=float, default=0.0)
    parser.add_argument("--peak-weight", type=float, default=0.0)
    parser.add_argument("--quiet-weight", type=float, default=0.0)
    parser.add_argument("--frames", type=int, default=2_048)
    parser.add_argument("--hidden", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--maximum-train-files", type=int)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--no-compile", action="store_true")
    parser.add_argument("--init", type=Path)
    args = parser.parse_args()
    report = train(
        args.corpus.resolve(),
        args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        frames=args.frames,
        hidden=args.hidden,
        layers=args.layers,
        seed=args.seed,
        maximum_train_files=args.maximum_train_files,
        eval_every=args.eval_every,
        augmentation=not args.no_augmentation,
        final=args.final,
        device_name=args.device,
        compile_model=not args.no_compile,
        init=args.init,
        learning_rate=args.learning_rate,
        esr_weight=args.esr_weight,
        peak_weight=args.peak_weight,
        quiet_weight=args.quiet_weight,
    )
    print(json.dumps({"status": report["status"], "calibration": report["calibration"], "stability": report["stability"]}, sort_keys=True))


if __name__ == "__main__":
    main()
