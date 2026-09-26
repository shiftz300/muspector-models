"""Canonical Muspector package validation and research-artifact migration.

This module performs file work only.  It never opens audio or MIDI devices and
never rewrites source checkpoints.  A package is accepted only when it owns one
of classifier, forward, inverse, or order.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = 1
KINDS = frozenset({"classifier", "forward", "inverse", "order"})
QUALITIES = frozenset({"experimental", "development", "release"})
IDENTIFIER = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
OPTIONAL = frozenset({"identity", "provenance"})
QUALITY = {
    "source_audio_immutable": True,
    "preserve_frames": True,
    "preserve_channels": True,
    "preserve_sample_rate": True,
    "processing_sample_format": "float32",
    "automatic_normalization": False,
    "automatic_limiting": False,
    "automatic_dither": False,
    "lossy_reencoding": False,
    "bypass_max_absolute_error": 0.0,
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError(f"invalid {label}: {value!r}")
    return value


def _path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("artifact path is empty")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"artifact path escapes package: {value!r}")
    if value == "package.json":
        raise ValueError("package.json is reserved")
    return value


def validate(package: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema", "id", "version", "display_name", "kind", "quality",
        "device", "runtime", "sample_rate", "channels", "controls",
        "audio_behavior", "audio_quality", "artifacts", "license",
        "evidence", "limitations",
    }
    extra = set(package) - required - OPTIONAL
    missing = required - set(package)
    if missing or extra:
        raise ValueError(f"package fields differ: missing={sorted(missing)} extra={sorted(extra)}")
    if package["schema"] != SCHEMA:
        raise ValueError("unsupported package schema")
    _identifier(package["id"], "package id")
    _identifier(package["version"], "package version")
    if not isinstance(package["display_name"], str) or not package["display_name"].strip():
        raise ValueError("package display name is empty")
    if package["kind"] not in KINDS or package["quality"] not in QUALITIES:
        raise ValueError("invalid package kind or quality")
    if package["device"] is not None:
        _identifier(package["device"], "device id")
    runtime = package["runtime"]
    if set(runtime) != {"backend", "entrypoint", "minimum_client_version"} or not all(
        isinstance(value, str) and value.strip() for value in runtime.values()
    ):
        raise ValueError("invalid runtime contract")

    controls = package["controls"]
    if not isinstance(controls, list):
        raise ValueError("controls must be a list")
    control_ids: set[str] = set()
    for control in controls:
        if set(control) != {"id", "label", "unit", "minimum", "maximum", "default"}:
            raise ValueError("invalid control fields")
        control_id = _identifier(control["id"], "control id")
        if control_id in control_ids:
            raise ValueError(f"duplicate control id {control_id}")
        control_ids.add(control_id)
        low, high, default = (float(control[key]) for key in ("minimum", "maximum", "default"))
        if not low < high or not low <= default <= high:
            raise ValueError(f"invalid control range for {control_id}")
        if not str(control["label"]).strip() or not str(control["unit"]).strip():
            raise ValueError(f"invalid control label for {control_id}")

    if package["kind"] == "forward":
        if package["device"] is None:
            raise ValueError("forward package requires a device")
        if package["audio_behavior"] != "loss_preserving_render" or package["audio_quality"] != QUALITY:
            raise ValueError("forward package violates the loss-preserving audio contract")
    elif package["kind"] == "inverse":
        if package["device"] is None or not controls:
            raise ValueError("inverse package requires a device and controls")
        if package["audio_behavior"] == "analysis_only":
            if package["audio_quality"] is not None:
                raise ValueError("analysis-only inverse package cannot claim an audio contract")
        elif package["audio_behavior"] == "loss_preserving_restore":
            if package["audio_quality"] != QUALITY:
                raise ValueError("restoration package violates the loss-preserving audio contract")
        else:
            raise ValueError("inverse package has an invalid audio behavior")
    else:
        if package["audio_behavior"] != "analysis_only" or package["audio_quality"] is not None:
            raise ValueError("classifier/order package must be analysis-only")
        if controls:
            raise ValueError("classifier/order package cannot own controls")

    artifacts = package["artifacts"]
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError("package has no artifacts")
    paths: set[str] = set()
    roles: set[str] = set()
    for artifact in artifacts:
        if set(artifact) != {"path", "role", "bytes", "sha256"}:
            raise ValueError("invalid artifact fields")
        path = _path(artifact["path"])
        role = artifact["role"]
        if path in paths or not isinstance(role, str) or not role or role in roles:
            raise ValueError("artifact paths and roles must be unique")
        paths.add(path)
        roles.add(role)
        if not isinstance(artifact["bytes"], int) or artifact["bytes"] < 0:
            raise ValueError(f"invalid artifact size for {path}")
        if not isinstance(artifact["sha256"], str) or not SHA256.fullmatch(artifact["sha256"]):
            raise ValueError(f"invalid artifact digest for {path}")

    for evidence in package["evidence"]:
        if _path(evidence) not in paths:
            raise ValueError(f"evidence must name a packaged artifact: {evidence}")

    license_data = package["license"]
    if set(license_data) != {"id", "commercial_use", "redistribution", "notice"}:
        raise ValueError("invalid license fields")
    if not isinstance(license_data["id"], str) or not license_data["id"].strip():
        raise ValueError("license id is empty")
    if not all(isinstance(license_data[key], bool) for key in ("commercial_use", "redistribution")):
        raise ValueError("license permissions must be boolean")
    if license_data["notice"] is not None and not isinstance(license_data["notice"], str):
        raise ValueError("license notice must be a string or null")

    if "identity" in package:
        identity = package["identity"]
        if set(identity) != {"manufacturer", "model", "family", "declared"}:
            raise ValueError("invalid declared identity fields")
        if identity["declared"] is not True:
            raise ValueError("model identity must be explicitly declared")
        for key in ("manufacturer", "model", "family"):
            if not isinstance(identity[key], str) or not identity[key].strip():
                raise ValueError(f"declared identity {key} is empty")

    if "provenance" in package:
        provenance = package["provenance"]
        if set(provenance) != {"author", "url", "capture"}:
            raise ValueError("invalid provenance fields")
        for key, value in provenance.items():
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"provenance {key} must be a non-empty string or null")
    return package


def load(path: Path) -> dict[str, Any]:
    return validate(json.loads(path.read_text()))


def validate_collection(path: Path) -> dict[str, Any]:
    collection = json.loads(path.read_text())
    if set(collection) != {"schema", "id", "display_name", "default_install", "packages"}:
        raise ValueError("invalid collection fields")
    if collection["schema"] != SCHEMA:
        raise ValueError("unsupported collection schema")
    _identifier(collection["id"], "collection id")
    members: set[tuple[str, str]] = set()
    for member in collection["packages"]:
        if set(member) != {"package_id", "version", "required"}:
            raise ValueError("invalid collection member")
        key = (_identifier(member["package_id"], "package id"), _identifier(member["version"], "version"))
        if key in members:
            raise ValueError(f"duplicate collection member {key}")
        members.add(key)
    return collection


def validate_catalog(path: Path) -> dict[str, Any]:
    catalog = json.loads(path.read_text())
    if catalog.get("schema") != SCHEMA or catalog.get("id") != "base":
        raise ValueError("unsupported pedal catalog")
    devices: set[str] = set()
    for device in catalog.get("devices", []):
        device_id = _identifier(device.get("id"), "device id")
        if device_id in devices:
            raise ValueError(f"duplicate device {device_id}")
        devices.add(device_id)
        capabilities = device.get("capabilities")
        if not isinstance(capabilities, dict) or set(capabilities) != KINDS:
            raise ValueError(f"device {device_id} must separate all four capabilities")
        for kind, references in capabilities.items():
            if not isinstance(references, list):
                raise ValueError(f"device {device_id} {kind} references are invalid")
            for reference in references:
                _identifier(reference.get("package_id"), "package id")
                if reference.get("quality") not in QUALITIES or not reference.get("availability"):
                    raise ValueError(f"device {device_id} has an invalid {kind} reference")
    return catalog


def materialize(source_path: Path, destination: Path, workspace: Path | None = None) -> list[Path]:
    source_path = source_path.resolve()
    if workspace is None:
        workspace = next(
            (parent for parent in source_path.parents if (parent / "Cargo.toml").is_file()),
            None,
        )
        if workspace is None:
            raise ValueError("cannot locate workspace from migration source")
    workspace = workspace.resolve()
    document = json.loads(source_path.read_text())
    if set(document) != {"schema", "id", "description", "packages"} or document["schema"] != SCHEMA:
        raise ValueError("invalid migration source document")
    _identifier(document["id"], "source collection id")
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    seen: set[tuple[str, str]] = set()
    for entry in document["packages"]:
        if set(entry) != {"package", "sources"}:
            raise ValueError("invalid migration entry")
        package = validate(entry["package"])
        key = (package["id"], package["version"])
        if key in seen:
            raise ValueError(f"duplicate migration package {key}")
        seen.add(key)
        sources = entry["sources"]
        expected = {artifact["path"] for artifact in package["artifacts"]}
        if set(sources) != expected:
            raise ValueError(f"source map differs for {package['id']}")
        target = destination / "packages" / package["id"] / package["version"]
        stage = destination / ".staging" / f"{package['id']}-{package['version']}-{os.getpid()}"
        if target.exists() or stage.exists():
            raise FileExistsError(f"refusing to replace package {target}")
        try:
            stage.mkdir(parents=True)
            for artifact in package["artifacts"]:
                source = (workspace / sources[artifact["path"]]).resolve()
                if workspace not in source.parents:
                    raise ValueError(f"source escapes workspace: {source}")
                if source.stat().st_size != artifact["bytes"] or digest(source) != artifact["sha256"]:
                    raise ValueError(f"source provenance mismatch: {source}")
                output = stage / artifact["path"]
                output.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, output)
            (stage / "package.json").write_text(json.dumps(package, indent=2) + "\n")
            verify(stage)
            target.parent.mkdir(parents=True, exist_ok=True)
            stage.rename(target)
            written.append(target)
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise
    return written


def install(
    manifest_path: Path,
    destination: Path,
    source_directory: Path | None = None,
) -> Path:
    """Materialize one source-tree package manifest into a package directory."""

    manifest_path = manifest_path.resolve()
    source_directory = (
        source_directory.resolve() if source_directory is not None else manifest_path.parent
    )
    package = load(manifest_path)
    target = destination.resolve() / "packages" / package["id"] / package["version"]
    stage = destination.resolve() / ".staging" / f"{package['id']}-{package['version']}-{os.getpid()}"
    if target.exists() or stage.exists():
        raise FileExistsError(f"refusing to replace package {target}")
    try:
        stage.mkdir(parents=True)
        for artifact in package["artifacts"]:
            source = (source_directory / artifact["path"]).resolve()
            if source_directory != source and source_directory not in source.parents:
                raise ValueError(f"source escapes asset directory: {source}")
            if source.stat().st_size != artifact["bytes"] or digest(source) != artifact["sha256"]:
                raise ValueError(f"source provenance mismatch: {source}")
            output = stage / artifact["path"]
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, output)
        (stage / "package.json").write_text(json.dumps(package, indent=2) + "\n")
        verify(stage)
        target.parent.mkdir(parents=True, exist_ok=True)
        stage.rename(target)
        return target
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def index(root: Path, url: str) -> Path:
    """Write the static HTTP index consumed by the Rust standalone store."""

    root = root.resolve()
    packages = []
    for manifest_path in sorted((root / "packages").glob("*/*/package.json")):
        package = verify(manifest_path.parent)
        relative = manifest_path.relative_to(root).as_posix()
        packages.append(
            {
                "package_id": package["id"],
                "version": package["version"],
                "manifest_url": f"{url.rstrip('/')}/{relative}",
                "manifest_sha256": digest(manifest_path),
            }
        )
    if not packages:
        raise ValueError("repository has no packages")
    output = root / "index.json"
    output.write_text(json.dumps({"schema": 1, "packages": packages}, indent=2) + "\n")
    return output


def verify(root: Path) -> dict[str, Any]:
    package = load(root / "package.json")
    for artifact in package["artifacts"]:
        path = root / artifact["path"]
        if path.stat().st_size != artifact["bytes"] or digest(path) != artifact["sha256"]:
            raise ValueError(f"installed artifact mismatch: {artifact['path']}")
    return package
