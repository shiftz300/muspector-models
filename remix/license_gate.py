"""Fail-closed license policy for data, gradients, weights, and audio.

This is a conservative engineering gate, not legal advice.  It prevents a
research-only benchmark from silently influencing a redistributable model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections.abc import Iterable, Mapping


REQUIRED_FIELDS = {
    "id",
    "license",
    "rights_status",
    "gradient_scope",
    "weight_scope",
    "audio_scope",
    "admission",
}
PRODUCT_WEIGHT_PREFIXES = ("redistributable",)
RESTRICTED_LICENSE_TOKENS = ("-NC", "-ND")
UNCLEAR_LICENSES = {"NOASSERTION", "LicenseRef-Mixed-Per-Record", "LicenseRef-NonCommercial-Local"}


def load_registry(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if payload.get("schema") != 1 or not isinstance(payload.get("sources"), list):
        raise ValueError(f"invalid data source registry: {path}")
    return payload


def by_id(registry: dict) -> dict[str, dict]:
    result = {}
    for source in registry["sources"]:
        missing = REQUIRED_FIELDS - set(source)
        if missing:
            raise ValueError(f"source {source.get('id')} lacks license fields: {sorted(missing)}")
        identifier = source["id"]
        if identifier in result:
            raise ValueError(f"duplicate source id: {identifier}")
        result[identifier] = source
    return result


def product_weight_allowed(source: dict) -> bool:
    return str(source["weight_scope"]).startswith(PRODUCT_WEIGHT_PREFIXES)


def attribution(source: dict) -> dict | None:
    if source["license"].startswith("CC-BY-"):
        return {
            "source_id": source["id"],
            "title": source["title"],
            "record_url": source.get("record_url"),
            "doi": source.get("doi"),
            "license": source["license"],
        }
    return None


def authorize_product_weights(registry: dict, source_ids: Iterable[str]) -> dict:
    sources = by_id(registry)
    selected = []
    blocked = []
    attributions = []
    for identifier in source_ids:
        if identifier not in sources:
            blocked.append({"id": identifier, "reason": "unregistered-source"})
            continue
        source = sources[identifier]
        license_name = str(source["license"])
        reason = None
        if any(token in license_name.upper() for token in RESTRICTED_LICENSE_TOKENS):
            reason = f"restricted-license:{license_name}"
        elif license_name in UNCLEAR_LICENSES or source["rights_status"] in {
            "mixed-needs-record-audit",
            "needs-record-audit",
            "rights-field-empty-on-record-page",
        }:
            reason = "rights-not-clear"
        elif not product_weight_allowed(source):
            reason = f"weight-scope:{source['weight_scope']}"
        if reason:
            blocked.append({"id": identifier, "reason": reason})
            continue
        selected.append(identifier)
        credit = attribution(source)
        if credit:
            attributions.append(credit)
    return {
        "authorized": not blocked and bool(selected),
        "sources": selected,
        "blocked": blocked,
        "required_attribution": attributions,
    }


def authorize_product_uses(
    registry: dict,
    requirements: Mapping[str, str | Iterable[str]],
) -> dict:
    """Authorize product weights only for explicitly registered task uses.

    A permissive weight license is necessary but not sufficient.  For example,
    EGFxSet may train the family recognizer and provide Clean programs, while its
    normalized Wet files are deliberately excluded from restoration gradients.
    """

    sources = by_id(registry)
    normalized: dict[str, tuple[str, ...]] = {}
    for identifier, uses in requirements.items():
        values = (uses,) if isinstance(uses, str) else tuple(uses)
        if not values or any(not value for value in values):
            raise ValueError(f"source {identifier} has no required product use")
        normalized[identifier] = values

    weight_result = authorize_product_weights(registry, normalized)
    blocked = list(weight_result["blocked"])
    weight_authorized = set(weight_result["sources"])
    selected = []
    attributions = []
    for identifier, required_uses in normalized.items():
        if identifier not in weight_authorized:
            continue
        allowed_uses = set(sources[identifier].get("allowed_uses", ()))
        missing = sorted(set(required_uses) - allowed_uses)
        if missing:
            blocked.append({
                "id": identifier,
                "reason": f"use-not-allowed:{','.join(missing)}",
            })
            continue
        selected.append(identifier)
        credit = attribution(sources[identifier])
        if credit:
            attributions.append(credit)
    return {
        "authorized": not blocked and bool(selected),
        "sources": selected,
        "required_uses": {key: list(value) for key, value in normalized.items()},
        "blocked": blocked,
        "required_attribution": attributions,
    }


def audit(registry: dict) -> dict:
    sources = by_id(registry)
    contradictions = []
    for source in sources.values():
        license_name = str(source["license"])
        if product_weight_allowed(source) and any(
            token in license_name.upper() for token in RESTRICTED_LICENSE_TOKENS
        ):
            contradictions.append(
                f"{source['id']} marks restricted {license_name} as product-weight eligible"
            )
        if product_weight_allowed(source) and (
            license_name in UNCLEAR_LICENSES or "needs" in str(source["rights_status"])
        ):
            contradictions.append(
                f"{source['id']} marks unresolved rights as product-weight eligible"
            )
    product_sources = sorted(
        source["id"] for source in sources.values() if product_weight_allowed(source)
    )
    research_only = sorted(
        source["id"] for source in sources.values() if not product_weight_allowed(source)
    )
    foundation = authorize_product_weights(
        registry,
        (
            "guitarjam",
            "guitar-techs",
            "eg-ipt",
            "longitudinal-guitar-string-ageing",
            "freepats-electric-guitar-direct",
            "karoryfer-emilyguitar",
            "multimodal-electric-guitar-data",
            "dafx25-guitar-effects-chains",
            "egfxset",
            "muspector-dsp",
            "aachen-chapel-rir",
            "marshall-jvm410h",
        ),
    )
    explicitly_blocked = authorize_product_weights(
        registry,
        (
            "asrnn-physical-effects",
            "apple-au-local",
            "spotify-pedalboard-renderer",
            "tonetwist-local-collection",
            "remfx-local",
            "remfx-pretrained-models",
            "pod-set",
        ),
    )
    task_use_probes = {
        "egfx_restoration_must_be_blocked": authorize_product_uses(
            registry, {"egfxset": "train-restoration"}
        ),
        "guitar_techs_amp_must_be_blocked": authorize_product_uses(
            registry, {"guitar-techs": "train-amp"}
        ),
        "aachen_reverb_must_be_allowed": authorize_product_uses(
            registry, {"aachen-chapel-rir": "train-reverb"}
        ),
        "marshall_amp_must_be_allowed": authorize_product_uses(
            registry, {"marshall-jvm410h": "train-amp"}
        ),
        "ok5_reverb_validation_must_be_allowed": authorize_product_uses(
            registry, {"ok5-rir": "validate-reverb"}
        ),
        "ok5_reverb_training_must_be_blocked": authorize_product_uses(
            registry, {"ok5-rir": "train-reverb"}
        ),
        "openair_reverb_validation_must_be_allowed": authorize_product_uses(
            registry, {"openair-rir-external-v1": "validate-reverb"}
        ),
        "openair_reverb_training_must_be_blocked": authorize_product_uses(
            registry, {"openair-rir-external-v1": "train-reverb"}
        ),
        "openslr26_reverb_training_must_be_allowed": authorize_product_uses(
            registry, {"openslr26-simulated-rir-external-v1": "train-reverb"}
        ),
        "rochester_reverb_validation_must_be_allowed": authorize_product_uses(
            registry, {"rochester-rir-fresh-v2": "validate-reverb"}
        ),
        "rochester_reverb_training_must_be_blocked": authorize_product_uses(
            registry, {"rochester-rir-fresh-v2": "train-reverb"}
        ),
    }
    passed = (
        not contradictions
        and foundation["authorized"]
        and not explicitly_blocked["authorized"]
        and not task_use_probes["egfx_restoration_must_be_blocked"]["authorized"]
        and not task_use_probes["guitar_techs_amp_must_be_blocked"]["authorized"]
        and task_use_probes["aachen_reverb_must_be_allowed"]["authorized"]
        and task_use_probes["marshall_amp_must_be_allowed"]["authorized"]
        and task_use_probes["ok5_reverb_validation_must_be_allowed"]["authorized"]
        and not task_use_probes["ok5_reverb_training_must_be_blocked"]["authorized"]
        and task_use_probes["openair_reverb_validation_must_be_allowed"]["authorized"]
        and not task_use_probes["openair_reverb_training_must_be_blocked"]["authorized"]
        and task_use_probes["openslr26_reverb_training_must_be_allowed"]["authorized"]
        and task_use_probes["rochester_reverb_validation_must_be_allowed"]["authorized"]
        and not task_use_probes["rochester_reverb_training_must_be_blocked"]["authorized"]
    )
    return {
        "schema": 1,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "policy": "fail-closed three-tier license gate; engineering policy, not legal advice",
        "product_weight_sources": product_sources,
        "research_only_or_blocked_sources": research_only,
        "foundation_product_training": foundation,
        "restricted_source_probe": explicitly_blocked,
        "task_use_probes": task_use_probes,
        "contradictions": contradictions,
        "research_isolation": {
            "gradients": False,
            "calibration": False,
            "model_selection": False,
            "distillation": False,
            "shared_learned_encoder": False,
            "external_veto_metrics": True,
        },
    }


def require_product_weights(registry_path: Path, source_ids: Iterable[str]) -> dict:
    result = authorize_product_weights(load_registry(registry_path), source_ids)
    if not result["authorized"]:
        raise PermissionError(f"product-weight source gate failed: {result['blocked']}")
    return result


def require_product_uses(
    registry_path: Path,
    requirements: Mapping[str, str | Iterable[str]],
) -> dict:
    result = authorize_product_uses(load_registry(registry_path), requirements)
    if not result["authorized"]:
        raise PermissionError(f"product-use source gate failed: {result['blocked']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("remix/data_sources.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/foundation/license/audit.json"))
    args = parser.parse_args()
    report = audit(load_registry(args.registry))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
