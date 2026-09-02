from remix.verify import GATES, gate


def row(scope, routed):
    return {"scope": scope, "model": f"{scope}-{routed}", "routed": routed,
            "automatic_delivery": False, "inputs_unchanged": True, "nonfinite": 0}


def test_gate_accepts_exact_frozen_boundary():
    rows = []
    for model in range(4):
        for routed in ([True, True, True, False] if model == 0 else [True] * 4):
            item = row("rat", routed); item["model"] = f"rat-{model}"; rows.append(item)
    for model in range(4):
        for routed in ([True, False, False, False] if model == 0 else [False] * 4):
            item = row("other", routed); item["model"] = f"other-{model}"; rows.append(item)
    accepted, failures, metrics = gate(rows)
    assert accepted and not failures
    assert metrics["rat_recall"] >= GATES["rat_recall"]
    assert metrics["other_false_route"] <= GATES["other_false_route"]


def test_gate_rejects_false_route():
    rows = []
    for model in range(4):
        for _ in range(4):
            item = row("rat", True); item["model"] = f"rat-{model}"; rows.append(item)
            item = row("other", model == 0); item["model"] = f"other-{model}"; rows.append(item)
    accepted, failures, _ = gate(rows)
    assert not accepted
    assert any(item.startswith("other_false_route") for item in failures)
