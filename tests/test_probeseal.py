from remix.probeseal import gate


def test_probe_gate_accepts_four_by_four_perfect_seal():
    rows=[]
    for scope,routed in (("rat",True),("other",False)):
        for index in range(4):
            rows.append({"scope":scope,"model":f"{scope}{index}","routed":routed,
                         "automatic_delivery":False,"inputs_unchanged":True,"finite":True})
    accepted,failures,metrics=gate(rows)
    assert accepted and not failures and metrics["rat_recall"]==1.0 and metrics["other_false_route"]==0.0
