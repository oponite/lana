"""Independent finite constraint oracle and executable protection study.

Policy constants are experimental assumptions. No LLM or incident outcome grades
the program. Unknowns are finite alternatives, never default Boolean values.
"""
import copy
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAPS = {False: (0, 1, 2, 2), True: (0, 1, 2, 3)}
CONSENT = (True, True, True, True, False)
POLICIES = [dict(id=f"standard_{i:02}", duty=duty, severe=severe,
                 threat_min=threat, success_min=success, burden_max=burden)
            for i, (duty, severe, threat, success, burden) in enumerate(
                itertools.product((False, True), (False, True), (50, 90), (50, 90), (0, 1)))]
DEFAULT = POLICIES[25]  # duty, severe, threat=50, success=50, burden=1


def action(name="help", **changes):
    a = dict(id=name, feasible=True, protects=True, success=100, time=1,
             duration=1, burden=0, impacts=[1, 0],
             consents=[[True, True, True, False, False]] * 2,
             unavailable=[False, False], refused=[False, False], uses_uninvolved=False)
    a.update(changes)
    return copy.deepcopy(a)


def world(**changes):
    w = dict(active=True, threat_grade=3, threat_probability=100, deadline=1,
             people=[dict(id="Alice", role="source"), dict(id="Bob", role="protected")],
             actions=[action("wait", protects=False, success=0, duration=0, impacts=[0, 0]), action()],
             require=[], forbid=[], context={})
    w.update(changes)
    return copy.deepcopy(w)


def case(name, w=None, policy=None):
    return dict(id=name, world=copy.deepcopy(w if w is not None else world()),
                policy=copy.deepcopy(policy if policy is not None else DEFAULT))


def reference(c):
    """Enumerate one-choice plans satisfying declarative constraints.

    Lana builds the acceptable set procedurally. This oracle instead enumerates
    Boolean selections and filters them; policy tables alone are shared meaning.
    """
    w, p = c["world"], c["policy"]
    acts = w["actions"]
    danger = w["active"] and w["threat_grade"] > 0 and w["threat_probability"] >= p["threat_min"]
    adequate = [a["feasible"] and a["protects"] and a["time"] <= w["deadline"]
                and a["success"] >= p["success_min"] for a in acts]
    consent = [[tuple(bits) == CONSENT for bits in a["consents"]] for a in acts]
    exceptional = [[
        danger and adequate[j] and a["duration"] <= w["deadline"]
        and impact <= CAPS[p["severe"]][w["threat_grade"]]
        and (impact != 3 or (w["threat_grade"] == 3 and w["threat_probability"] >= 90))
        and (person["role"] == "source" or
             (person["role"] == "protected" and a["unavailable"][i] and not a["refused"][i]))
        for i, (person, impact) in enumerate(zip(w["people"], a["impacts"]))]
        for j, a in enumerate(acts)]
    eligible = [a["feasible"] and not a["uses_uninvolved"] and all(
        impact == 0 or consent[j][i] or exceptional[j][i]
        for i, impact in enumerate(a["impacts"])) for j, a in enumerate(acts)]
    vectors = [tuple(a["impacts"]) + (a["duration"], a["burden"]) for a in acts]
    dominates = {(b, a) for a in range(len(acts)) for b in range(len(acts))
                 if all(x <= y for x, y in zip(vectors[b], vectors[a])) and vectors[b] != vectors[a]}
    necessary = [danger and adequate[a] and not any(
        eligible[b] and adequate[b] and (b, a) in dominates for b in range(len(acts)))
        for a in range(len(acts))]
    permitted = [eligible[j] and all(impact == 0 or consent[j][i] or necessary[j]
                                   for i, impact in enumerate(a["impacts"])) for j, a in enumerate(acts)]
    duty = p["duty"] and danger and w["threat_grade"] >= 2 and any(
        allowed and effective and a["burden"] <= p["burden_max"]
        for a, allowed, effective in zip(acts, permitted, adequate))
    plans = []
    for selected in itertools.product((False, True), repeat=len(acts)):
        constraints = [sum(selected) == 1,
                       all(not chosen or permitted[i] for i, chosen in enumerate(selected)),
                       all(not chosen or acts[i]["id"] not in w["forbid"] for i, chosen in enumerate(selected)),
                       all(any(chosen and acts[i]["id"] == name for i, chosen in enumerate(selected))
                           for name in w["require"]),
                       not duty or any(chosen and adequate[i] for i, chosen in enumerate(selected))]
        if all(constraints):
            plans.append(selected)
    if not plans:
        return dict(id=c["id"], validity="conflict", accepted=[], statuses={}, must_help=None)
    statuses = {a["id"]: ("required" if all(row[i] for row in plans) else
                            "optional" if any(row[i] for row in plans) else "forbidden")
                for i, a in enumerate(acts)}
    return dict(id=c["id"], validity="valid",
                accepted=[a["id"] for a in acts if statuses[a["id"]] != "forbidden"],
                statuses=statuses, must_help=all(any(on and adequate[i] for i, on in enumerate(row)) for row in plans))


def fictional_cases():
    """24 frozen witnesses; literal accepted sets under DEFAULT are independent goldens."""
    result = []

    def add(name, w, accepted):
        result.append((case(name, w), accepted))

    add("danger_absent", world(active=False), ["wait"])
    add("danger_active", world(), ["help"])
    w = world(active=False); w["actions"][1]["consents"][0] = list(CONSENT)
    add("consent_free", w, ["wait", "help"])
    w["actions"][1]["consents"][0][2] = False
    add("consent_coerced", w, ["wait"])
    w = world(); w["people"][0]["role"] = "protected"
    w["actions"][1]["unavailable"][0] = True; w["actions"][1]["refused"][0] = True
    add("competent_refusal", w, ["wait"])
    w["actions"][1]["refused"][0] = False; w["actions"][1]["consents"][0][0] = False
    add("emergency_incapacity", w, ["help"])
    w = world(threat_grade=1); w["actions"][1]["impacts"][0] = 2
    add("minor_threat", w, ["wait"])
    w["threat_grade"] = 3
    add("grave_threat", w, ["help"])
    w = world(); w["actions"].append(action("call", impacts=[0, 0], duration=0))
    add("lesser_available", w, ["call"])
    w["actions"][2]["feasible"] = False
    add("lesser_unavailable", w, ["help"])
    add("protection_timely", world(), ["help"])
    w = world(); w["actions"][1]["time"] = 2
    add("protection_late", w, ["wait"])
    add("danger_continues", world(), ["help"])
    w = world(); w["actions"][1]["duration"] = 2
    add("force_continues_after_deadline", w, ["wait"])
    add("burden_low", world(), ["help"])
    w = world(); w["actions"][1]["burden"] = 3
    add("burden_high", w, ["wait", "help"])
    add("no_competing_duty", world(), ["help"])
    add("competing_duty", world(require=["wait"]), [])
    add("one_way_to_help", world(), ["help"])
    w = world(); w["actions"].append(action("other_help"))
    add("two_ways_to_help", w, ["help", "other_help"])
    add("uninvolved_respected", world(), ["help"])
    w = world(); w["actions"][1]["uses_uninvolved"] = True
    add("uninvolved_sacrificed", w, ["wait"])
    add("danger_report_accurate", world(), ["help"])
    add("danger_report_mistaken", world(active=False, threat_grade=0, threat_probability=0), ["wait"])
    return result


def generated_cases():
    for bits in itertools.product((False, True), repeat=5):
        w = world(active=False); w["actions"][1]["consents"][0] = list(bits)
        yield "consent", case("consent_" + str(bits), w)
    for role, consented, refused, unavailable, impact, active in itertools.product(
            ("source", "protected", "uninvolved"), (False, True), (False, True),
            (False, True), range(4), (False, True)):
        if consented and refused:
            continue  # Contradictory consent is an invalid input, tested separately.
        w = world(active=active); w["people"][0]["role"] = role
        a = w["actions"][1]
        a["consents"][0] = list(CONSENT if consented else (True, True, True, False, False))
        a["refused"][0] = refused; a["unavailable"][0] = unavailable; a["impacts"][0] = impact
        yield "protective_exception", case(f"exception_{role}_{consented}_{refused}_{unavailable}_{impact}_{active}", w)
    for threat, impact, severe in itertools.product(range(4), range(4), (False, True)):
        w = world(threat_grade=threat); w["actions"][1]["impacts"][0] = impact
        p = dict(DEFAULT, severe=severe)
        yield "proportion", case(f"grade_{threat}_{impact}_{severe}", w, p)
    # Six independent necessity premises, realized as factual witnesses.
    for bits in itertools.product((False, True), repeat=6):
        active, plausible, feasible, effective, timely, lesser = bits
        w = world(active=active, threat_probability=100 if plausible else 0)
        w["actions"][1].update(feasible=feasible, success=100 if effective else 0, time=1 if timely else 2)
        w["actions"].append(action("call", feasible=lesser, impacts=[0, 0], duration=0))
        yield "necessity", case("necessity_" + str(bits), w, dict(DEFAULT, duty=False))
    for bits in itertools.product((False, True), repeat=8):
        enabled, active, serious, plausible, feasible, effective, timely, cheap = bits
        w = world(active=active, threat_grade=2 if serious else 1, threat_probability=100 if plausible else 0)
        w["actions"][1].update(impacts=[0, 0], duration=0, feasible=feasible,
                               success=100 if effective else 0, time=1 if timely else 2, burden=0 if cheap else 3)
        yield "duty", case("duty_" + str(bits), w, dict(DEFAULT, duty=enabled))
    for n in range(1, 5):
        for chosen in itertools.product((False, True), repeat=n):
            w = world(active=False, actions=[action(str(i), protects=False, impacts=[0, 0], duration=0) for i in range(n)])
            w["forbid"] = [str(i) for i, on in enumerate(chosen) if not on]
            yield "plan_sets", case(f"plans_{n}_{chosen}", w)
    for p in POLICIES:
        for field, cutoff in (("threat_probability", p["threat_min"]), ("success", p["success_min"]), ("burden", p["burden_max"])):
            ceiling = 3 if field == "burden" else 100
            for value in sorted({0, ceiling, max(0, cutoff - 1), cutoff, min(ceiling, cutoff + 1)}):
                w = world()
                (w if field == "threat_probability" else w["actions"][1])[field] = value
                yield "boundaries", case(f"boundary_{field}_{value}_{p['id']}", w, p)
        for probability in (89, 90, 91):
            w = world(threat_probability=probability); w["actions"][1]["impacts"][0] = 3
            yield "irreversible_boundary", case(f"irreversible_{probability}_{p['id']}", w, p)
    for c, _ in fictional_cases():
        for p in POLICIES:
            yield "fictional", case(c["id"] + ":" + p["id"], c["world"], p)


def run_study(lana, output, command, check):
    fixture = ROOT / "tests/regression/is_ought_protection.lana"
    bytecode = output / "protection.labc"
    logs, counts = [], {}
    summary = dict(status="running", standards=POLICIES, counts=counts, mutants={}, information=[], incidents=[])

    def run(mode, data, executable=bytecode):
        return command([lana, "run", executable, "--seed", "7", "--", mode, json.dumps(data, separators=(",", ":"))])

    def batch(entries, executable=bytecode, verify=True):
        data = [c for _, c in entries]
        r = run("batch", data, executable)
        check(r.returncode == 0, f"protection runtime failed: {r.stderr}; {data}")
        rows = [json.loads(line) for line in r.stdout.splitlines()]
        check(len(rows) == len(data), "missing protection output")
        for (group, c), actual in zip(entries, rows):
            expected = reference(c)
            if verify:
                logs.append(dict(group=group, input=c, expected=expected, actual=actual))
                counts[group] = counts.get(group, 0) + 1
                check(actual == expected, f"protection mismatch: {c['id']}: {actual} != {expected}")
        return rows

    def probe(cases, form="possibility", resolve=False, weights=None, select=-1, executable=bytecode):
        payload = dict(cases=cases, form=form, action="help", resolve=resolve,
                       weights=weights if weights is not None else [1 / len(cases)] * len(cases), select=select)
        expected = [reference(c)["statuses"]["help"] for c in cases]
        original_expected = expected[:]
        if select >= 0:
            expected = [expected[select]]
        support = sorted(set(expected))
        r = run("information", payload, executable)
        unresolved = resolve and len(support) != 1
        check((r.returncode != 0 and "LANA_ERR_UNRESOLVED_VALUE" in r.stderr) if unresolved else r.returncode == 0,
              f"information {form} returned wrong resolution: {r.stderr}")
        rows = [json.loads(line) for line in r.stdout.splitlines()]
        snap = rows[0]["snapshot"]
        wanted_form = "definite" if form == "paths" and len(support) == 1 else form
        check(snap["form"] == wanted_form, f"wrong form {snap}")
        if "support" in snap:
            actual = [row.get("value", row.get("assignment", {}).get("status")) for row in snap["support"]]
            check(sorted(set(actual)) == support, f"wrong information support {snap}")
        for field, values in (("may", [s != "forbidden" for s in support]), ("must", [s == "required" for s in support])):
            observed = rows[0][field]
            if "support" in observed:
                check({v.get("value", v.get("assignment", {}).get(field)) for v in observed["support"]} == set(values), f"wrong {field}")
        check(rows[0]["dependency_completeness"] == "not_certified", "unsupported provenance claim")
        if resolve and not unresolved:
            check(rows[-1]["resolved"] == support[0], "wrong exact resolution")
        if form == "distribution":
            weights_by_status = {s: (1 if select >= 0 else sum(w for s0, w in zip(original_expected, payload["weights"]) if s0 == s)) for s in support}
            check(all(abs(v["weight"] - weights_by_status[v["value"]]) < 1e-12 for v in snap["support"]), "changed weights")
        summary["information"].append(dict(input=payload, support=support, output=rows, unresolved=unresolved))
        return rows

    try:
        compiled = command([lana, "compile", fixture, "-o", bytecode])
        check(compiled.returncode == 0, compiled.stderr)
        for c, expected in fictional_cases():
            check(reference(c)["accepted"] == expected, f"oracle contradicts frozen golden: {c['id']}")
        summary["fictional_goldens"] = [dict(input=c, accepted=expected) for c, expected in fictional_cases()]
        pending = []
        for item in generated_cases():
            pending.append(item)
            if len(pending) == 16:
                batch(pending); pending = []
        if pending:
            batch(pending)
        # Lower impact to others does not dominate if it increases the helper's risk.
        tradeoff = world()
        tradeoff["actions"].append(action("risky_help", impacts=[0, 0], duration=0, burden=3))
        c = case("actor_burden_tradeoff", tradeoff)
        check(reference(c)["accepted"] == ["help", "risky_help"], "actor burden ignored in necessity")
        batch([("actor_burden", c)])

        # Structural invariance, each checked against both its original and the oracle.
        for original, _ in fictional_cases():
            for p in POLICIES:
                c = case(original["id"], original["world"], p)
                renamed = copy.deepcopy(c)
                for person in renamed["world"]["people"]: person["id"] = "renamed_" + person["id"]
                irrelevant = copy.deepcopy(c); irrelevant["world"]["context"] = {"weather": "not used"}
                reordered = copy.deepcopy(c); reordered["world"]["actions"].reverse()
                persons = copy.deepcopy(c); persons["world"]["people"].reverse()
                for a in persons["world"]["actions"]:
                    for field in ("impacts", "consents", "unavailable", "refused"): a[field].reverse()
                rows = batch([("invariance", changed) for changed in (c, renamed, irrelevant, reordered, persons)])
                check(all(row["statuses"] == rows[0]["statuses"] for row in rows), "invariance failed")

        for p in POLICIES:
            base = case("relations", policy=p)
            off = copy.deepcopy(base); off["policy"]["duty"] = False
            burden = copy.deepcopy(base); burden["world"]["actions"][1]["burden"] = 3
            infeasible = copy.deepcopy(base); infeasible["world"]["actions"].append(action("impossible", feasible=False))
            lesser = copy.deepcopy(base); lesser["world"]["actions"].append(action("call", impacts=[0, 0], duration=0))
            rows = batch([("relations", c) for c in (base, off, burden, infeasible, lesser)])
            check(set(rows[0]["accepted"]) <= set(rows[1]["accepted"]), "removing duty narrowed choices")
            check(rows[0]["must_help"] or not rows[2]["must_help"], "higher burden created duty")
            check(rows[3]["statuses"]["help"] == rows[0]["statuses"]["help"], "infeasible option changed answer")
            check(rows[4]["statuses"]["help"] == "forbidden", "lesser adequate option did not exclude force")

        required = case("required")
        optional = case("optional", policy=dict(DEFAULT, duty=False))
        forbidden = case("forbidden", world(active=False))
        for pair in ((optional, required), (optional, forbidden), (required, forbidden), (required, required)):
            for form in ("possibility", "distribution", "joint", "paths"):
                probe(list(pair), form, resolve=True)
                if form == "distribution": probe(list(pair), form, weights=[0.99, 0.01])
        probe([required], "definite", resolve=True)
        for form in ("joint", "possibility", "distribution"):
            probe([optional, required], form, resolve=True, select=1)
        probe([required, required, optional], "possibility")
        # Singleton support still resolves even when its form remains possibility.
        probe([optional], "possibility", resolve=True)
        try_unknown = probe([optional, required], "possibility")
        may_support = try_unknown[0]["may"].get("support", [])
        check([v["value"] for v in may_support] == [True], "permission should be settled despite unknown duty")

        invalids = []
        def invalid(label, change):
            c = case(label); change(c); invalids.append(c)
        invalid("grade", lambda c: c["world"].update(threat_grade=4))
        invalid("grade_fraction", lambda c: c["world"].update(threat_grade=1.5))
        invalid("grade_boolean", lambda c: c["world"].update(threat_grade=True))
        invalid("probability", lambda c: c["world"].update(threat_probability=-1))
        invalid("probability_type", lambda c: c["world"].update(threat_probability="90"))
        invalid("reference", lambda c: c["world"].update(require=["absent"]))
        invalid("consent_size", lambda c: c["world"]["actions"][1]["consents"].__setitem__(0, [True]))
        invalid("consent_type", lambda c: c["world"]["actions"][1]["consents"][0].__setitem__(0, "true"))
        invalid("duplicate_action", lambda c: c["world"]["actions"][1].update(id="wait"))
        invalid("duplicate_person", lambda c: c["world"]["people"][1].update(id="Alice"))
        invalid("empty_actions", lambda c: c["world"].update(actions=[]))
        invalid("missing", lambda c: c["world"].pop("active"))
        invalid("smuggled_judgment", lambda c: c["world"].update(unjustified_threat=True))
        invalid("contradictory_consent", lambda c: c["world"]["actions"][1].update(consents=[list(CONSENT)] * 2, refused=[True, False]))
        for c in invalids:
            r = run("batch", [c]); check(r.returncode != 0 and "LANA_ERR_ASSERTION" in r.stderr, f"accepted invalid {c['id']}")
        for payload in ([], [case("overflow")] * 33):
            r = run("batch", payload); check(r.returncode != 0 and "LANA_ERR_ASSERTION" in r.stderr, "invalid batch accepted")
        r = command([lana, "run", bytecode, "--", "batch", "{"])
        check(r.returncode != 0 and "invalid JSON" in r.stderr, "malformed JSON accepted")
        summary["invalid_inputs"] = len(invalids) + 3
        payload = dict(cases=[optional, required], form="distribution", action="help", resolve=False, weights=[0.5, 0.5], select=-1)
        malformed = [dict(payload, **change) for change in (
            {"weights": [0, 1]}, {"weights": [-0.1, 1.1]}, {"weights": [0.5]},
            {"weights": [0.1, 0.1]}, {"weights": ["0.5", 0.5]},
            {"cases": []}, {"form": "invented"}, {"action": "missing"},
            {"select": 2}, {"resolve": "true"}, {"cases": [witness for witness, _ in fictional_cases() if witness["id"] == "competing_duty"]})]
        for bad in malformed:
            r = run("information", bad)
            check(r.returncode != 0 and "LANA_ERR_ASSERTION" in r.stderr, "invalid information law accepted")
        summary["invalid_inputs"] += len(malformed)

        # Mutants are temporary compiled copies of the actual Lana evaluator.
        text = fixture.read_text()
        witnesses = {c["id"]: c for c, _ in fictional_cases()}
        mutants = {
            "ignored_consent": ('if (!consent(a.consents[i])) {', 'if (false) {', "consent_coerced"),
            "omitted_necessity": ('if (!necessary(a, w, p)) { return false; }', '', "lesser_available"),
            "omitted_proportionality": ('if (!proportionate(a.impacts[i], w, p)) { return false; }', '', "minor_threat"),
            "one_method_mandatory": ('if (array_length(accepted) == 1) { status = "required"; }',
                                     'if (contains(accepted, a.id)) { status = "required"; }', "two_ways_to_help"),
            "unknown_as_false": ('return refine_status(possibility(values), input, values);', 'return inspect_status("optional", input);', None),
            "majority_as_unanimity": ('return refine_status(possibility(values), input, values);',
                                     'if (array_length(values) == 3) { return inspect_status(values[0], input); } return inspect_status(possibility(values), input);', None),
        }
        for name, (old, new, witness) in mutants.items():
            check(old in text, f"mutation anchor absent: {name}")
            source = output / (name + ".lana"); executable = output / (name + ".labc")
            source.write_text(text.replace(old, new))
            r = command([lana, "compile", source, "-o", executable]); check(r.returncode == 0, f"mutant failed compilation: {name}: {r.stderr}")
            if witness:
                c = witnesses[witness]
                actual = batch([("mutant", c)], executable, verify=False)[0]
                killed = actual != reference(c)
            else:
                cs = [required, required, optional] if name == "majority_as_unanimity" else [optional, required]
                r = run("information", dict(cases=cs, form="possibility", action="help", resolve=False, weights=[], select=-1), executable)
                check(r.returncode == 0, f"mutant runtime error does not count: {name}")
                actual = json.loads(r.stdout.splitlines()[0])["snapshot"]
                killed = actual["form"] != "possibility" or actual["remaining_alternatives"] != 2
            summary["mutants"][name] = dict(detected=killed, witness=witness, observed=actual)
            check(killed, f"surviving mutant: {name}")

        run_incidents(summary, batch, check)
        summary.update(status="passed", evaluated_cases=sum(counts.values()),
                       fixture_sha256=hashlib.sha256(fixture.read_bytes()).hexdigest(),
                       fixture_bytecode_sha256=hashlib.sha256(bytecode.read_bytes()).hexdigest(),
                       driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       philosophical_status="conditional_on_declared_standards")
    except Exception as error:
        summary.update(status="failed", error=str(error))
        raise
    finally:
        (output / "protection.json").write_text(json.dumps(summary, indent=2) + "\n")
        with (output / "protection-cases.jsonl").open("w") as stream:
            for record in logs: stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    return summary


def run_incidents(summary, batch, check):
    path = ROOT / "tests/fixtures/is_ought_incidents.json"
    document = json.loads(path.read_text())
    summary["incident_fixture_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary["incident_sources"] = document["sources"]
    for incident in document["incidents"]:
        for basis in ("available_then", "later_record"):
            view = incident["views"][basis]
            check(all(field["source_refs"] for field in view["coding"]), "uncited incident field")
            check(all(ref in document["sources"] for field in view["coding"] for ref in field["source_refs"]), "unknown evidence reference")
            check({path[0] for entry in view["coding"] for path in entry["paths"]} == set(view["world"]), "uncoded world field")
            unknown_paths = {tuple(path) for entry in view["coding"] if entry["kind"] == "unknown_alternatives" for path in entry["paths"]}
            check(unknown_paths == {tuple(d["path"]) for d in view["alternatives"]}, "unknown field silently fixed")
            w = copy.deepcopy(view["world"])
            domains = view["alternatives"]
            interpretations = []
            for values in itertools.product(*(d["values"] for d in domains)):
                completed = copy.deepcopy(w)
                for d, value in zip(domains, values):
                    target = completed
                    for key in d["path"][:-1]: target = target[key]
                    target[d["path"][-1]] = value
                interpretations.append(completed)
            assessment = dict(incident=incident["id"], basis=basis, standards=[],
                              interpretation_count=len(interpretations), limitations=incident["limitations"],
                              coding=view["coding"], moral_truth_label=None)
            for p in POLICIES:
                cs = [case(f"{incident['id']}:{basis}:{p['id']}:{i}", value, p) for i, value in enumerate(interpretations)]
                rows = []
                for i in range(0, len(cs), 16): rows.extend(batch([("incident", c) for c in cs[i:i + 16]]))
                statuses = sorted({r["statuses"]["help"] for r in rows if r["validity"] == "valid"})
                witnesses = {s: cs[next(i for i, r in enumerate(rows) if r["statuses"].get("help") == s)] for s in statuses}
                assessment["standards"].append(dict(policy=p["id"], supported_statuses=statuses,
                                                     conflicts=sum(r["validity"] != "valid" for r in rows), witnesses=witnesses))
            assessment["supported_statuses"] = sorted({s for row in assessment["standards"] for s in row["supported_statuses"]})
            assessment["conflicts"] = sum(row["conflicts"] for row in assessment["standards"])
            assessment["settled_under_all_declared_alternatives"] = len(assessment["supported_statuses"]) == 1 and assessment["conflicts"] == 0
            if incident["id"] == "menezes" and basis == "later_record":
                check(assessment["supported_statuses"] == ["forbidden"], "absent threat permitted nonconsensual irreversible force")
            summary["incidents"].append(assessment)
