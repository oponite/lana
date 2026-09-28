"""Finite is-ought study. A green test run is not a philosophical verdict."""
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STATUSES = ("required", "optional", "forbidden")
WORLDS = tuple(itertools.product((False, True), repeat=2))
# Literal tables, independently specified in world order 00, 01, 10, 11.
RULES = {
    "sharing": {"first": ("required", "optional", "required", "optional"),
                "second": ("optional", "optional", "required", "optional")},
    "promise": {"first": ("optional", "optional", "required", "optional"),
                "second": ("optional",) * 4},
    "harm": {"first": ("optional", "optional", "required", "required"),
             "second": ("optional", "optional", "required", "optional")},
}


def admissible_models(constraints):
    """Constraints are explicit (world, status) premises, never inferred ethics."""
    return [row for row in itertools.product(STATUSES, repeat=4)
            if all(row[world] == status for world, status in constraints)]


def obligation_support(models, world):
    if not models:
        raise ValueError("empty admissible model set")
    return sorted({row[world] for row in models})


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: test_is_ought.py LANA ARTIFACT_DIRECTORY")
    lana = Path(sys.argv[1]).resolve()
    output = Path(sys.argv[2]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    compiler = Path(os.environ.get("LANA_COMPILER_LABC", lana.parent / "lana-compiler.labc")).resolve()
    env = dict(os.environ, LANA_COMPILER_LABC=str(compiler), LANA_STDLIB_DIR=str(ROOT / "stdlib"))
    fixture = ROOT / "tests/regression/is_ought.lana"
    bytecode = output / "is-ought.labc"
    records = []
    reference = {}
    summary = {"engineering_status": "running", "philosophical_status": "not_established",
               "scope": "finite study; supplied semantics and rules", "cases": 0, "capabilities": {}}

    def command(args):
        try:
            result = subprocess.run([str(arg) for arg in args], cwd=ROOT, env=env,
                                    capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired as error:
            records.append({"command": [str(arg) for arg in args], "error": str(error)})
            raise
        records.append({"command": [str(arg) for arg in args], "exit": result.returncode,
                        "stdout": result.stdout, "stderr": result.stderr})
        return result

    def check(condition, description):
        if not condition:
            raise AssertionError(description)

    def case(mode, scenario="sharing", world=0, expected=None, form=None, error=None,
             error_message=None, probe=False, **changes):
        a, b = WORLDS[world]
        data = dict(scenario=scenario, a=a, b=b, not_a=not a, agents=["Alice", "Bob"],
                    left="first", right="second", resolve=True, add_forbid=False,
                    weight=0.99, tables=[], model_assumptions=[], label="Alice ought to share")
        data.update(changes)
        if mode in ("models", "joint", "representation"):
            # Do not even supply irrelevant control rules to the facts-only model test.
            for name in ("left", "right", "weight", "add_forbid"):
                data.pop(name)
        result = command([lana, "run", bytecode, "--seed", "7", "--", mode, json.dumps(data)])
        record = records[-1]
        record.update(mode=mode, scenario=scenario, world=world, input=data,
                      expected_support=expected, expected_error=error)
        summary["cases"] += 1
        if probe:
            known_lowering_failure = (result.returncode != 0 and "LANA_ERR_TYPE" in result.stderr
                                      and "operation SAMPLE_STATE_DIST" in result.stderr)
            check(result.returncode == 0 or known_lowering_failure, f"unexpected capability failure: {record}")
            summary["capabilities"][mode] = "specialized_opcode_type_error" if known_lowering_failure else "available"
            record["capability_status"] = summary["capabilities"][mode]
            if known_lowering_failure:
                return []
        if error:
            check(result.returncode != 0 and error in result.stderr,
                  f"{mode}/{scenario}/{world}: expected {error}; {result.stderr}")
            if error_message:
                check(error_message in result.stderr, f"wrong rejection: {record}")
        else:
            check(result.returncode == 0, f"{mode}/{scenario}/{world}: {result.stderr}")
        rows = [json.loads(line) for line in result.stdout.splitlines()]
        if form:
            check(rows and rows[0]["snapshot"]["form"] == form, f"wrong form: {record}")
        if rows and "snapshot" in rows[0]:
            snapshot = rows[0]["snapshot"]
            record["observed_form"] = snapshot["form"]
            check(rows[0]["context"]["advisory"] is True, "context must remain advisory")
            check(rows[0]["automatic_dependency_completeness"] == "not_certified", "unsupported proof claim")
            declared = rows[0]["context"]["assumptions"]
            annotation = rows[0]["annotation_snapshot"]
            check(len(annotation["assumptions"]) == (1 if declared else 0), "declared annotation missing")
            if mode == "models":
                check(declared == data["model_assumptions"], "model assumptions were changed")
            if expected is not None and "support" in snapshot:
                actual = [row.get("value", row.get("assignment", {}).get("duty"))
                          for row in snapshot["support"]]
                check(sorted(set(actual)) == sorted(expected), f"wrong support: {record}")
                if form == "distribution":
                    wanted_weights = {}
                    for rule, weight in ((data["left"], data["weight"]), (data["right"], 1 - data["weight"])):
                        status = RULES[scenario][rule][world]
                        wanted_weights[status] = wanted_weights.get(status, 0) + weight
                    check(all(abs(row["weight"] - wanted_weights[row["value"]]) < 1e-12
                              for row in snapshot["support"]), "distribution changed the supplied weights")
            if expected is not None and len(expected) == 1 and data["resolve"] and not error:
                value = rows[-1]["resolved"]
                check(value == expected[0], f"wrong resolved value: {record}")
            if form == "paths":
                check(snapshot["remaining_alternatives"] == 2, "disagreement lost a path")
                record["support_visibility"] = "path values not exposed by inspector; exact membership and resolution checked"
        return rows

    try:
        version = command([lana, "version"])
        check(version.returncode == 0, version.stderr)
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
        diff = subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=ROOT)
        untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=ROOT)
        summary["provenance"] = {
            "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_status": status, "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "untracked_sha256": {name: digest(ROOT / name) for name in untracked.decode().split("\0") if name},
            "lana": str(lana), "lana_sha256": digest(lana), "version": version.stdout.strip(),
            "compiler": str(compiler), "compiler_sha256": digest(compiler),
            "fixture_sha256": digest(fixture), "driver_sha256": digest(__file__),
        }
        compiled = command([lana, "compile", fixture, "-o", bytecode])
        check(compiled.returncode == 0, compiled.stderr)
        summary["provenance"]["fixture_bytecode_sha256"] = digest(bytecode)

        for scenario, rules in RULES.items():
            models = admissible_models([])
            check(len(models) == 81 and len(set(models)) == 81, "enumeration is incomplete")
            # Conflicting premises admit no models: reject vacuous entailment.
            impossible = admissible_models([(0, "required"), (0, "forbidden")])
            check(impossible == [], "contradictory constraints accepted")
            try:
                obligation_support(impossible, 0)
            except ValueError:
                pass
            else:
                raise AssertionError("empty models treated as agreement")
            reference[scenario] = {"worlds": WORLDS, "models": models, "rules": rules, "witnesses": []}
            for world in range(4):
                support = obligation_support(models, world)
                check(support == sorted(STATUSES), "baseline unexpectedly fixes a duty")
                witnesses = {status: next(row for row in models if row[world] == status) for status in STATUSES}
                reference[scenario]["witnesses"].append({"world": world, "same_facts": WORLDS[world],
                                                        "assignments": witnesses})
                case("models", scenario, world, support, "possibility", "LANA_ERR_UNRESOLVED_VALUE", tables=models)
                case("models", scenario, world, ["optional", "required"], "possibility",
                     "LANA_ERR_UNRESOLVED_VALUE", tables=[witnesses["required"], witnesses["optional"]],
                     model_assumptions=["two witnesses selected from unconstrained baseline"])
                for rule, table in {**rules, "forbid": ("forbidden",) * 4}.items():
                    case("rule", scenario, world, [table[world]], "definite", left=rule)
                restricted = admissible_models(list(enumerate(rules["first"])))
                check(restricted == [rules["first"]], "restriction did not encode exactly the supplied rule")
                case("models", scenario, world, [rules["first"][world]], "possibility", tables=restricted,
                     model_assumptions=[scenario + ":first truth table supplied as admissibility constraint"])
                # Removing that constraint restores the baseline; no premise-free moral inference.
                check(obligation_support(admissible_models([]), world) == support, "ablation failed")
                expected = sorted({rules["first"][world], rules["second"][world]})
                failure = "LANA_ERR_UNRESOLVED_VALUE" if len(expected) > 1 else None
                for mode in ("possibility", "paths", "distribution"):
                    form = "definite" if mode == "paths" and len(expected) == 1 else mode
                    normal = case(mode, scenario, world, expected, form, failure)
                    swapped = case(mode, scenario, world, expected, form, failure,
                                   left="second", right="first", weight=0.01)
                    renamed = case(mode, scenario, world, expected, form, failure, agents=["Bob", "Alice"])
                    check(normal[0]["snapshot"]["remaining_alternatives"] == swapped[0]["snapshot"]["remaining_alternatives"] ==
                          renamed[0]["snapshot"]["remaining_alternatives"], "renaming/reordering changed alternatives")
                    if mode == "distribution":
                        reverse = case(mode, scenario, world, expected, form, failure, weight=0.01)
                        weights = {r["value"]: r["weight"] for r in reverse[0]["snapshot"]["support"]}
                        if len(expected) > 1:
                            check(abs(weights[rules["first"][world]] - 0.01) < 1e-12, "weights were ignored")
                expanded = sorted(set(expected) | {"forbidden"})
                case("possibility", scenario, world, expanded, "possibility", "LANA_ERR_UNRESOLVED_VALUE", add_forbid=True)

        for world in (0, 2):
            expected = ["optional", "required"] if world == 0 else ["required"]
            case("joint", world=world, expected=expected, form="joint",
                 error="LANA_ERR_UNRESOLVED_VALUE" if world == 0 else None)
        sampled = case("sample")[0]
        check(sampled["selected"] in ("required", "optional"), "sample outside support")
        check(sampled["original"]["remaining_alternatives"] == 2, "sampling erased the alternative")
        check("operation" in sampled["metadata"] and "source_dependency" in sampled["metadata"], "sample lineage missing")
        check(sampled == case("sample")[0], "seeded sample replay changed")
        direct_sample = case("sample_direct", probe=True)
        if direct_sample:
            check(direct_sample[0]["selected"] in ("required", "optional"), "direct sample outside support")
        for label in ("Alice ought to share", "Alice ought not share", "Alice has ninety cookies"):
            row = case("representation", label=label)[0]
            check(row["proposition"] == label and row["supplied_value"] is True, "claim construction changed payload")
            check(row["descriptive_uncertainty"]["form"] == "possibility", "uncertain description lost its form")
            check(row["claim"]["form"] == "definite", "claim representation changed")
        case("models", error="LANA_ERR_ASSERTION", error_message="empty admissible model set", tables=[])
        case("models", error="LANA_ERR_ASSERTION", error_message="model must cover four worlds", tables=[["required"]])
        case("models", error="LANA_ERR_ASSERTION", error_message="invalid obligation status", tables=[["certain"] * 4])
        case("rule", error="LANA_ERR_ASSERTION", error_message="inconsistent facts", not_a=False)
        case("rule", error="LANA_ERR_ASSERTION", error_message="unknown rule", left="undeclared moral axiom")
        case("rule", error="LANA_ERR_ASSERTION", error_message="facts must be Boolean", a="true")
        case("rule", scenario="invented", error="LANA_ERR_ASSERTION", error_message="unknown scenario")
        case("invented", error="LANA_ERR_ASSERTION", error_message="unknown study mode")
        case("distribution", world=2, error="LANA_ERR_ASSERTION", error_message="weight must be positive", weight=0)
        case("distribution", world=2, error="LANA_ERR_ASSERTION", error_message="both rule weights must be positive", weight=1)
        from is_ought_protection import run_study
        protection = run_study(lana, output, command, check)
        check(digest(lana) == summary["provenance"]["lana_sha256"], "CLI changed during study")
        check(digest(compiler) == summary["provenance"]["compiler_sha256"], "compiler changed during study")
        summary["protection"] = {key: protection[key] for key in ("status", "evaluated_cases", "counts", "philosophical_status")}
        summary.update(engineering_status="passed", models_per_scenario=81, factual_states=12,
                       findings=["conditional_rules_evaluated", "same_facts_admit_contrary_obligations_in_baseline",
                                 "information_form_does_not_validate_claim", "full_moral_bridge_not_established",
                                 "path_support_not_exposed_by_inspector", "annotations_are_not_dependency_certification"])
    except Exception as error:
        summary.update(engineering_status="failed", error=str(error))
        raise
    finally:
        (output / "commands.json").write_text(json.dumps(records, indent=2) + "\n")
        (output / "models.json").write_text(json.dumps(reference, indent=2) + "\n")
        (output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"IS_OUGHT_STUDY_PASS: {summary['cases']} control cases; "
          f"{summary['protection']['evaluated_cases']} protection cases; moral bridge not established")


if __name__ == "__main__":
    main()
