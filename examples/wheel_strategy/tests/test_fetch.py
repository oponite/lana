import importlib.util
import json
import math
import subprocess
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[3]
FETCH_PATH = ROOT / "examples" / "wheel_strategy" / "fetch.py"
DECISION_PATH = ROOT / "examples" / "wheel_strategy" / "decision.lana"
LANA_PATH = ROOT / "build" / "lana"


spec = importlib.util.spec_from_file_location("wheel_fetch", FETCH_PATH)
assert spec is not None and spec.loader is not None
fetch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetch)


def history_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Adj Close": [100.0, 102.0, 101.0, 105.0, 103.0],
            "Volume": [1000.0, 1200.0, 1100.0, 1500.0, 1300.0],
        }
    )


def test_feature_calculations_from_fixed_fixture() -> None:
    features = fetch.calculate_features(history_fixture(), 25000.0)

    assert features["trading_days"] == 5
    assert features["price"] == 103.0
    assert features["share_cost"] == 10300.0
    assert features["one_year_return"] == pytest.approx(0.03)
    assert features["max_drawdown"] == pytest.approx(103.0 / 105.0 - 1.0)
    assert features["average_dollar_volume"] == pytest.approx(124980.0)
    assert features["annualized_volatility"] > 0.0


def test_empty_history_fails_clearly() -> None:
    with pytest.raises(ValueError, match="no usable price history"):
        fetch.calculate_features(pd.DataFrame(columns=["Adj Close", "Volume"]), 25000.0)


def test_missing_column_fails_clearly() -> None:
    history = pd.DataFrame({"Adj Close": [100.0, 101.0]})

    with pytest.raises(ValueError, match="missing required columns: Volume"):
        fetch.calculate_features(history, 25000.0)


def test_ticker_is_normalized_in_contract() -> None:
    contract = fetch.build_contract(
        " aapl ",
        25000.0,
        history_fixture(),
        ("2026-01-16",),
        observed_at="2026-01-01T00:00:00Z",
    )

    assert contract["ticker"] == "AAPL"
    assert contract["has_options"] is True
    assert contract["option_expiration_count"] == 1


def test_json_serialization_contains_only_finite_values(tmp_path: Path) -> None:
    contract = fetch.build_contract(
        "AAPL",
        25000.0,
        history_fixture(),
        (),
        observed_at="2026-01-01T00:00:00Z",
    )
    output = tmp_path / "features.json"
    fetch.write_contract(output, contract)
    loaded = json.loads(output.read_text(encoding="utf-8"))

    def assert_finite(value: object) -> None:
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, (int, float)):
            assert math.isfinite(float(value))
        elif isinstance(value, dict):
            for item in value.values():
                assert_finite(item)
        elif isinstance(value, list):
            for item in value:
                assert_finite(item)

    assert_finite(loaded)
    assert json.dumps(loaded, allow_nan=False)


@pytest.mark.parametrize(
    ("name", "changes", "expected"),
    [
        (
            "passing candidate",
            {},
            ["AAPL", True, "passes the version-0 wheel candidate screen"],
        ),
        (
            "insufficient history",
            {"trading_days": 199},
            ["AAPL", False, "needs at least 200 trading days"],
        ),
        ("no options", {"has_options": False}, ["AAPL", False, "has no listed options"]),
        (
            "capital limit",
            {"share_cost": 30000.0},
            ["AAPL", False, "share cost exceeds max capital"],
        ),
        (
            "low liquidity",
            {"average_dollar_volume": 9999999.0},
            ["AAPL", False, "average dollar volume is below 10000000"],
        ),
        (
            "weak return",
            {"one_year_return": -0.21},
            ["AAPL", False, "one-year return is below -20%"],
        ),
        (
            "excessive drawdown",
            {"max_drawdown": -0.36},
            ["AAPL", False, "maximum drawdown is below -35%"],
        ),
        (
            "volatility too low",
            {"annualized_volatility": 0.14},
            ["AAPL", False, "volatility is below 15%"],
        ),
        (
            "volatility too high",
            {"annualized_volatility": 0.61},
            ["AAPL", False, "volatility is above 60%"],
        ),
    ],
)
def test_lana_decision_cases(
    tmp_path: Path, name: str, changes: dict[str, object], expected: list[object]
) -> None:
    if not LANA_PATH.exists():
        pytest.fail(f"Lana executable not found at {LANA_PATH}; build Lana before running this test")
    features = {
        "schema": 1,
        "ticker": "AAPL",
        "observed_at": "2026-01-01T00:00:00Z",
        "source": "fixture",
        "trading_days": 252,
        "price": 200.0,
        "share_cost": 20000.0,
        "max_capital": 25000.0,
        "one_year_return": 0.12,
        "annualized_volatility": 0.28,
        "max_drawdown": -0.17,
        "average_dollar_volume": 1000000000.0,
        "has_options": True,
        "option_expiration_count": 12,
    }
    features.update(changes)
    input_path = tmp_path / f"{name.replace(' ', '_')}.json"
    input_path.write_text(json.dumps(features), encoding="utf-8")

    completed = subprocess.run(
        [str(LANA_PATH), "run", str(DECISION_PATH), "--", str(input_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert json.loads(completed.stdout) == expected
