"""Fetch market data and emit the version-1 feature contract for Lana."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


def normalize_ticker(ticker: str) -> str:
    """Return the canonical ticker representation used by the contract."""

    normalized = ticker.strip().upper()
    if not normalized:
        raise ValueError("ticker must not be empty")
    return normalized


def fetch_history(ticker: str) -> pd.DataFrame:
    """Retrieve one year of adjusted daily history for one ticker."""

    symbol = normalize_ticker(ticker)
    try:
        history = yf.Ticker(symbol).history(
            period="1y", auto_adjust=False, actions=False
        )
    except Exception as error:
        raise RuntimeError(f"price history retrieval failed for {symbol}: {error}") from error
    if history is None or history.empty:
        raise ValueError(f"no usable price history returned for {symbol}")
    return history


def fetch_option_expirations(ticker: str) -> tuple[str, ...]:
    """Retrieve the currently listed option expiration dates."""

    symbol = normalize_ticker(ticker)
    try:
        expirations = yf.Ticker(symbol).options
    except Exception as error:
        raise RuntimeError(
            f"option expiration retrieval failed for {symbol}: {error}"
        ) from error
    if expirations is None:
        raise RuntimeError(f"option expiration retrieval returned no result for {symbol}")
    try:
        return tuple(str(expiration) for expiration in expirations)
    except TypeError as error:
        raise RuntimeError(
            f"option expiration retrieval returned invalid data for {symbol}"
        ) from error


def calculate_features(history: pd.DataFrame, max_capital: float) -> dict[str, Any]:
    """Calculate deterministic scalar features from already-retrieved history."""

    required = {"Adj Close", "Volume"}
    missing = sorted(required.difference(history.columns))
    if missing:
        raise ValueError(f"history is missing required columns: {', '.join(missing)}")
    if not math.isfinite(max_capital):
        raise ValueError("max_capital must be finite")
    if max_capital < 0:
        raise ValueError("max_capital must not be negative")

    frame = history.loc[:, ["Adj Close", "Volume"]].copy()
    frame["Adj Close"] = pd.to_numeric(frame["Adj Close"], errors="coerce")
    frame["Volume"] = pd.to_numeric(frame["Volume"], errors="coerce")
    finite = np.isfinite(frame["Adj Close"]) & np.isfinite(frame["Volume"])
    frame = frame.loc[finite]
    frame = frame.loc[(frame["Adj Close"] > 0) & (frame["Volume"] >= 0)]
    if frame.empty:
        raise ValueError("no usable price history returned")
    if len(frame) < 2:
        raise ValueError("at least two usable adjusted closes are required")

    prices = frame["Adj Close"]
    volumes = frame["Volume"]
    daily_returns = prices.pct_change().dropna()
    price = float(prices.iloc[-1])
    share_cost = price * 100.0
    one_year_return = price / float(prices.iloc[0]) - 1.0
    annualized_volatility = float(daily_returns.std(ddof=1) * math.sqrt(252.0))
    drawdowns = prices / prices.cummax() - 1.0
    max_drawdown = float(drawdowns.min())
    average_dollar_volume = float((prices * volumes).mean())

    features: dict[str, Any] = {
        "trading_days": int(len(frame)),
        "price": price,
        "share_cost": share_cost,
        "max_capital": float(max_capital),
        "one_year_return": float(one_year_return),
        "annualized_volatility": annualized_volatility,
        "max_drawdown": max_drawdown,
        "average_dollar_volume": average_dollar_volume,
    }
    _require_finite_numbers(features)
    return features


def build_contract(
    ticker: str,
    max_capital: float,
    history: pd.DataFrame,
    option_expirations: tuple[str, ...],
    observed_at: str | None = None,
) -> dict[str, Any]:
    """Combine calculated features and retrieved option availability."""

    symbol = normalize_ticker(ticker)
    features = calculate_features(history, max_capital)
    contract: dict[str, Any] = {
        "schema": 1,
        "ticker": symbol,
        "observed_at": observed_at or utc_timestamp(),
        "source": "yfinance",
        **features,
        "has_options": len(option_expirations) > 0,
        "option_expiration_count": len(option_expirations),
    }
    _require_finite_numbers(contract)
    return contract


def fetch_contract(ticker: str, max_capital: float) -> dict[str, Any]:
    """Retrieve all external data and build the versioned feature contract."""

    symbol = normalize_ticker(ticker)
    history = fetch_history(symbol)
    expirations = fetch_option_expirations(symbol)
    return build_contract(symbol, max_capital, history, expirations)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _require_finite_numbers(value: Any) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("feature contract contains a non-finite number")
        return
    if isinstance(value, dict):
        for item in value.values():
            _require_finite_numbers(item)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _require_finite_numbers(item)


def write_contract(path: Path, contract: dict[str, Any]) -> None:
    """Write stable, finite JSON for the Lana boundary."""

    _require_finite_numbers(contract)
    path.write_text(
        json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticker")
    parser.add_argument("--max-capital", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_args(argv)
    try:
        contract = fetch_contract(arguments.ticker, arguments.max_capital)
        write_contract(arguments.output, contract)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
