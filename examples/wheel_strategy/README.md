# Wheel candidate screen

This example is a small two-stage research screen:

- Python retrieves yfinance data and calculates observed scalar features.
- Lana applies an ordinary deterministic policy to the versioned JSON file.

A passing result means only **candidate for option-chain analysis**. This is a
research screen, not a profitability model, trade recommendation, personalized
investment advice, or order-execution system.

## Setup

From the repository root:

```bash
cd examples/wheel_strategy
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cd ../..
```

The Lana executable is built at `build/lana` by the repository's Debug build:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build --parallel
```

## Run

Fetch a contract. The output file is generated data and should not be
committed:

```bash
examples/wheel_strategy/.venv/bin/python examples/wheel_strategy/fetch.py AAPL \
  --max-capital 25000 \
  --output examples/wheel_strategy/features.json
```

Run the deterministic Lana policy:

```bash
build/lana run examples/wheel_strategy/decision.lana -- \
  examples/wheel_strategy/features.json
```

The exact example command using the checked-in fixture is:

```bash
build/lana run examples/wheel_strategy/decision.lana -- \
  examples/wheel_strategy/sample_features.json
```

## Contracts

The Python-to-Lana input is a schema-1 JSON object:

```json
{
  "schema": 1,
  "ticker": "AAPL",
  "observed_at": "UTC ISO-8601 timestamp",
  "source": "yfinance",
  "trading_days": 252,
  "price": 200.0,
  "share_cost": 20000.0,
  "max_capital": 25000.0,
  "one_year_return": 0.12,
  "annualized_volatility": 0.28,
  "max_drawdown": -0.17,
  "average_dollar_volume": 1000000000.0,
  "has_options": true,
  "option_expiration_count": 12
}
```

`share_cost` is `price * 100`. Volatility is daily-return sample standard
deviation multiplied by `sqrt(252)`. Prices use adjusted daily closes; dollar
volume is adjusted close multiplied by reported volume.

Lana prints exactly one JSON array:

```json
["AAPL",true,"passes the version-0 wheel candidate screen"]
```

The policy gates, in order, are trading history, listed options, share cost,
liquidity, one-year return, maximum drawdown, and a volatility range of 15% to
60%. It returns on the first failed gate.

## Tests

The tests use fixed in-memory history and synthetic JSON inputs. They do not
make network requests:

```bash
examples/wheel_strategy/.venv/bin/python -m pytest examples/wheel_strategy/tests
```

## Version 1: option-chain checks still missing

Version 0 does not choose or evaluate an option contract. A future version
needs to check:

- specific expiration and strike
- cash obligation based on `strike * 100`
- delta
- bid/ask spread
- open interest and volume
- premium yield
- earnings before expiration
- willingness to own the shares
