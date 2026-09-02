import csv
import json
import os

from lana_integrations.bridge import BridgeRunner


WINDOW = 32

SYSTEM_PROMPT = (
    "You are lana-ghost, a specialist model for analyzing OHLCV time-series data. "
    "Identify statistically and structurally meaningful patterns in price, volume, "
    "volatility, momentum, and market regime. "
    "Do not invent information not supported by the supplied data."
)


# Load OHLCV
with open("ohlcv.csv", newline="") as f:
    rows = list(csv.DictReader(f))


# Create one bridge runner and reuse it
runner = BridgeRunner(
    os.environ.get("LANA_EXECUTABLE", "./build/lana")
)


examples = []

total_windows = len(rows) - WINDOW + 1

for start in range(total_windows):

    window = rows[start:start + WINDOW]

    bars = []

    for row in window:
        bars.append({
            "timestamp": row["timestamp"],
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row["volume"]),
        })

    request = {
        "symbol": "QQQ",
        "bars": bars,
    }


    # Send the OHLCV window through Lana
    envelope = runner.run(
        "ghost_pattern.lana",
        request,
    )


    if not envelope["ok"]:
        raise RuntimeError(
            "Lana failed:\n"
            + envelope.get("stderr", "")
        )


    label = envelope["result"]


    # Convert the OHLCV window into text for model training
    csv_text = (
        "timestamp,open,high,low,close,volume\n"
        + "\n".join(
            f'{bar["timestamp"]},'
            f'{bar["open"]},'
            f'{bar["high"]},'
            f'{bar["low"]},'
            f'{bar["close"]},'
            f'{bar["volume"]}'
            for bar in bars
        )
    )


    example = {
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": (
                    "Analyze this OHLCV sequence and identify meaningful patterns:\n\n"
                    + csv_text
                ),
            },
            {
                "role": "assistant",
                "content": json.dumps(
                    label,
                    separators=(",", ":"),
                ),
            },
        ]
    }


    examples.append(example)

    if (start + 1) % 100 == 0:
        print(
            f"Processed {start + 1}/{total_windows} windows"
        )


# Write Hugging Face training dataset
with open("train_generated.jsonl", "w") as f:
    for example in examples:
        f.write(
            json.dumps(example)
            + "\n"
        )


print()
print(f"Generated {len(examples)} training examples")
print("Saved to train_generated.jsonl")
