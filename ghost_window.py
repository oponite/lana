import csv
import json

with open("ohlcv.csv", newline="") as f:
    rows = list(csv.DictReader(f))

bars = []

for row in rows[-32:]:
    bars.append({
        "timestamp": row["timestamp"],
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row["volume"]),
    })

print(json.dumps({
    "symbol": "QQQ",
    "bars": bars
}))
