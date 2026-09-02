import json

INPUT = "train_generated.jsonl"

TRAIN_OUT = "train.jsonl"
VAL_OUT = "validation.jsonl"
TEST_OUT = "test.jsonl"

with open(INPUT) as f:
    rows = [json.loads(line) for line in f if line.strip()]

total = len(rows)

train_end = int(total * 0.80)
val_end = int(total * 0.90)

train = rows[:train_end]
validation = rows[train_end:val_end]
test = rows[val_end:]

with open(TRAIN_OUT, "w") as f:
    for row in train:
        f.write(json.dumps(row) + "\n")

with open(VAL_OUT, "w") as f:
    for row in validation:
        f.write(json.dumps(row) + "\n")

with open(TEST_OUT, "w") as f:
    for row in test:
        f.write(json.dumps(row) + "\n")

print(f"Total: {total}")
print(f"Train: {len(train)}")
print(f"Validation: {len(validation)}")
print(f"Test: {len(test)}")
