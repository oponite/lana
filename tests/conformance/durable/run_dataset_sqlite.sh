#!/usr/bin/env bash
set -euo pipefail
root="$(mktemp -d "${TMPDIR:-/tmp}/lana-sqlite.XXXXXX")"
trap 'rm -rf "$root"' EXIT
expected="$(python3 - "$root/data.db" <<'PY'
import hashlib
import json
import sqlite3
import struct
import sys

path = sys.argv[1]
db = sqlite3.connect(path)
db.execute('CREATE TABLE data (id TEXT, n INTEGER, note TEXT, info TEXT, region TEXT)')
info = '{"tag":"definite","value":{"tag":"string","value":"yes"}}'
db.executemany('INSERT INTO data VALUES (?, ?, ?, ?, ?)', [
    ('a', 1, None, info, 'east'), ('b', 2, 'ready', info, 'east'),
    ('a', 3, None, info, 'duplicate'), ('a', 4, None, info, 'duplicate'),
    ('large', 9007199254740993, None, info, 'large'),
])
db.commit()
db.close()
def tagged(value):
    if value is None: return {'tag': 'null'}
    if isinstance(value, int): return {'tag': 'number', 'bits': '%016x' % struct.unpack('>Q', struct.pack('>d', float(value)))[0]}
    return {'tag': 'string', 'value': value}
payload = {
    'columns': ['id', 'n', 'note', 'info'],
    'parameters': [tagged('east')],
    'rows': [[tagged(id), tagged(n), tagged(note), json.loads(info)] for id, n, note in [('a', 1, None), ('b', 2, 'ready')]],
    'schema': [{'name': name, 'kind': kind} for name, kind in [('id', 'string'), ('n', 'number'), ('note', 'nullable_string'), ('info', 'information_json')]],
}
print(hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest())
PY
)"
sql='SELECT id, n, note, info FROM data WHERE region = ? ORDER BY id'
"$1" run "$2" -- "$root/data.db" "$sql" "$expected"
"$1" run "$2" -- "$root/data.db" 'SELECT lower(id) AS id, n, note, info FROM data WHERE region = ? ORDER BY id' "$expected"
python3 - "$root/data.db" "$1" "$2" "$sql" "$expected" <<'PY'
import sqlite3
import subprocess
import sys

path, cli, source, sql, expected = sys.argv[1:]
writer = sqlite3.connect(path)
writer.execute('PRAGMA journal_mode=WAL')
writer.execute('BEGIN IMMEDIATE')
writer.execute("UPDATE data SET note = 'uncommitted' WHERE id = 'b' AND region = 'east'")
read = subprocess.run([cli, 'run', source, '--', path, sql, expected], capture_output=True, text=True)
writer.rollback()
# The concurrent WAL read is complete. Remove WAL recovery from the later
# malformed-input cases: macOS read-only SQLite cannot recreate missing sidecars.
writer.execute('PRAGMA journal_mode=DELETE')
writer.close()
if read.returncode or 'DATASET_SQLITE_PASS' not in read.stdout:
    raise SystemExit(read.stdout + read.stderr)
PY
for bad in \
  'SELECT id, n, note, info FROM data WHERE region = "duplicate" ORDER BY id' \
  'SELECT id, n, note, info FROM data WHERE region = "large" ORDER BY id' \
  'SELECT id, n, note, info FROM data WHERE region = ?; DELETE FROM data' \
  'SELECT load_extension(id) AS id, n, note, info FROM data WHERE region = ? ORDER BY id' \
  'PRAGMA table_info(data)'; do
    if "$1" run "$2" -- "$root/data.db" "$bad" "$expected" > "$root/rejected.out" 2>&1; then
      echo "unsafe or invalid SQLite statement succeeded: $bad" >&2
      exit 1
    fi
    if grep -q DATASET_SQLITE_PASS "$root/rejected.out"; then
      echo "failed SQLite statement exposed partial result" >&2
      exit 1
    fi
    if ! grep -Eq 'LANA_ERR_(SCHEMA|UNSUPPORTED_OPERATION)' "$root/rejected.out"; then
      cat "$root/rejected.out" >&2
      echo "SQLite boundary did not reject the input itself: $bad" >&2
      exit 1
    fi
done
