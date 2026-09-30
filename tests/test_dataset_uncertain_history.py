#!/usr/bin/env python3
"""Exercise captured source updates in separate CLI processes."""
import pathlib
import subprocess
import sys
import tempfile

cli, source = sys.argv[1:]
with tempfile.TemporaryDirectory(prefix="lana-uncertain-history-") as directory:
    store = str(pathlib.Path(directory) / "store")
    for mode, error in [("init", None), ("retry", None), ("extend", None),
                        ("independent", "LANA_ERR_UNSUPPORTED_OPERATION"),
                        ("live", "LANA_ERR_UNSUPPORTED_VALUE"),
                        ("correct", None), ("delete", None)]:
        result = subprocess.run([cli, "run", source, "--", store, mode], capture_output=True, text=True)
        output = result.stdout + result.stderr
        if error:
            assert result.returncode and error in output, (mode, output)
        else:
            assert result.returncode == 0 and "DATASET_UNCERTAIN_HISTORY_PASS" in output, (mode, output)
print("DATASET_UNCERTAIN_HISTORY_PASS")
