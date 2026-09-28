"""A success marker must never hide a failed/crashed test process."""
import subprocess
from run import check_result
from workflows import run_process
import sys


def rejected(case, code, output):
    try:
        check_result(case, subprocess.CompletedProcess([], code, output))
    except AssertionError:
        return
    raise AssertionError((case, code, output))


rejected({'contains':['PASS']}, 1, 'PASS\nlater operation failed')
rejected({'contains':['PASS','SECOND_PASS']}, 0, 'PASS')
rejected({'expect_failure':True}, -11, '')
rejected({'expect_failure':True, 'contains':['EXPECTED_ERROR']}, 0, 'EXPECTED_ERROR')
rejected({'expect_failure':True, 'source_span':'source.lana'}, 1, 'source.lana:1')
check_result({'expect_failure':True, 'contains':['EXPECTED_ERROR'], 'source_span':'source.lana'},
             subprocess.CompletedProcess([], 1, 'source.lana:1:2-1:3 EXPECTED_ERROR'))
try:
    run_process([sys.executable, '-c', 'import time; time.sleep(30)'], timeout=0.02)
except subprocess.TimeoutExpired:
    pass
else:
    raise AssertionError('timeout was ignored')
print('WORKFLOW_RUNNER_PASS')
