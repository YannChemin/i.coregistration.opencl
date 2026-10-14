# SPDX-License-Identifier: Unlicense
"""Run the robust estimation check program (tests/robust_check.c).

It needs no OpenCL device, but must be built first with
``make MODULE_TOPDIR=... robust_check``; the test is skipped otherwise.
"""

import subprocess
from pathlib import Path

import pytest


def test_robust_check():
    """Models, minimum cut and estimators on synthetic matches."""
    found = sorted(Path(__file__).parent.parent.glob("OBJ.*/robust_check"))
    if not found:
        pytest.skip("build tests/robust_check.c with 'make robust_check'")
    p = subprocess.run([str(found[0])], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout[-3000:]
    assert "0 failure(s)" in p.stdout
