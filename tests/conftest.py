# SPDX-License-Identifier: Unlicense
"""pytest fixtures for i.coregistration.opencl.

The matching tests need the DISK checkpoint and an OpenCL device, neither
of which can be assumed on a test machine. They are skipped unless the
checkpoint path is given in I_COREGISTRATION_OPENCL_CHECKPOINT.
I_COREGISTRATION_OPENCL_PLATFORM optionally selects the OpenCL platform
(for Mesa rusticl, also export RUSTICL_ENABLE=radeonsi or the matching
driver).
"""

import os
from pathlib import Path

import grass.script as gs
import pytest
from grass.tools import Tools

SIZE = 256
# Shift of the image to co-register against the reference, in cells
# (east, south): the image shows at column c what the reference shows at
# column c + SHIFT_X.
SHIFT_X, SHIFT_Y = 3, 2


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    """Session with a textured reference group <ref>, the same texture
    shifted by (SHIFT_X, SHIFT_Y) cells in group <img>, and a 2-band
    group <two>."""
    project = tmp_path_factory.mktemp("i_coregistration_opencl") / "project"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        tools = Tools(session=session)
        tools.g_region(n=SIZE + 32, s=-32, e=SIZE + 32, w=-32, res=1)
        # Blobs and edges: smoothed random noise in three bands, larger than
        # the test region so that the shifted copy has no NULL border.
        for b in (1, 2, 3):
            tools.r_surf_random(output=f"noise{b}", min=0, max=255, seed=b)
            tools.r_neighbors(input=f"noise{b}", output=f"smooth{b}", size=7, method="average")
            tools.r_mapcalc(expression=f"ref.{b} = max(0, min(255, round(smooth{b} * 4 - 384)))")
            tools.r_mapcalc(
                expression=f"img.{b} = ref.{b}[{SHIFT_Y}, {SHIFT_X}]"
            )
        tools.g_region(n=SIZE, s=0, e=SIZE, w=0, res=1)
        tools.i_group(group="ref", input="ref.1,ref.2,ref.3")
        tools.i_group(group="img", input="img.1,img.2,img.3")
        tools.i_group(group="two", input="ref.1,ref.2")
        yield session


@pytest.fixture
def checkpoint():
    """Path of the DISK checkpoint, or skip."""
    path = os.environ.get("I_COREGISTRATION_OPENCL_CHECKPOINT")
    if not path or not Path(path).is_file():
        pytest.skip("set I_COREGISTRATION_OPENCL_CHECKPOINT to depth-save.pth")
    return path


@pytest.fixture
def platform():
    return os.environ.get("I_COREGISTRATION_OPENCL_PLATFORM")
