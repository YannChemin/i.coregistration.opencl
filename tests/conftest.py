# SPDX-License-Identifier: Unlicense
"""pytest fixtures for i.coregistration.opencl.

The matching tests need the weights and an OpenCL device, neither of
which can be assumed on a test machine. They are skipped unless the
weights are given: the DISK checkpoint in
I_COREGISTRATION_OPENCL_CHECKPOINT, the Efficient LoFTR model.safetensors
in I_COREGISTRATION_OPENCL_ELOFTR, MatchAnything's in
I_COREGISTRATION_OPENCL_MATCHANYTHING, the folder of RoMa's
roma_outdoor.pth and dinov2_vitl14_pretrain.pth in
I_COREGISTRATION_OPENCL_ROMA.
I_COREGISTRATION_OPENCL_PLATFORM optionally selects the OpenCL platform
(for Mesa rusticl, also export RUSTICL_ENABLE=radeonsi or the matching
driver). I_COREGISTRATION_OPENCL_BINDIR, when set, is put first in the
session's PATH, to test a build that is not installed (the session puts
the installation's bin first otherwise).
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
# A moved object: rows and columns (from the top-left of the extended
# region) of the patch's original place, and its displacement east.
PATCH = (120, 100)
MOVE = 12


@pytest.fixture(scope="module")
def session(tmp_path_factory):
    """Session with a textured reference group <ref>, the same texture
    shifted by (SHIFT_X, SHIFT_Y) cells in group <img>, group <moved>:
    <img> with a 64 x 64 patch moved MOVE cells east (an object that
    moved between dates), and a 2-band group <two>."""
    project = tmp_path_factory.mktemp("i_coregistration_opencl") / "project"
    gs.create_project(project)
    with gs.setup.init(project, env=os.environ.copy()) as session:
        bindir = os.environ.get("I_COREGISTRATION_OPENCL_BINDIR")
        if bindir:
            session.env["PATH"] = bindir + os.pathsep + session.env["PATH"]
        tools = Tools(session=session)
        tools.g_region(n=SIZE + 32, s=-32, e=SIZE + 32, w=-32, res=1)
        # Blobs and edges: smoothed random noise in three bands, larger than
        # the test region so that the shifted copy has no NULL border.
        for b in (1, 2, 3):
            tools.r_surf_random(output=f"noise{b}", min=0, max=255, seed=b)
            tools.r_neighbors(
                input=f"noise{b}", output=f"smooth{b}", size=7, method="average"
            )
            tools.r_mapcalc(
                expression=f"ref.{b} = max(0, min(255, round(smooth{b} * 4 - 384)))"
            )
            tools.r_mapcalc(expression=f"img.{b} = ref.{b}[{SHIFT_Y}, {SHIFT_X}]")
            # The patch shows, MOVE cells east of where it belongs, what
            # <img> shows at its own place.
            tools.r_mapcalc(
                expression=f"moved.{b} = if(row() > {PATCH[0]} && row() <= {PATCH[0] + 64}"
                f" && col() > {PATCH[1] + MOVE} && col() <= {PATCH[1] + MOVE + 64},"
                f" ref.{b}[{SHIFT_Y}, {SHIFT_X - MOVE}], img.{b})"
            )
        tools.g_region(n=SIZE, s=0, e=SIZE, w=0, res=1)
        tools.i_group(group="ref", input="ref.1,ref.2,ref.3")
        tools.i_group(group="img", input="img.1,img.2,img.3")
        tools.i_group(group="moved", input="moved.1,moved.2,moved.3")
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
def eloftr_weights():
    """Path of the Efficient LoFTR weights (model.safetensors), or skip."""
    path = os.environ.get("I_COREGISTRATION_OPENCL_ELOFTR")
    if not path or not Path(path).is_file():
        pytest.skip("set I_COREGISTRATION_OPENCL_ELOFTR to model.safetensors")
    return path


@pytest.fixture
def matchanything_weights():
    """Path of MatchAnything's Efficient LoFTR weights, or skip."""
    path = os.environ.get("I_COREGISTRATION_OPENCL_MATCHANYTHING")
    if not path or not Path(path).is_file():
        pytest.skip("set I_COREGISTRATION_OPENCL_MATCHANYTHING to model.safetensors")
    return path


@pytest.fixture
def roma_dir():
    """Folder of RoMa's and DINOv2's weights, or skip."""
    path = os.environ.get("I_COREGISTRATION_OPENCL_ROMA")
    files = ("roma_outdoor.pth", "dinov2_vitl14_pretrain.pth")
    if not path or not all((Path(path) / f).is_file() for f in files):
        pytest.skip("set I_COREGISTRATION_OPENCL_ROMA to the folder of " + " and ".join(files))
    return path


@pytest.fixture
def matchanything_roma_weights():
    """Path of MatchAnything's RoMa weights (matchanything_roma.ckpt), or
    skip."""
    path = os.environ.get("I_COREGISTRATION_OPENCL_MATCHANYTHING_ROMA")
    if not path or not Path(path).is_file():
        pytest.skip(
            "set I_COREGISTRATION_OPENCL_MATCHANYTHING_ROMA to matchanything_roma.ckpt"
        )
    return path


@pytest.fixture
def platform():
    return os.environ.get("I_COREGISTRATION_OPENCL_PLATFORM")
