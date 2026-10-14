# SPDX-License-Identifier: Unlicense
"""Tests for i.coregistration.opencl."""

import statistics

import grass.script as gs
import pytest
from conftest import SHIFT_X, SHIFT_Y
from grass.tools import ToolError, Tools


def test_group_must_hold_three_bands(session, tmp_path):
    """A group with other than three raster maps is rejected up front."""
    tools = Tools(session=session)
    dummy = tmp_path / "dummy.pth"
    dummy.write_bytes(b"not used")

    with pytest.raises(ToolError, match="exactly 3"):
        tools.i_coregistration_opencl(
            group="two", reference="ref", checkpoint=str(dummy), points="p", flags="n"
        )


def test_dry_run_needs_points(session, tmp_path):
    """With -n and no points= nothing would be written."""
    tools = Tools(session=session)
    dummy = tmp_path / "dummy.pth"
    dummy.write_bytes(b"not used")

    with pytest.raises(ToolError, match="points="):
        tools.i_coregistration_opencl(
            group="img", reference="ref", checkpoint=str(dummy), flags="n"
        )


def test_missing_checkpoint_offline(session, tmp_path):
    """A missing checkpoint is not downloaded with -o."""
    tools = Tools(session=session)

    with pytest.raises(ToolError, match="not found"):
        tools.i_coregistration_opencl(
            group="img", reference="ref", weights_dir=str(tmp_path), flags="o"
        )


def test_rejects_file_that_is_not_a_checkpoint(session, tmp_path):
    """A file that is neither a zip nor a legacy PyTorch checkpoint fails
    before any matching."""
    tools = Tools(session=session)
    bad = tmp_path / "bad.pth"
    bad.write_bytes(b"\x00" * 64)

    with pytest.raises(ToolError, match="checkpoint"):
        tools.i_coregistration_opencl(
            group="img", reference="ref", checkpoint=str(bad), points="p", flags="n"
        )


def test_recovers_known_shift(session, checkpoint, platform):
    """On a texture shifted by (SHIFT_X, SHIFT_Y) cells, the control points
    give that shift, to a fraction of a cell with -s."""
    tools = Tools(session=session)
    tools.i_coregistration_opencl(
        group="img",
        reference="ref",
        checkpoint=checkpoint,
        points="gcp",
        tile_size=256,
        platform=platform,
        flags="s",
    )
    rows = gs.vector_db_select("gcp", columns="dx,dy,refined", env=session.env)["values"]
    assert len(rows) >= 20
    dx = statistics.median(float(r[0]) for r in rows.values())
    dy = statistics.median(float(r[1]) for r in rows.values())
    # The image shows at (c, r) what the reference shows at (c + SHIFT_X,
    # r + SHIFT_Y): east +SHIFT_X, north -SHIFT_Y in map units (1 m cells).
    assert dx == pytest.approx(SHIFT_X, abs=0.1)
    assert dy == pytest.approx(-SHIFT_Y, abs=0.1)

    # The group's control points, for i.rectify.
    found = gs.parse_command(
        "g.findfile", element="group/img", file="POINTS", format="shell", env=session.env
    )
    assert found["name"].strip("'") == "POINTS"
