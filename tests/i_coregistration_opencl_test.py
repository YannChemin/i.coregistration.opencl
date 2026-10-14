# SPDX-License-Identifier: Unlicense
"""Tests for i.coregistration.opencl."""

import math
import statistics

import grass.script as gs
import pytest
from conftest import MOVE, SHIFT_X, SHIFT_Y
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


def test_refinement_flags_contradict(session, tmp_path):
    """-r (no refinement) and the deprecated -s (refinement) together are
    rejected."""
    tools = Tools(session=session)
    dummy = tmp_path / "dummy.pth"
    dummy.write_bytes(b"not used")

    with pytest.raises(ToolError, match="contradict"):
        tools.i_coregistration_opencl(
            group="img", reference="ref", checkpoint=str(dummy), flags="rs"
        )


def test_missing_checkpoint_offline(session, tmp_path):
    """A missing checkpoint is not downloaded with -o."""
    tools = Tools(session=session)

    with pytest.raises(ToolError, match="not found"):
        tools.i_coregistration_opencl(
            group="img", reference="ref", weights_dir=str(tmp_path), flags="o"
        )


def test_matchanything_offline(session, tmp_path):
    """Missing MatchAnything weights are not downloaded with -o."""
    tools = Tools(session=session)

    with pytest.raises(ToolError, match="not found"):
        tools.i_coregistration_opencl(
            group="img",
            reference="ref",
            matcher="matchanything",
            weights_dir=str(tmp_path),
            flags="o",
        )
    assert not any(tmp_path.iterdir())


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
    """On a texture shifted by (SHIFT_X, SHIFT_Y) cells, the DISK control
    points give that shift, to a fraction of a cell once refined."""
    tools = Tools(session=session)
    tools.i_coregistration_opencl(
        group="img",
        reference="ref",
        matcher="disk",
        checkpoint=checkpoint,
        points="gcp",
        tile_size=256,
        platform=platform,
    )
    rows = gs.vector_db_select("gcp", columns="dx,dy,refined", env=session.env)[
        "values"
    ]
    assert len(rows) >= 20
    assert any(int(r[2]) for r in rows.values())
    dx = statistics.median(float(r[0]) for r in rows.values())
    dy = statistics.median(float(r[1]) for r in rows.values())
    # The image shows at (c, r) what the reference shows at (c + SHIFT_X,
    # r + SHIFT_Y): east +SHIFT_X, north -SHIFT_Y in map units (1 m cells).
    assert dx == pytest.approx(SHIFT_X, abs=0.1)
    assert dy == pytest.approx(-SHIFT_Y, abs=0.1)

    # The group's control points, for i.rectify.
    found = gs.parse_command(
        "g.findfile",
        element="group/img",
        file="POINTS",
        format="shell",
        env=session.env,
    )
    assert found["name"].strip("'") == "POINTS"


@pytest.mark.parametrize("flags", ["", "r"])
def test_eloftr_recovers_known_shift(session, eloftr_weights, platform, flags):
    """Efficient LoFTR (the default matcher) gives the known shift to a
    fraction of a cell, refined (the default) or from its own fine
    matching alone (-r)."""
    tools = Tools(session=session)
    points = f"gcp_eloftr_{flags or 'refined'}"
    tools.i_coregistration_opencl(
        group="img",
        reference="ref",
        checkpoint=eloftr_weights,
        points=points,
        tile_size=256,
        platform=platform,
        flags=flags,
    )
    rows = gs.vector_db_select(points, columns="dx,dy,refined", env=session.env)[
        "values"
    ]
    assert len(rows) >= 100
    refined = sum(int(r[2]) for r in rows.values())
    if flags == "r":
        assert refined == 0
    else:
        assert refined >= 0.5 * len(rows)
    dx = [float(r[0]) for r in rows.values()]
    dy = [float(r[1]) for r in rows.values()]
    assert statistics.median(dx) == pytest.approx(SHIFT_X, abs=0.1)
    assert statistics.median(dy) == pytest.approx(-SHIFT_Y, abs=0.1)
    close = sum(
        math.hypot(x - SHIFT_X, y + SHIFT_Y) < 0.5 for x, y in zip(dx, dy)
    )
    assert close >= 0.9 * len(dx)


def test_estimator_rejects_moved_object(session, checkpoint, platform):
    """Matches on a patch that moved MOVE cells pass AdaLAM; the robust
    estimation per tile makes them inactive and keeps the ground."""
    tools = Tools(session=session)
    moved_dx = SHIFT_X - MOVE

    def run(points, **extra):
        tools.i_coregistration_opencl(
            group="moved",
            reference="ref",
            matcher="disk",
            checkpoint=checkpoint,
            points=points,
            tile_size=256,
            platform=platform,
            flags="n",
            **extra,
        )
        rows = gs.vector_db_select(points, columns="dx,dy,inlier", env=session.env)
        return [(float(dx), float(dy), int(a)) for dx, dy, a in rows["values"].values()]

    plain = run("gcp_plain", estimator="none")
    assert any(abs(dx - moved_dx) < 1.5 for dx, _, _ in plain), (
        "the moved patch gives no match: the test cannot show anything"
    )
    assert all(a == 1 for _, _, a in plain)

    # superansac is the default.
    robust = run("gcp_robust")
    active = [(dx, dy) for dx, dy, a in robust if a]
    assert len(active) >= 20
    # Active points lie within the threshold (3 cells by default) of the
    # true shift; the median is exact.
    off = [(dx, dy) for dx, dy in active if math.hypot(dx - SHIFT_X, dy + SHIFT_Y) > 3]
    assert not off, (
        f"{len(off)} of {len(active)} active points off the shift: {off[:10]}"
    )
    assert statistics.median(dx for dx, _ in active) == pytest.approx(SHIFT_X, abs=0.1)
    assert statistics.median(dy for _, dy in active) == pytest.approx(-SHIFT_Y, abs=0.1)
    assert not any(abs(dx - moved_dx) < 1.5 for dx, _ in active)


def test_matchanything_recovers_known_shift(session, matchanything_weights, platform):
    """MatchAnything's weights on the Efficient LoFTR network give the
    known shift to a fraction of a cell."""
    tools = Tools(session=session)
    tools.i_coregistration_opencl(
        group="img",
        reference="ref",
        matcher="matchanything",
        checkpoint=matchanything_weights,
        points="gcp_ma",
        tile_size=256,
        platform=platform,
    )
    rows = gs.vector_db_select("gcp_ma", columns="dx,dy", env=session.env)["values"]
    assert len(rows) >= 100
    dx = [float(r[0]) for r in rows.values()]
    dy = [float(r[1]) for r in rows.values()]
    assert statistics.median(dx) == pytest.approx(SHIFT_X, abs=0.1)
    assert statistics.median(dy) == pytest.approx(-SHIFT_Y, abs=0.1)
