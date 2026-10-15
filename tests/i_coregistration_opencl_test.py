# SPDX-License-Identifier: Unlicense
"""Tests for i.coregistration.opencl."""

import math
import statistics
from pathlib import Path

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
            matcher="disk",
            group="two", reference="ref", checkpoint=str(dummy), points="p", flags="n"
        )


def test_dry_run_needs_points(session, tmp_path):
    """With -n and neither points= nor dense= nothing would be written."""
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


def test_matchanything_roma_not_downloaded(session, tmp_path):
    """MatchAnything's RoMa weights are never downloaded: the error says
    where to get them."""
    tools = Tools(session=session)

    with pytest.raises(ToolError, match="not downloaded automatically"):
        tools.i_coregistration_opencl(
            group="img",
            reference="ref",
            matcher="matchanything_roma",
            weights_dir=str(tmp_path),
            points="p",
            flags="n",
        )
    assert not any(tmp_path.iterdir())


def test_max_sd_needs_roma_v2ref(session):
    """max_sd= only applies to a matcher that predicts a standard
    deviation."""
    tools = Tools(session=session)

    with pytest.raises(ToolError, match="max_sd="):
        tools.i_coregistration_opencl(
            group="img", reference="ref", matcher="roma", max_sd=1, points="p",
            flags="n",
        )


def test_rejects_file_that_is_not_a_checkpoint(session, tmp_path):
    """A file that is neither a zip nor a legacy PyTorch checkpoint fails
    before any matching."""
    tools = Tools(session=session)
    bad = tmp_path / "bad.pth"
    bad.write_bytes(b"\x00" * 64)

    with pytest.raises(ToolError, match="checkpoint"):
        tools.i_coregistration_opencl(
            matcher="disk",
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
    """Efficient LoFTR gives the known shift to a fraction of a cell,
    refined (the default) or from its own fine matching alone (-r)."""
    tools = Tools(session=session)
    points = f"gcp_eloftr_{flags or 'refined'}"
    tools.i_coregistration_opencl(
        group="img",
        reference="ref",
        matcher="eloftr",
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


def test_roma_recovers_known_shift(session, roma_dir, platform):
    """RoMa (the default matcher): its dense warp, sampled where most
    certain, gives the known shift to a fraction of a cell, never
    refined by correlation; so do its dense rasters, blended over 2 x 2
    tiles."""
    tools = Tools(session=session)
    tools.i_coregistration_opencl(
        group="img",
        reference="ref",
        matcher="roma",
        weights_dir=roma_dir,
        points="gcp_roma",
        dense="roma",
        tile_size=160,
        platform=platform,
    )
    rows = gs.vector_db_select("gcp_roma", columns="dx,dy,refined", env=session.env)[
        "values"
    ]
    # RoMa alone predicts no standard deviation.
    sds = gs.vector_db_select("gcp_roma", columns="sd", env=session.env)["values"]
    assert all(r[0] == "" for r in sds.values())
    assert not gs.find_file("roma_sd", element="cell", env=session.env)["name"]
    assert len(rows) >= 100
    assert not any(int(r[2]) for r in rows.values())
    dx = [float(r[0]) for r in rows.values()]
    dy = [float(r[1]) for r in rows.values()]
    assert statistics.median(dx) == pytest.approx(SHIFT_X, abs=0.1)
    assert statistics.median(dy) == pytest.approx(-SHIFT_Y, abs=0.1)
    close = sum(math.hypot(x - SHIFT_X, y + SHIFT_Y) < 0.5 for x, y in zip(dx, dy))
    assert close >= 0.9 * len(dx)

    # Dense rasters: east +SHIFT_X, north -SHIFT_Y where certain (about 40 %
    # of this texture's cells; 28 % with one 256-cell tile).
    tools.r_mapcalc(expression="roma_sure = if(roma_certainty >= 0.5, 1, null())")
    sure = gs.parse_command("r.univar", map="roma_sure", flags="g", env=session.env)
    region = gs.region(env=session.env)
    assert int(sure["n"]) >= 0.3 * region["cells"]
    for name, shift in (("dx", SHIFT_X), ("dy", -SHIFT_Y)):
        tools.r_mapcalc(
            expression=f"roma_err_{name} = if(roma_sure, abs(roma_{name} - {shift}))"
        )
        err = gs.parse_command(
            "r.univar", map=f"roma_err_{name}", flags="ge", env=session.env
        )
        assert float(err["median"]) < 0.1
        assert float(err["percentile_90"]) < 0.25


def test_matchanything_roma_recovers_known_shift(
    session, matchanything_roma_weights, roma_dir, platform
):
    """RoMa with MatchAnything's weights (inputs not normalized) gives the
    known shift to a fraction of a cell, never refined."""
    tools = Tools(env={**session.env, "ROMA_MODELS_DIR": roma_dir})
    tools.i_coregistration_opencl(
        group="img",
        reference="ref",
        matcher="matchanything_roma",
        weights_dir=str(Path(matchanything_roma_weights).parent),
        points="gcp_ma_roma",
        tile_size=256,
        platform=platform,
    )
    rows = gs.vector_db_select(
        "gcp_ma_roma", columns="dx,dy,refined", env=session.env
    )["values"]
    assert len(rows) >= 100
    assert not any(int(r[2]) for r in rows.values())
    dx = [float(r[0]) for r in rows.values()]
    dy = [float(r[1]) for r in rows.values()]
    # Less accurate than RoMa's own weights (0.08-0.13 cell median on the
    # basemap synthetic pairs; 0.15 cell north on this smoothed noise),
    # with the authors' code agreeing within 5e-4 cell.
    assert statistics.median(dx) == pytest.approx(SHIFT_X, abs=0.25)
    assert statistics.median(dy) == pytest.approx(-SHIFT_Y, abs=0.25)


def test_roma_v2ref_offline(session, roma_dir, tmp_path):
    """Missing RoMa v2 refiners are not downloaded with -o."""
    tools = Tools(
        env={**session.env, "ROMA_MODELS_DIR": roma_dir, "ROMAV2_MODELS_DIR": str(tmp_path)}
    )

    with pytest.raises(ToolError, match="not found"):
        tools.i_coregistration_opencl(
            group="img", reference="ref", matcher="roma_v2ref", flags="o"
        )
    assert not any(tmp_path.iterdir())


def test_roma_v2ref_recovers_known_shift(session, roma_dir, romav2_dir, platform):
    """RoMa down to 1/8 then RoMa v2's refiners: the known shift to a
    fraction of a cell, in the control points and the dense rasters."""
    tools = Tools(
        env={**session.env, "ROMA_MODELS_DIR": roma_dir, "ROMAV2_MODELS_DIR": romav2_dir}
    )
    tools.i_coregistration_opencl(
        group="img",
        reference="ref",
        matcher="roma_v2ref",
        points="gcp_v2ref",
        dense="v2ref",
        tile_size=160,
        platform=platform,
    )
    rows = gs.vector_db_select("gcp_v2ref", columns="dx,dy,refined", env=session.env)[
        "values"
    ]
    assert len(rows) >= 100
    assert not any(int(r[2]) for r in rows.values())
    dx = [float(r[0]) for r in rows.values()]
    dy = [float(r[1]) for r in rows.values()]
    assert statistics.median(dx) == pytest.approx(SHIFT_X, abs=0.1)
    assert statistics.median(dy) == pytest.approx(-SHIFT_Y, abs=0.1)
    # Predicted standard deviations: given, positive, a fraction of a cell
    # where certain.
    sds = gs.vector_db_select("gcp_v2ref", columns="sd", env=session.env)["values"]
    sds = [float(r[0]) for r in sds.values()]
    assert all(v > 0 for v in sds)
    assert statistics.median(sds) < 0.5
    sdr = gs.parse_command("r.univar", map="v2ref_sd", flags="g", env=session.env)
    assert int(sdr["n"]) > 0 and float(sdr["min"]) > 0
    # max_sd= drops the matches above it.
    cut = statistics.median(sds)
    tools.i_coregistration_opencl(
        group="img",
        reference="ref",
        matcher="roma_v2ref",
        points="gcp_v2ref_cut",
        max_sd=cut,
        tile_size=160,
        platform=platform,
        flags="n",
    )
    kept = gs.vector_db_select("gcp_v2ref_cut", columns="sd", env=session.env)[
        "values"
    ]
    assert 0 < len(kept) < len(sds)
    assert all(float(r[0]) <= cut + 1e-4 for r in kept.values())
    tools.r_mapcalc(
        expression=f"v2ref_err = if(v2ref_certainty >= 0.5, abs(v2ref_dx - {SHIFT_X}))"
    )
    err = gs.parse_command("r.univar", map="v2ref_err", flags="ge", env=session.env)
    assert int(err["n"]) >= 0.3 * gs.region(env=session.env)["cells"]
    assert float(err["median"]) < 0.1
