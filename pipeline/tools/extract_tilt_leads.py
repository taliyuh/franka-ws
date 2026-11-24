"""Regenerate config/tilt_leads.yaml: the tilt search's geometry per hole, for `pipeline insert`.

Uses rl_manip/peg_board_env/tilt_search.py (the simulation the robot's tilt_slide.h follows; its
`--screen`). Run with that project's interpreter, like extract_holes.py:

    cd <rl_manip checkout>
    JAX_PLATFORMS=cpu PYTHONPATH=. .venv/bin/python <workspace>/pipeline/tools/extract_tilt_leads.py \
        > <workspace>/pipeline/config/tilt_leads.yaml

Per hole, every lead of the shape's default lead kind (tilt_search.DEFAULT_LEADS; the L-shape's two corners), in
the hole frame of holes.yaml (profiles.profile_frame, the simulation's too): the slide direction, the lead point
along and across it from the peg axis (peg_search --peg-radius, --lead-across), and the search sized for the
simulation's hole error disc (--lead-run-up, --slide-length, --line-spacing, --lines, --spiral-pitch,
--spiral-radius). Leads of one `group` are the same lead turned by the shape's symmetry: `pipeline insert` picks
the one whose slide direction in the base frame is nearest insert.slide_dir_base_deg.

The dip region (2026-11-21): the lead positions, from the hole centre, at which the lead dips (the hole eroded by
the lead's footprint, at every peg yaw within the margin; tilt_search.lead_geometry's, kept as a polygon here).
near_m / far_m: the hole's extent along -d / +d; band_m: the region's extent across d (lead offsets from the hole
centre, along a = z x d); along_m: its extent along d (along_m[0]: where a lead sliding along d enters it, i.e.
dips: `pipeline insert` puts the hole centre at dip lead - along_m[0] d - mid(band_m) a). eff_band_m: the offsets
across d whose line crosses the region over at least --min-chord (1 mm): the projection overstates a region crossed
obliquely (a rectangle's corner lead at 45 deg: near the band's ends the line only cuts a corner of it). The line
spacing is band_fraction x the effective band's width (at most the simulation's line_spacing); the simulation's
(the projected band's) is kept as line_spacing_sim_m. dip_region_m: that region's outline (hole frame, simplified to
20 um), for `pipeline insert`'s belief grid.

free_m (per hole, 2026-11-21): the peg offsets from the hole centre (hole frame, the peg at the hole's yaw) at which
the upright peg fits in the hole: an upright plunge there goes in, and a peg in the hole touches its walls on this
outline (`pipeline insert`'s wall-touch centring fits the hole centre to it).
"""

import argparse
import math

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Polygon

from peg_board_env.board_geometry import extract_board_holes
from peg_board_env.build_xml import build_xml
from peg_board_env.config import PegBoardConfig
from peg_board_env.insertion_rl import InsertionRLConfig
from peg_board_env.tilt_insertion import lead_local_point, slide_direction
from peg_board_env.tilt_search import (DEFAULT_LEADS, DEFAULT_PATTERNS, TILT_SLIDE_PRELOAD, TiltSearchConfig,
                                       lead_geometry, outline, parse_lead)

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--min-chord", type=float, default=0.001,
                    help="m: a line counts as crossing the dip region if it runs this far inside it (default 1 mm)")
args = parser.parse_args()

cfg = TiltSearchConfig()
rl = InsertionRLConfig()
depth = cfg.footprint_dip * cfg.dip_depth / math.sin(math.radians(cfg.tilt_deg))


def leads_of(shape: str, profile) -> list[tuple[int, str]]:
    """(group, lead): every lead equivalent to each entry of the shape's default cycle."""
    cycle = DEFAULT_LEADS[shape][0].split("+")
    if shape == "l_shape":  # no symmetry: the cycle's leads themselves
        return [(g, f"{kind}{index}") for g, (kind, index) in enumerate(parse_lead(lead, profile) for lead in cycle)]
    kind, _ = parse_lead(cycle[0], profile)
    count = len(getattr(profile, "normals", [None]))
    convex = profile.convex_corners() if hasattr(profile, "normals") else [True]
    return [(0, f"{kind}{i}") for i in range(count) if kind == "flat" or convex[i]]


def dip_region(scene, lead: str, depth: float, yaw_margin: float):
    """The dip region as a polygon (hole frame, lead positions from the hole centre): tilt_search.lead_geometry's
    region (the widest part per yaw) intersected over the peg yawed -yaw_margin, 0, +yaw_margin. None: none."""
    kind, index = parse_lead(lead, scene.hole_profile)
    d = slide_direction(scene.hole_profile, kind, index)
    across = np.array([-d[1], d[0]])
    hole_poly = Polygon(outline(scene.hole_profile))
    peg = outline(scene.peg_profile)
    lead_local = lead_local_point(scene.peg_profile, d[None])[0]
    convex = hole_poly.convex_hull.area - hole_poly.area < 1.0e-6 * hole_poly.area
    result = None
    for yaw in (-yaw_margin, 0.0, yaw_margin):
        c, s = math.cos(yaw), math.sin(yaw)
        rot = np.array([[c, -s], [s, c]])
        pts = peg @ rot.T
        top = (pts @ d).max()
        half = 1.0
        keep = Polygon([(top - depth) * d - half * across, (top + half) * d - half * across,
                        (top + half) * d + half * across, (top - depth) * d + half * across])
        footprint = Polygon(pts).intersection(keep)
        footprint = footprint.convex_hull if convex else footprint.segmentize(2.0e-4)
        region = hole_poly
        for v in np.asarray(footprint.exterior.coords)[:-1] - rot @ lead_local[:2]:
            region = region.intersection(affinity.translate(hole_poly, -v[0], -v[1])).simplify(1.0e-7)
            if region.is_empty:
                return None
        parts = [g for g in getattr(region, "geoms", [region]) if g.geom_type == "Polygon" and g.area > 0.0]
        if not parts:
            return None
        widest = max(parts, key=lambda g: np.ptp(np.asarray(g.exterior.coords) @ across))
        result = widest if result is None else result.intersection(widest)
        if result.is_empty:
            return None
    parts = [g for g in getattr(result, "geoms", [result]) if g.geom_type == "Polygon" and g.area > 0.0]
    return max(parts, key=lambda g: g.area) if parts else None


def free_region(scene):
    """The offsets at which the upright peg (the hole's yaw) fits in the hole: the hole eroded by the peg."""
    hole_poly = Polygon(outline(scene.hole_profile))
    peg = Polygon(outline(scene.peg_profile))
    convex = hole_poly.convex_hull.area - hole_poly.area < 1.0e-6 * hole_poly.area
    points = peg.convex_hull if convex else peg.segmentize(2.0e-4)
    region = hole_poly
    for v in np.asarray(points.exterior.coords)[:-1]:
        region = region.intersection(affinity.translate(hole_poly, -v[0], -v[1])).simplify(1.0e-7)
        if region.is_empty:
            return None
    parts = [g for g in getattr(region, "geoms", [region]) if g.geom_type == "Polygon" and g.area > 0.0]
    return max(parts, key=lambda g: g.area) if parts else None


def outline_text(polygon, tolerance: float = 2.0e-5) -> str:
    xy = np.asarray(polygon.simplify(tolerance).exterior.coords)[:-1]
    return "[" + ", ".join(f"[{x:.6f}, {y:.6f}]" for x, y in xy) + "]"


def effective_band(region, d: np.ndarray, min_chord: float, step: float = 2.0e-5) -> tuple[float, float] | None:
    """The longest run of offsets across d whose line along d runs at least min_chord inside the region."""
    across = np.array([-d[1], d[0]])
    xy = np.asarray(region.exterior.coords)
    lo, hi = (xy @ across).min(), (xy @ across).max()
    best, start, previous = None, None, None
    for t in np.arange(lo, hi + step, step):
        line = LineString([t * across - d, t * across + d])
        cut = region.intersection(line)
        pieces = getattr(cut, "geoms", [cut])
        chord = max((p.length for p in pieces if p.geom_type == "LineString"), default=0.0)
        if chord >= min_chord:
            start = t if start is None else start
            previous = t
        elif start is not None:
            best = max(best or (start, previous), (start, previous), key=lambda b: b[1] - b[0])
            start = None
    if start is not None:
        best = max(best or (start, previous), (start, previous), key=lambda b: b[1] - b[0])
    return best


print("# Generated by pipeline/tools/extract_tilt_leads.py from rl_manip peg_board_env/tilt_search.py -- do not edit.")
print("#")
print("# Per hole: the tilt search (peg_search tilt-slide) of `pipeline insert`, hole frame of holes.yaml.")
print("# d_deg: slide direction from the hole x axis; peg_radius_m / lead_across_m: the lead point along d and")
print("# across it (left, z x d) from the peg axis; the search is sized for the simulation's hole error disc")
print(f"# (error_m). Tilt {cfg.tilt_deg:g} deg, dip depth {cfg.dip_depth * 1e6:g} um, band yaw margin "
      f"{cfg.band_yaw_margin_deg:g} deg. Leads of one group are equivalent (the shape's symmetry).")
print("# near_m / far_m: the hole along -d / +d; band_m / along_m: the dip region (lead positions from the hole centre at")
print("# which it dips) across / along d; eff_band_m: the offsets whose line runs at least "
      f"{args.min_chord * 1e3:g} mm inside it.")
print("# line_spacing_m: band fraction x the effective band (line_spacing_sim_m: x the projected band, the simulation's).")
print("# pattern: the simulation's choice (tilt_search.DEFAULT_PATTERNS); config insert.pattern overrides it.")
print("holes:")
for hole in extract_board_holes():
    try:
        scene = build_xml(PegBoardConfig(nworld=1, shape=hole.shape, target_clearance_mm=hole.clearance_mm))
    except ValueError as error:  # the 3 mm rectangle hole (2026-11-21): `pipeline insert` uses the shape's nearest
        print(f"  # {hole.name}: not in the simulation ({error})")
        continue
    profile = scene.hole_profile
    radius = rl.error_radius(hole.clearance_mm)
    print(f"  {hole.name}:")
    print(f"    pattern: {DEFAULT_PATTERNS[hole.shape][0]}")
    print(f"    error_m: {radius:.4f}")
    free = free_region(scene)
    print(f"    free_m: {outline_text(free, 2.0e-6) if free is not None else 'null'}")
    print("    leads:")
    for group, lead in leads_of(hole.shape, profile):
        g = lead_geometry(scene, lead, depth, math.radians(cfg.band_yaw_margin_deg))
        kind, _ = parse_lead(lead, profile)
        d = np.asarray(g.d, float)
        across = np.array([-d[1], d[0]])
        along_r = float(np.dot(g.lead_local[:2], d))
        across_r = float(np.dot(g.lead_local[:2], across))
        head = (f"      - {{lead: {lead}, kind: {kind}, group: {group}, d_deg: {math.degrees(math.atan2(d[1], d[0])):.2f}, "
                f"peg_radius_m: {along_r:.5f}, lead_across_m: {across_r:.5f}, near_m: {g.near:.5f}, far_m: {g.far:.5f}")
        if g.band is None:
            print(head + ", dips: false}")
            continue
        region = dip_region(scene, lead, depth, math.radians(cfg.band_yaw_margin_deg))
        eff = effective_band(region, d, args.min_chord) if region is not None else None
        width = g.band[1] - g.band[0]
        eff_width = (eff[1] - eff[0]) if eff else 0.0
        spacing_sim = min(cfg.line_spacing, cfg.band_fraction * width)
        spacing = min(cfg.line_spacing, cfg.band_fraction * eff_width) if eff else spacing_sim
        pitch = min(cfg.line_spacing, cfg.band_fraction * min(width, g.along[1] - g.along[0]))
        run_up = radius + g.near + cfg.line_margin
        slide = run_up + radius + g.far + cfg.line_margin - TILT_SLIDE_PRELOAD
        eff_text = f"[{eff[0]:.5f}, {eff[1]:.5f}]" if eff else "null"
        print(head + f", band_m: [{g.band[0]:.5f}, {g.band[1]:.5f}], along_m: [{g.along[0]:.5f}, {g.along[1]:.5f}], "
              f"eff_band_m: {eff_text}, lead_run_up_m: {run_up:.5f}, slide_length_m: {slide:.5f}, "
              f"line_spacing_m: {spacing:.5f}, line_spacing_sim_m: {spacing_sim:.5f}, "
              f"lines: {1 + 2 * math.ceil(radius / spacing)}, spiral_pitch_m: {pitch:.5f}, "
              f"spiral_radius_m: {radius + max(g.near, g.far) + cfg.line_margin:.5f}, "
              f"dip_region_m: {outline_text(region) if region is not None else 'null'}}}")
