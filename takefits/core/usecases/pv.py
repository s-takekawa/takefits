"""PV diagram usecases."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
from scipy.ndimage import map_coordinates

from takefits.core.app_state import AppState
from .utils import get_axis_ctype, parse_world_coordinate, update_datamin_datamax_if_present

POSITION_ORIGIN_START = "start"
POSITION_ORIGIN_CENTER = "center"
PV_X_AXIS_POSITION = "position"
PV_X_AXIS_PHI = "phi"

# Polyline spline interpolation types. "none" = straight segments (legacy
# behavior); "catmull_rom" = centripetal Catmull-Rom that *interpolates* the
# nodes; "bspline" = clamped cubic B-spline that *approximates* the interior
# nodes (smoother / de-noises rough nodes, stays inside the control hull). New
# types can be added without changing the PathSamples contract.
PV_SPLINE_NONE = "none"
PV_SPLINE_CATMULL_ROM = "catmull_rom"
PV_SPLINE_BSPLINE = "bspline"


def normalize_pv_spline_type(value) -> str:
    """Normalize a polyline spline-type value (forward/back compatible)."""
    text = str(value if value is not None else "none").strip().lower()
    text = text.replace("-", "_").replace(" ", "")
    if text in ("catmull_rom", "catmullrom", "cr", "smooth", "spline"):
        return PV_SPLINE_CATMULL_ROM
    if text in ("bspline", "b_spline", "basis_spline", "approx", "approximate"):
        return PV_SPLINE_BSPLINE
    return PV_SPLINE_NONE


def clamp_pv_smoothness(value) -> float:
    """Clamp a polyline smoothness amount into [0, 1] (default 1.0)."""
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return 1.0
    if not np.isfinite(amount):
        return 1.0
    return float(min(1.0, max(0.0, amount)))


def normalize_pv_position_unit(position_unit: Optional[str]) -> str:
    """Normalize GUI/display PV position-axis units for FITS export."""
    value = str(position_unit or "deg").strip().lower()
    value = value.replace(" ", "")
    if value in ("pixel", "pixels", "pix"):
        return "pix"
    if value in ("arcsec", "arcsecond", "arcseconds"):
        return "arcsec"
    if value in ("arcmin", "arcminute", "arcminutes"):
        return "arcmin"
    if value in ("deg", "degree", "degrees"):
        return "deg"
    return "deg"


@dataclass(frozen=True)
class StraightPathGeometry:
    """Straight PV path in image pixel coordinates."""

    start: tuple[float, float]
    end: tuple[float, float]

    def __post_init__(self):
        object.__setattr__(self, "start", (float(self.start[0]), float(self.start[1])))
        object.__setattr__(self, "end", (float(self.end[0]), float(self.end[1])))

    @classmethod
    def from_endpoints(cls, x0: float, y0: float, x1: float, y1: float) -> "StraightPathGeometry":
        return cls(start=(x0, y0), end=(x1, y1))

    @property
    def length_px(self) -> float:
        return float(np.hypot(self.end[0] - self.start[0], self.end[1] - self.start[1]))


@dataclass(frozen=True)
class CirclePathGeometry:
    """Circular PV path or circular arc in image pixel coordinates."""

    center: tuple[float, float]
    radius_px: float
    start_angle_rad: float = 0.0
    end_angle_rad: float = 2.0 * np.pi

    def __post_init__(self):
        radius = float(self.radius_px)
        if not np.isfinite(radius) or radius < 0.0:
            raise ValueError("radius_px must be a non-negative finite value")
        object.__setattr__(self, "center", (float(self.center[0]), float(self.center[1])))
        object.__setattr__(self, "radius_px", radius)
        object.__setattr__(self, "start_angle_rad", float(self.start_angle_rad))
        object.__setattr__(self, "end_angle_rad", float(self.end_angle_rad))

    @property
    def sweep_angle_rad(self) -> float:
        return float(self.end_angle_rad - self.start_angle_rad)

    @property
    def length_px(self) -> float:
        return float(self.radius_px * abs(self.sweep_angle_rad))


@dataclass(frozen=True)
class EllipsePathGeometry:
    """Elliptical PV path or elliptical arc in image pixel coordinates."""

    center: tuple[float, float]
    semi_major_px: float
    semi_minor_px: float
    pa_rad: float = 0.0
    start_phi_rad: float = 0.0
    end_phi_rad: float = 2.0 * np.pi

    def __post_init__(self):
        major = float(self.semi_major_px)
        minor = float(self.semi_minor_px)
        if not np.isfinite(major) or major < 0.0:
            raise ValueError("semi_major_px must be a non-negative finite value")
        if not np.isfinite(minor) or minor < 0.0:
            raise ValueError("semi_minor_px must be a non-negative finite value")
        object.__setattr__(self, "center", (float(self.center[0]), float(self.center[1])))
        object.__setattr__(self, "semi_major_px", major)
        object.__setattr__(self, "semi_minor_px", minor)
        object.__setattr__(self, "pa_rad", float(self.pa_rad))
        object.__setattr__(self, "start_phi_rad", float(self.start_phi_rad))
        object.__setattr__(self, "end_phi_rad", float(self.end_phi_rad))

    @property
    def sweep_phi_rad(self) -> float:
        return float(self.end_phi_rad - self.start_phi_rad)

    @property
    def length_px(self) -> float:
        return _ellipse_arc_length(
            self.semi_major_px,
            self.semi_minor_px,
            self.start_phi_rad,
            self.end_phi_rad,
        )


def _catmull_rom_segment(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    t0: float,
    t1: float,
    t2: float,
    t3: float,
    ts: np.ndarray,
) -> np.ndarray:
    """Evaluate a non-uniform Catmull-Rom segment (P1->P2) at parameters ``ts``.

    Uses the Barry-Goldman recursive (pyramidal) form. Each linear blend guards
    against a zero knot interval so duplicated phantom endpoints (where two knots
    coincide and the two points are identical) do not divide by zero.
    """
    ts = np.asarray(ts, dtype=float).reshape(-1, 1)

    def lerp(a, b, ta, tb):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if tb == ta:
            # Coincident knots only occur for duplicated phantom points where
            # a == b, so the blended value is just a (broadcast to all samples).
            return a + np.zeros((ts.shape[0], 1))
        w = (ts - ta) / (tb - ta)
        return a * (1.0 - w) + b * w

    a1 = lerp(p0, p1, t0, t1)
    a2 = lerp(p1, p2, t1, t2)
    a3 = lerp(p2, p3, t2, t3)
    b1 = lerp(a1, a2, t0, t2)
    b2 = lerp(a2, a3, t1, t3)
    return lerp(b1, b2, t1, t2)


def catmull_rom_densify(
    vertices,
    *,
    samples_per_segment: Optional[int] = None,
    alpha: float = 0.5,
    smoothness: float = 1.0,
) -> np.ndarray:
    """Densify control ``vertices`` with a centripetal Catmull-Rom spline.

    The returned polyline interpolates (passes through) every control vertex and
    is clamped at the ends by duplicating the first/last control point as phantom
    points, so the curve is tangent to the end nodes. With fewer than three
    distinct vertices the spline is identical to the straight chord, so the
    control vertices are returned unchanged.

    ``smoothness`` (0..1) blends each densified sample toward its own control
    segment's straight chord at the same parameter. Because segment endpoints are
    control nodes, the blend keeps the curve passing through *every node* at all
    smoothness levels: ``1.0`` is the full spline, ``0.0`` collapses to the
    straight polyline, and intermediate values reduce the bulge (and any
    overshoot) proportionally.

    Args:
        vertices: ordered control points, shape (N, 2).
        samples_per_segment: fixed sample count per control segment. When None,
            each segment uses ``max(8, ceil(chord_length_px))`` samples.
        alpha: parameterization exponent (0.5 = centripetal, the default; 0 =
            uniform; 1 = chordal).
        smoothness: blend amount in [0, 1] between the straight chords (0) and the
            full spline (1).

    Returns:
        ``(M, 2)`` float array of densified points including every control vertex.
    """
    pts = np.asarray(vertices, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("vertices must have shape (N, 2)")
    n = pts.shape[0]
    if n < 3:
        return pts.copy()
    s = clamp_pv_smoothness(smoothness)

    # Phantom endpoints clamp the spline to the first/last nodes.
    padded = np.vstack([pts[0], pts, pts[-1]])  # real points live at index 1..n
    knots = np.zeros(padded.shape[0], dtype=float)
    for i in range(1, padded.shape[0]):
        dist = float(np.hypot(*(padded[i] - padded[i - 1])))
        knots[i] = knots[i - 1] + (dist ** alpha if dist > 0.0 else 0.0)

    out: list[np.ndarray] = []
    for seg in range(1, n):  # control segment between padded[seg] and padded[seg+1]
        p0, p1, p2, p3 = padded[seg - 1], padded[seg], padded[seg + 1], padded[seg + 2]
        t0, t1, t2, t3 = knots[seg - 1], knots[seg], knots[seg + 1], knots[seg + 2]
        if samples_per_segment is not None:
            count = max(2, int(samples_per_segment))
        else:
            chord = float(np.hypot(*(p2 - p1)))
            count = max(8, int(np.ceil(chord)))
        params = np.linspace(t1, t2, count + 1)
        is_last = seg == n - 1
        if not is_last:
            params = params[:-1]  # drop the shared node; the next segment emits it
        seg_pts = _catmull_rom_segment(p0, p1, p2, p3, t0, t1, t2, t3, params)
        if s < 1.0:
            # Blend toward the straight chord at the same (knot) parameter; the
            # control nodes (segment endpoints) are unaffected, so the result
            # still interpolates every node.
            u = ((params - t1) / (t2 - t1)).reshape(-1, 1)
            chord_pts = p1 + u * (p2 - p1)
            seg_pts = (1.0 - s) * chord_pts + s * seg_pts
        out.append(seg_pts)

    return np.vstack(out)


def _bspline_point(degree: int, knots: np.ndarray, ctrl: np.ndarray, t: float) -> np.ndarray:
    """Evaluate a clamped B-spline at parameter ``t`` via de Boor's algorithm."""
    ncp = ctrl.shape[0]
    n = ncp - 1
    p = degree
    # Knot span k with knots[k] <= t < knots[k+1] (clamped right end at t == 1).
    if t >= knots[n + 1]:
        k = n
    elif t <= knots[p]:
        k = p
    else:
        lo, hi = p, n + 1
        mid = (lo + hi) // 2
        while t < knots[mid] or t >= knots[mid + 1]:
            if t < knots[mid]:
                hi = mid
            else:
                lo = mid
            mid = (lo + hi) // 2
        k = mid
    d = [ctrl[k - p + j].astype(float).copy() for j in range(p + 1)]
    for r in range(1, p + 1):
        for j in range(p, r - 1, -1):
            denom = knots[k + 1 + j - r] - knots[k + j - p]
            alpha = 0.0 if denom == 0.0 else (t - knots[k + j - p]) / denom
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]
    return d[p]


def bspline_densify(
    vertices,
    *,
    samples_per_segment: Optional[int] = None,
    smoothness: float = 1.0,
) -> np.ndarray:
    """Densify control ``vertices`` with a clamped cubic B-spline.

    Unlike the Catmull-Rom curve this *approximates* the interior nodes (it does
    not pass through them), which smooths/de-noises roughly placed nodes. The
    curve is clamped so it still starts and ends exactly on the first/last node,
    and a cubic B-spline stays within the convex hull of its control points, so
    it never overshoots. The degree drops to ``min(3, N-1)`` for short paths, and
    with fewer than three vertices the result is the straight chord.

    ``smoothness`` (0..1) blends each sample toward the control polygon at the
    same normalized parameter: ``1.0`` is the full B-spline, ``0.0`` collapses to
    the straight polyline.

    Returns:
        ``(M, 2)`` float array of densified points.
    """
    pts = np.asarray(vertices, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("vertices must have shape (N, 2)")
    n = pts.shape[0]
    if n < 3:
        return pts.copy()
    s = clamp_pv_smoothness(smoothness)

    degree = min(3, n - 1)
    n_internal = n - degree - 1
    internal = [float(i + 1) / float(n - degree) for i in range(n_internal)] if n_internal > 0 else []
    knots = np.array([0.0] * (degree + 1) + internal + [1.0] * (degree + 1), dtype=float)

    if samples_per_segment is not None:
        count = max(2, int(samples_per_segment)) * (n - 1) + 1
    else:
        seg = np.diff(pts, axis=0)
        total = float(np.sum(np.hypot(seg[:, 0], seg[:, 1])))
        count = max(8 * (n - 1), int(np.ceil(total)))
    # Include the control-node parameters so the control polygon is sampled at its
    # corners; this makes the smoothness=0 blend an exact straight polyline (no
    # corner cutting) and keeps the blended poly component accurate.
    node_params = np.linspace(0.0, 1.0, n)
    params = np.unique(np.concatenate([np.linspace(0.0, 1.0, count), node_params]))
    curve = np.array([_bspline_point(degree, knots, pts, float(t)) for t in params])

    if s < 1.0:
        # Blend toward the control polygon (degree-1 B-spline by index parameter);
        # at s == 0 this is exactly the straight node-to-node polyline.
        idx = params * (n - 1)
        i = np.clip(np.floor(idx).astype(int), 0, n - 2)
        frac = (idx - i).reshape(-1, 1)
        poly = pts[i] + frac * (pts[i + 1] - pts[i])
        curve = (1.0 - s) * poly + s * curve

    return curve


@dataclass(frozen=True)
class PolylinePathGeometry:
    """Polyline (multi-segment) PV path in image pixel coordinates.

    The ordered ``vertices`` are the control nodes. When ``spline_type`` is a
    curve (e.g. ``"catmull_rom"``) they become control points of an interpolating
    spline whose roundness is set by ``smoothness`` (0..1). Sampling, length, and
    the slice normals are all derived from ``effective_vertices`` (the densified
    curve) so the shared ``PathSamples`` contract and the straight/polyline width
    integration are reused unchanged. ``spline_type="none"`` is the straight
    multi-segment path.
    """

    vertices: tuple[tuple[float, float], ...]
    spline_type: str = PV_SPLINE_NONE
    smoothness: float = 1.0

    def __post_init__(self):
        coerced = [(float(vx), float(vy)) for vx, vy in self.vertices]
        if len(coerced) < 2:
            raise ValueError("Polyline path requires at least 2 vertices")
        # Drop consecutive duplicate vertices: zero-length segments would make
        # the local segment direction (and therefore the slice normal) undefined.
        deduped: list[tuple[float, float]] = [coerced[0]]
        for vx, vy in coerced[1:]:
            px, py = deduped[-1]
            if np.hypot(vx - px, vy - py) > 0.0:
                deduped.append((vx, vy))
        if len(deduped) < 2:
            raise ValueError("Polyline path requires at least 2 distinct vertices")
        object.__setattr__(self, "vertices", tuple(deduped))
        object.__setattr__(self, "spline_type", normalize_pv_spline_type(self.spline_type))
        object.__setattr__(self, "smoothness", clamp_pv_smoothness(self.smoothness))

    @classmethod
    def from_points(
        cls,
        points,
        spline_type: str = PV_SPLINE_NONE,
        smoothness: float = 1.0,
    ) -> "PolylinePathGeometry":
        return cls(
            vertices=tuple((float(p[0]), float(p[1])) for p in points),
            spline_type=spline_type,
            smoothness=smoothness,
        )

    @property
    def is_smooth(self) -> bool:
        """True when the path renders as a curve rather than straight segments."""
        return (
            self.spline_type != PV_SPLINE_NONE
            and self.smoothness > 0.0
            and len(self.vertices) >= 3
        )

    @property
    def effective_vertices(self) -> tuple[tuple[float, float], ...]:
        """Control vertices, or the densified spline curve when smoothing is on."""
        cached = getattr(self, "_effective_cache", None)
        if cached is not None:
            return cached
        if len(self.vertices) >= 3 and self.smoothness > 0.0 and self.spline_type in (
            PV_SPLINE_CATMULL_ROM,
            PV_SPLINE_BSPLINE,
        ):
            arr = np.asarray(self.vertices, dtype=float)
            if self.spline_type == PV_SPLINE_BSPLINE:
                dense = bspline_densify(arr, smoothness=self.smoothness)
            else:
                dense = catmull_rom_densify(arr, smoothness=self.smoothness)
            result = tuple((float(x), float(y)) for x, y in dense)
        else:
            # "none", a degenerate 2-node path, or zero smoothness -> straight.
            result = self.vertices
        object.__setattr__(self, "_effective_cache", result)
        return result

    @property
    def segment_lengths_px(self) -> np.ndarray:
        verts = np.asarray(self.effective_vertices, dtype=float)
        deltas = np.diff(verts, axis=0)
        return np.hypot(deltas[:, 0], deltas[:, 1])

    @property
    def length_px(self) -> float:
        return float(np.sum(self.segment_lengths_px))


@dataclass(frozen=True)
class PathSamples:
    """Sampled path positions and local normals in pixel coordinates."""

    xs: np.ndarray
    ys: np.ndarray
    normal_x: np.ndarray
    normal_y: np.ndarray
    distance_px: np.ndarray
    length_px: float
    phi_rad: Optional[np.ndarray] = None

    @property
    def num_samples(self) -> int:
        return int(self.xs.size)


def sample_count_from_spacing(line_length_px: float, sample_spacing_pix: Optional[float] = None) -> int:
    """Return the number of PV samples for a line length and optional pixel spacing."""
    length = max(0.0, float(line_length_px))
    if sample_spacing_pix is None:
        return max(1, int(round(length)))

    spacing = float(sample_spacing_pix)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("sample_spacing_pix must be a positive finite value")
    if length <= 0.0:
        return 1
    return max(2, int(round(length / spacing)) + 1)


def _sample_count(line_length_px: float, num_samples: Optional[int], sample_spacing_pix: Optional[float]) -> int:
    if num_samples is not None:
        return max(1, int(num_samples))
    return sample_count_from_spacing(line_length_px, sample_spacing_pix)


def _ellipse_speed(semi_major_px: float, semi_minor_px: float, phi_rad: np.ndarray) -> np.ndarray:
    return np.sqrt(
        (float(semi_major_px) * np.sin(phi_rad)) ** 2
        + (float(semi_minor_px) * np.cos(phi_rad)) ** 2
    )


def _ellipse_arc_lookup(
    semi_major_px: float,
    semi_minor_px: float,
    start_phi_rad: float,
    end_phi_rad: float,
    *,
    min_steps: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    sweep = float(end_phi_rad - start_phi_rad)
    if abs(sweep) <= 0.0:
        phis = np.array([float(start_phi_rad), float(end_phi_rad)], dtype=float)
        return phis, np.array([0.0, 0.0], dtype=float)

    steps = max(int(min_steps), int(np.ceil(abs(sweep) / (2.0 * np.pi) * 2048)))
    phis = np.linspace(float(start_phi_rad), float(end_phi_rad), steps + 1)
    speeds = _ellipse_speed(semi_major_px, semi_minor_px, phis)
    deltas = np.abs(np.diff(phis)) * (speeds[:-1] + speeds[1:]) / 2.0
    distances = np.concatenate(([0.0], np.cumsum(deltas)))
    return phis, distances


def _ellipse_arc_length(
    semi_major_px: float,
    semi_minor_px: float,
    start_phi_rad: float,
    end_phi_rad: float,
) -> float:
    _, distances = _ellipse_arc_lookup(
        semi_major_px,
        semi_minor_px,
        start_phi_rad,
        end_phi_rad,
    )
    return float(distances[-1]) if distances.size else 0.0


def _sample_straight_path_points(
    path: StraightPathGeometry,
    *,
    num_samples: Optional[int] = None,
    sample_spacing_pix: Optional[float] = None,
) -> PathSamples:
    length_px = path.length_px
    count = _sample_count(length_px, num_samples, sample_spacing_pix)
    x0, y0 = path.start
    x1, y1 = path.end
    xs = np.linspace(x0, x1, count)
    ys = np.linspace(y0, y1, count)
    distance_px = np.linspace(0.0, length_px, count) if count > 1 else np.array([0.0], dtype=float)

    if length_px > 0.0:
        perp_x = -(y1 - y0) / length_px
        perp_y = (x1 - x0) / length_px
    else:
        perp_x, perp_y = 0.0, 1.0
    return PathSamples(
        xs=xs,
        ys=ys,
        normal_x=np.full(count, perp_x, dtype=float),
        normal_y=np.full(count, perp_y, dtype=float),
        distance_px=distance_px,
        length_px=length_px,
    )


def _ellipse_samples_from_phi(
    path: EllipsePathGeometry,
    phi: np.ndarray,
    distance_px: np.ndarray,
    length_px: float,
) -> PathSamples:
    count = int(phi.size)
    cos_pa = np.cos(path.pa_rad)
    sin_pa = np.sin(path.pa_rad)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    local_x = path.semi_major_px * cos_phi
    local_y = path.semi_minor_px * sin_phi
    cx, cy = path.center
    xs = cx + local_x * cos_pa - local_y * sin_pa
    ys = cy + local_x * sin_pa + local_y * cos_pa

    sweep_direction = 1.0 if path.sweep_phi_rad >= 0.0 else -1.0
    tangent_local_x = -path.semi_major_px * sin_phi * sweep_direction
    tangent_local_y = path.semi_minor_px * cos_phi * sweep_direction
    tangent_x = tangent_local_x * cos_pa - tangent_local_y * sin_pa
    tangent_y = tangent_local_x * sin_pa + tangent_local_y * cos_pa
    tangent_norm = np.hypot(tangent_x, tangent_y)
    valid = tangent_norm > 0.0
    normal_x = np.zeros(count, dtype=float)
    normal_y = np.ones(count, dtype=float)
    normal_x[valid] = -tangent_y[valid] / tangent_norm[valid]
    normal_y[valid] = tangent_x[valid] / tangent_norm[valid]

    return PathSamples(
        xs=xs,
        ys=ys,
        normal_x=normal_x,
        normal_y=normal_y,
        distance_px=distance_px,
        length_px=length_px,
        phi_rad=phi,
    )


def _sample_ellipse_path_points(
    path: EllipsePathGeometry,
    *,
    num_samples: Optional[int] = None,
    sample_spacing_pix: Optional[float] = None,
) -> PathSamples:
    lookup_phi, lookup_distance = _ellipse_arc_lookup(
        path.semi_major_px,
        path.semi_minor_px,
        path.start_phi_rad,
        path.end_phi_rad,
    )
    length_px = float(lookup_distance[-1]) if lookup_distance.size else 0.0
    count = _sample_count(length_px, num_samples, sample_spacing_pix)
    distance_px = np.linspace(0.0, length_px, count) if count > 1 else np.array([0.0], dtype=float)
    if length_px > 0.0:
        phi = np.interp(distance_px, lookup_distance, lookup_phi)
    else:
        phi = np.full(count, path.start_phi_rad, dtype=float)
    return _ellipse_samples_from_phi(path, phi, distance_px, length_px)


def _sample_ellipse_path_points_by_phi(
    path: EllipsePathGeometry,
    *,
    num_samples: Optional[int] = None,
    sample_spacing_pix: Optional[float] = None,
) -> PathSamples:
    lookup_phi, lookup_distance = _ellipse_arc_lookup(
        path.semi_major_px,
        path.semi_minor_px,
        path.start_phi_rad,
        path.end_phi_rad,
    )
    length_px = float(lookup_distance[-1]) if lookup_distance.size else 0.0
    count = _sample_count(length_px, num_samples, sample_spacing_pix)
    phi = np.linspace(path.start_phi_rad, path.end_phi_rad, count)
    if length_px > 0.0:
        if lookup_phi[0] <= lookup_phi[-1]:
            distance_px = np.interp(phi, lookup_phi, lookup_distance)
        else:
            distance_px = np.interp(phi, lookup_phi[::-1], lookup_distance[::-1])
    else:
        distance_px = np.zeros(count, dtype=float)
    return _ellipse_samples_from_phi(path, phi, distance_px, length_px)


def _sample_circle_path_points(
    path: CirclePathGeometry,
    *,
    num_samples: Optional[int] = None,
    sample_spacing_pix: Optional[float] = None,
) -> PathSamples:
    length_px = path.length_px
    count = _sample_count(length_px, num_samples, sample_spacing_pix)
    cx, cy = path.center
    radius = path.radius_px
    angles = np.linspace(path.start_angle_rad, path.end_angle_rad, count)
    xs = cx + radius * np.cos(angles)
    ys = cy + radius * np.sin(angles)
    distance_px = np.linspace(0.0, length_px, count) if count > 1 else np.array([0.0], dtype=float)

    if radius > 0.0 and abs(path.sweep_angle_rad) > 0.0:
        direction = 1.0 if path.sweep_angle_rad >= 0.0 else -1.0
        tangent_x = -np.sin(angles) * direction
        tangent_y = np.cos(angles) * direction
        normal_x = -tangent_y
        normal_y = tangent_x
    else:
        normal_x = np.zeros(count, dtype=float)
        normal_y = np.ones(count, dtype=float)

    return PathSamples(
        xs=xs,
        ys=ys,
        normal_x=normal_x,
        normal_y=normal_y,
        distance_px=distance_px,
        length_px=length_px,
        phi_rad=angles,
    )


def _sample_polyline_path_points(
    path: PolylinePathGeometry,
    *,
    num_samples: Optional[int] = None,
    sample_spacing_pix: Optional[float] = None,
) -> PathSamples:
    verts = np.asarray(path.effective_vertices, dtype=float)
    seg_lengths = path.segment_lengths_px
    length_px = float(np.sum(seg_lengths))
    count = _sample_count(length_px, num_samples, sample_spacing_pix)

    if length_px <= 0.0 or count <= 1:
        size = max(1, count)
        return PathSamples(
            xs=np.full(size, verts[0, 0], dtype=float),
            ys=np.full(size, verts[0, 1], dtype=float),
            normal_x=np.zeros(size, dtype=float),
            normal_y=np.ones(size, dtype=float),
            distance_px=np.zeros(size, dtype=float),
            length_px=length_px,
        )

    cum = np.concatenate(([0.0], np.cumsum(seg_lengths)))  # arc length at each node
    distance_px = np.linspace(0.0, length_px, count)
    # Locate the segment each sample falls on, then interpolate within it.
    seg_idx = np.clip(np.searchsorted(cum, distance_px, side="right") - 1, 0, len(seg_lengths) - 1)
    seg_len = seg_lengths[seg_idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(seg_len > 0.0, (distance_px - cum[seg_idx]) / seg_len, 0.0)

    p0 = verts[seg_idx]
    p1 = verts[seg_idx + 1]
    pts = p0 + (p1 - p0) * frac[:, None]

    # Slice normal = perpendicular of the local segment direction (matches the
    # straight-path convention: normal = (-dy, dx) / len).
    seg_vec = p1 - p0
    seg_norm = np.hypot(seg_vec[:, 0], seg_vec[:, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        dir_x = np.where(seg_norm > 0.0, seg_vec[:, 0] / seg_norm, 0.0)
        dir_y = np.where(seg_norm > 0.0, seg_vec[:, 1] / seg_norm, 1.0)

    return PathSamples(
        xs=pts[:, 0],
        ys=pts[:, 1],
        normal_x=-dir_y,
        normal_y=dir_x,
        distance_px=distance_px,
        length_px=length_px,
    )


def sample_path_points(
    path: Union[StraightPathGeometry, CirclePathGeometry, EllipsePathGeometry, PolylinePathGeometry],
    *,
    num_samples: Optional[int] = None,
    sample_spacing_pix: Optional[float] = None,
    sample_axis: str = PV_X_AXIS_POSITION,
) -> PathSamples:
    """Sample a PV path in pixel coordinates."""
    if isinstance(path, StraightPathGeometry):
        return _sample_straight_path_points(
            path,
            num_samples=num_samples,
            sample_spacing_pix=sample_spacing_pix,
        )
    if isinstance(path, PolylinePathGeometry):
        return _sample_polyline_path_points(
            path,
            num_samples=num_samples,
            sample_spacing_pix=sample_spacing_pix,
        )
    if isinstance(path, CirclePathGeometry):
        return _sample_circle_path_points(
            path,
            num_samples=num_samples,
            sample_spacing_pix=sample_spacing_pix,
        )
    if isinstance(path, EllipsePathGeometry):
        if normalize_pv_x_axis_mode(sample_axis) == PV_X_AXIS_PHI:
            return _sample_ellipse_path_points_by_phi(
                path,
                num_samples=num_samples,
                sample_spacing_pix=sample_spacing_pix,
            )
        return _sample_ellipse_path_points(
            path,
            num_samples=num_samples,
            sample_spacing_pix=sample_spacing_pix,
        )
    raise TypeError(f"Unsupported PV path geometry: {type(path).__name__}")


def normalize_position_origin(position_origin: Optional[str]) -> str:
    """Normalize PV position-axis origin values."""
    value = str(position_origin or POSITION_ORIGIN_START).strip().lower()
    if value in {"center", "centre", "middle"}:
        return POSITION_ORIGIN_CENTER
    return POSITION_ORIGIN_START


def normalize_pv_x_axis_mode(x_axis_mode: Optional[str]) -> str:
    """Normalize PV horizontal-axis mode values."""
    value = str(x_axis_mode or PV_X_AXIS_POSITION).strip().lower()
    if value in {"phi", "angle", "azimuth"}:
        return PV_X_AXIS_PHI
    return PV_X_AXIS_POSITION


def position_axis_bounds(length: float, position_origin: str, num_samples: int = 2) -> tuple[float, float]:
    """Return display/export bounds for the PV position axis."""
    if num_samples <= 1:
        return -0.5, 0.5
    length_value = float(length)
    if normalize_position_origin(position_origin) == POSITION_ORIGIN_CENTER:
        half = length_value / 2.0
        return -half, half
    return 0.0, length_value


def position_from_fraction(fraction: float, length: float, position_origin: str) -> float:
    """Convert a 0..1 path fraction to a position-axis coordinate."""
    frac = float(fraction)
    length_value = float(length)
    if normalize_position_origin(position_origin) == POSITION_ORIGIN_CENTER:
        return (frac - 0.5) * length_value
    return frac * length_value


def fraction_from_position(position: float, length: float, position_origin: str) -> float:
    """Convert a position-axis coordinate to a path fraction."""
    length_value = float(length)
    if abs(length_value) <= 1e-9:
        return 0.0
    pos = float(position)
    if normalize_position_origin(position_origin) == POSITION_ORIGIN_CENTER:
        return (pos / length_value) + 0.5
    return pos / length_value


def anchored_straight_line(
    start: tuple[float, float],
    end: tuple[float, float],
    length: float,
    angle_rad: float,
    position_origin: str,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return straight-line endpoints after editing length/angle around the active origin."""
    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(end[0]), float(end[1])
    length_value = max(0.0, float(length))
    angle = float(angle_rad)
    dx = length_value * np.cos(angle)
    dy = length_value * np.sin(angle)

    if normalize_position_origin(position_origin) == POSITION_ORIGIN_CENTER:
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        half_dx = dx / 2.0
        half_dy = dy / 2.0
        return (cx - half_dx, cy - half_dy), (cx + half_dx, cy + half_dy)

    return (x0, y0), (x0 + dx, y0 + dy)


def straight_line_from_center(
    center: tuple[float, float],
    length: float,
    angle_rad: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return straight-line endpoints from center, length, and display angle."""
    cx, cy = float(center[0]), float(center[1])
    length_value = max(0.0, float(length))
    angle = float(angle_rad)
    half_dx = length_value * np.cos(angle) / 2.0
    half_dy = length_value * np.sin(angle) / 2.0
    return (cx - half_dx, cy - half_dy), (cx + half_dx, cy + half_dy)


def set_pv_endpoints(
    state: AppState,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    width: float = 0.0
) -> AppState:
    """
    Set the PV diagram slice endpoints.

    Args:
        state: The AppState to update
        x0, y0: Start point (pixel coordinates)
        x1, y1: End point (pixel coordinates)
        width: Slice width in pixels (0 = single pixel width)

    Returns:
        The updated AppState
    """
    state.pv_x0 = x0
    state.pv_y0 = y0
    state.pv_x1 = x1
    state.pv_y1 = y1
    state.pv_width = width
    return state


def compute_pv(
    state: AppState,
    x0: Optional[float] = None,
    y0: Optional[float] = None,
    x1: Optional[float] = None,
    y1: Optional[float] = None,
    width: Optional[float] = None,
    num_samples: Optional[int] = None,
    sample_spacing_pix: Optional[float] = None,
    weight_mode: int = 0,
    start_world: Optional[List[Union[float, str]]] = None,
    end_world: Optional[List[Union[float, str]]] = None,
    path_geometry: Optional[Union[StraightPathGeometry, CirclePathGeometry, EllipsePathGeometry, PolylinePathGeometry]] = None,
    path_type: str = "straight",
    center: Optional[List[float]] = None,
    semi_major_px: Optional[float] = None,
    semi_minor_px: Optional[float] = None,
    pa_rad: float = 0.0,
    start_phi_rad: float = 0.0,
    end_phi_rad: Optional[float] = None,
    vertices: Optional[List[List[float]]] = None,
    vertices_world: Optional[List[List[Union[float, str]]]] = None,
    spline_type: str = PV_SPLINE_NONE,
    smoothness: float = 1.0,
    smooth: Optional[bool] = None,
    sample_axis: str = PV_X_AXIS_POSITION,
) -> np.ndarray:
    """
    Extract a position-velocity diagram from the data cube.

    This is a headless implementation of the PV extraction from
    tools/pv_diagram.py.

    Args:
        state: AppState with loaded data
        x0, y0, x1, y1: Slice endpoints in pixel coordinates.
            If not provided, uses values from state.pv_x0/y0/x1/y1
            (Unless start_world/end_world are provided)
        width: Slice width in pixels (0 = single pixel width)
        num_samples: Number of samples along the slice (default: line length)
        sample_spacing_pix: Approximate spacing between samples in pixels.
            Used only when num_samples is not provided.
        weight_mode: Interpolation mode (0=bilinear/average, 1=gaussian weighted)
        start_world: Start point in world coordinates [x_world, y_world].
                     Overrides x0, y0.
        end_world: End point in world coordinates [x_world, y_world].
                   Overrides x1, y1.
        path_geometry: Optional sampled path geometry. If provided, it takes
            precedence over straight endpoint arguments.
        path_type: Path type for headless callers that do not construct a
            geometry object. Currently supports "straight" and "ellipse".
        center, semi_major_px, semi_minor_px, pa_rad, start_phi_rad,
            end_phi_rad: Ellipse geometry parameters used when
            ``path_type="ellipse"`` and ``path_geometry`` is not provided.
        spline_type: For ``path_type="polyline"`` (without ``path_geometry``),
            ``"catmull_rom"`` interpolates the vertices with a centripetal
            Catmull-Rom spline; ``"none"`` (default) keeps straight segments.
        smoothness: Spline roundness in [0, 1] for ``spline_type="catmull_rom"``
            (1 = full spline, 0 = straight). Ignored for other path types.
        smooth: Deprecated boolean alias for ``spline_type="catmull_rom"``.
        sample_axis: Sampling coordinate for ellipse paths. "position"
            samples uniformly by arc length; "phi" samples uniformly in Phi.

    Returns:
        2D numpy array with shape (n_channels, num_samples)
    """
    if state.data is None:
        raise ValueError("No data loaded")

    path_kind = str(path_type or "straight").strip().lower()
    if path_kind == "ellipse_arc":
        path_kind = "ellipse"

    if path_geometry is None and path_kind == "ellipse":
        if center is None or semi_major_px is None or semi_minor_px is None:
            raise ValueError("Ellipse PV requires center, semi_major_px, and semi_minor_px")
        if len(center) < 2:
            raise ValueError("Ellipse center must have at least 2 coordinates")
        path_geometry = EllipsePathGeometry(
            center=(float(center[0]), float(center[1])),
            semi_major_px=float(semi_major_px),
            semi_minor_px=float(semi_minor_px),
            pa_rad=float(pa_rad),
            start_phi_rad=float(start_phi_rad),
            end_phi_rad=float(end_phi_rad) if end_phi_rad is not None else float(start_phi_rad) + 2.0 * np.pi,
        )

    if path_geometry is None and path_kind == "polyline":
        verts = vertices
        if verts is None and vertices_world is not None:
            if state.wcs is None:
                raise ValueError("WCS required for vertices_world")
            verts = []
            for vw in vertices_world:
                if len(vw) < 2:
                    raise ValueError("each vertex in vertices_world needs at least 2 coordinates")
                w = []
                for i, val in enumerate(vw):
                    if i >= state.wcs.naxis:
                        break
                    ctype = get_axis_ctype(state, i)
                    w.append(parse_world_coordinate(val, ctype))
                while len(w) < state.wcs.naxis:
                    w.append(0.0)
                pix = state.wcs.wcs_world2pix([w], 0)[0]
                verts.append([float(pix[0]), float(pix[1])])
        if not verts or len(verts) < 2:
            raise ValueError("Polyline PV requires at least 2 vertices")
        resolved_spline = normalize_pv_spline_type(spline_type)
        if resolved_spline == PV_SPLINE_NONE and smooth:
            # Backward-compatible alias for the original boolean flag.
            resolved_spline = PV_SPLINE_CATMULL_ROM
        path_geometry = PolylinePathGeometry.from_points(
            verts, spline_type=resolved_spline, smoothness=smoothness
        )

    if path_geometry is not None and (start_world is not None or end_world is not None):
        raise ValueError("start_world/end_world cannot be combined with path_geometry")

    if path_geometry is None and path_kind not in ("straight", ""):
        raise ValueError(f"Unsupported PV path type: {path_type}")

    # Handle world coordinates if provided for straight paths
    if path_geometry is None and start_world is not None:
        if state.wcs is None:
            raise ValueError("WCS required for start_world")
        if len(start_world) < 2:
            raise ValueError("start_world must have at least 2 coordinates (x, y)")

        # Parse strings
        w0 = []
        for i, val in enumerate(start_world):
            # Map input indices to WCS axes 0, 1
            if i >= state.wcs.naxis:
                break
            ctype = get_axis_ctype(state, i)
            w0.append(parse_world_coordinate(val, ctype))

        # Pad with 0s to match naxis
        while len(w0) < state.wcs.naxis:
            w0.append(0.0)

        pix0 = state.wcs.wcs_world2pix([w0], 0)[0]
        x0 = float(pix0[0])
        y0 = float(pix0[1])

    if path_geometry is None and end_world is not None:
        if state.wcs is None:
            raise ValueError("WCS required for end_world")
        if len(end_world) < 2:
            raise ValueError("end_world must have at least 2 coordinates (x, y)")

        w1 = []
        for i, val in enumerate(end_world):
            if i >= state.wcs.naxis:
                break
            ctype = get_axis_ctype(state, i)
            w1.append(parse_world_coordinate(val, ctype))

        while len(w1) < state.wcs.naxis:
            w1.append(0.0)

        pix1 = state.wcs.wcs_world2pix([w1], 0)[0]
        x1 = float(pix1[0])
        y1 = float(pix1[1])

    if width is None:
        width = state.pv_width

    width_value = 0.0 if width is None else float(width)
    if path_geometry is None:
        # Use state values if not provided
        if x0 is None:
            x0 = state.pv_x0
        if y0 is None:
            y0 = state.pv_y0
        if x1 is None:
            x1 = state.pv_x1
        if y1 is None:
            y1 = state.pv_y1

        if any(v is None for v in [x0, y0, x1, y1]):
            raise ValueError("PV slice endpoints not specified")

        path = StraightPathGeometry.from_endpoints(x0, y0, x1, y1)
    else:
        path = path_geometry

    if isinstance(path, StraightPathGeometry):
        set_pv_endpoints(
            state,
            x0=float(path.start[0]),
            y0=float(path.start[1]),
            x1=float(path.end[0]),
            y1=float(path.end[1]),
            width=width_value,
        )

    data = state.data

    # Handle 4D data by selecting current S slice
    if data.ndim == 4:
        data = data[state.current_s]

    if data.ndim != 3:
        raise ValueError(f"Expected 3D data cube, got {data.ndim}D")

    n_vel = data.shape[0]

    samples = sample_path_points(
        path,
        num_samples=num_samples,
        sample_spacing_pix=sample_spacing_pix,
        sample_axis=sample_axis,
    )
    line_length_px = samples.length_px
    num_samples = samples.num_samples

    if line_length_px == 0:
        num_samples = 1
        pv = np.full((n_vel, num_samples), np.nan)
    else:
        xs = samples.xs
        ys = samples.ys
        normal_x = samples.normal_x
        normal_y = samples.normal_y

        pv = np.zeros((n_vel, num_samples), dtype=np.float64)

        if width_value <= 0:
            # Simple interpolation along the line (no width)
            for v in range(n_vel):
                pv[v, :] = map_coordinates(
                    data[v], [ys, xs],
                    order=1, mode='constant', cval=np.nan
                )

        elif weight_mode == 0:
            # Bilinear interpolation averaged over width
            n_width = max(1, int(round(width_value)))
            offsets = np.linspace(-width_value / 2, width_value / 2, n_width)

            for v in range(n_vel):
                values = np.full((n_width, num_samples), np.nan, dtype=np.float64)
                for i, offset in enumerate(offsets):
                    # Use perp vectors directly like in original code
                    # Original: off_x = -np.sin(theta) * off
                    # theta = arctan2(dy, dx), so cos(theta)=dx/len, sin(theta)=dy/len
                    # off_x = - (dy/len) * off = perp_x * off
                    xs_offset = xs + offset * normal_x
                    ys_offset = ys + offset * normal_y
                    values[i, :] = map_coordinates(
                        data[v], [ys_offset, xs_offset],
                        order=1, mode='constant', cval=np.nan
                    )
                # Avoid RuntimeWarning from np.nanmean on all-NaN columns.
                valid = ~np.isnan(values)
                count = valid.sum(axis=0)
                sums = np.nansum(values, axis=0)
                row = np.full(num_samples, np.nan, dtype=np.float64)
                valid_cols = count > 0
                row[valid_cols] = sums[valid_cols] / count[valid_cols]
                pv[v, :] = row

        elif weight_mode == 1:
            # Gaussian weighted interpolation
            sigma = width_value / (2.0 * np.sqrt(2.0 * np.log(2)))
            n_offsets = int(np.ceil(width_value * 2)) + 1
            offsets = np.linspace(-width_value / 2, width_value / 2, n_offsets)
            weights = np.exp(-0.5 * (offsets / sigma) ** 2)
            weights /= weights.sum()

            for v in range(n_vel):
                prof_sum = np.zeros(num_samples)
                weight_sum = np.zeros(num_samples)

                for offset, w in zip(offsets, weights):
                    xs_offset = xs + offset * normal_x
                    ys_offset = ys + offset * normal_y
                    prof = map_coordinates(
                        data[v], [ys_offset, xs_offset],
                        order=1, mode='constant', cval=np.nan
                    )

                    valid = ~np.isnan(prof)
                    prof_sum[valid] += w * prof[valid]
                    weight_sum[valid] += w

                result = np.full(num_samples, np.nan)
                valid_mask = weight_sum > 0
                result[valid_mask] = prof_sum[valid_mask] / weight_sum[valid_mask]
                pv[v, :] = result

    return pv


def export_pv_fits(
    state: AppState,
    pv_data: np.ndarray,
    output_path: str,
    x0: float, y0: float, x1: float, y1: float,
    is_swapped: bool = False,
    history_entries: Optional[list] = None,
    position_origin: str = POSITION_ORIGIN_START,
    path_type: str = "straight",
    path_length_px: Optional[float] = None,
    x_axis_mode: str = PV_X_AXIS_POSITION,
    phi_start_deg: Optional[float] = None,
    phi_end_deg: Optional[float] = None,
    position_unit: str = "deg",
) -> str:
    """
    Export PV diagram to a FITS file.

    Args:
        state: AppState with original header/WCS info
        pv_data: 2D PV data array
        output_path: Path for output FITS file
        x0, y0, x1, y1: Slice endpoints in pixel coordinates
        is_swapped: Whether axes are swapped (True: X=Vel, Y=Pos)
        history_entries: Optional list of strings to add as HISTORY keywords
        position_origin: Position-axis origin ("start" or "center")
        path_type: PV path type for metadata.
        path_length_px: Optional path length override in pixels. Curved paths
            should pass their sampled arc length here.
        x_axis_mode: Horizontal axis mode ("position" or "phi").
        phi_start_deg, phi_end_deg: Phi-axis bounds in degrees when
            ``x_axis_mode="phi"``.
        position_unit: Unit for the position axis when ``x_axis_mode`` is
            ``"position"``. Supported values are pixel/pix, arcsec, arcmin,
            and deg.

    Returns:
        The output file path
    """
    from astropy.io import fits
    from astropy.wcs.utils import proj_plane_pixel_scales
    import re

    # --- 1. Prepare Data Array ---
    data_to_save = pv_data
    if isinstance(data_to_save, np.ma.MaskedArray):
        data_to_save = data_to_save.filled(np.nan)

    # --- 2. Create FITS Header and WCS ---
    header = fits.Header()
    header['NAXIS'] = 2
    header['NAXIS1'] = data_to_save.shape[1]  # X-axis length
    header['NAXIS2'] = data_to_save.shape[0]  # Y-axis length
    if state.header:
        header['BUNIT'] = state.header.get('BUNIT', 'UNKNOWN')

    # Add history
    if history_entries:
        for entry in history_entries:
            header.add_history(entry)

    # --- Define Position/Phi Axis ---
    line_length_px = float(path_length_px) if path_length_px is not None else np.hypot(x1 - x0, y1 - y0)
    path_label = str(path_type or "straight").strip().upper() or "STRAIGHT"
    header['PVPATH'] = (path_label, 'PV path type')
    x_axis_mode = normalize_pv_x_axis_mode(x_axis_mode)
    header['PVXAXIS'] = (x_axis_mode.upper(), 'PV horizontal axis mode')

    # Calculate pixel scale in degrees
    pixel_scale_deg = None
    if state.wcs:
        try:
            if state.wcs.is_celestial:
                scales = proj_plane_pixel_scales(state.wcs)
                pixel_scale_deg = (abs(scales[0]) + abs(scales[1])) / 2.0

            if pixel_scale_deg is None and state.wcs.wcs.cdelt is not None:
                cdelt = state.wcs.wcs.cdelt
                # Use first two axes
                if len(cdelt) >= 2:
                    cdelt1 = abs(cdelt[0])
                    cdelt2 = abs(cdelt[1])
                    unit_str = str(state.wcs.wcs.cunit[0]).lower() if state.wcs.wcs.cunit else ''
                    if unit_str in ('deg', 'degree', 'degrees'):
                        pixel_scale_deg = (cdelt1 + cdelt2) / 2.0
        except Exception:
            pass

    if is_swapped:
        num_pos_pixels = data_to_save.shape[0]  # Position is NAXIS2
    else:
        num_pos_pixels = data_to_save.shape[1]  # Position is NAXIS1

    if x_axis_mode == PV_X_AXIS_PHI:
        phi_start = 0.0 if phi_start_deg is None else float(phi_start_deg)
        phi_end = phi_start + 360.0 if phi_end_deg is None else float(phi_end_deg)
        length_deg = float(phi_end - phi_start)
        pos_ctype = 'PHI'
        pos_cunit = 'deg'
        pos_crval = 0.0
        header['PVPHI0'] = (phi_start, 'PV ellipse Phi start angle [deg]')
        header['PVPHI1'] = (phi_end, 'PV ellipse Phi end angle [deg]')
    else:
        pos_cunit = normalize_pv_position_unit(position_unit)
        if pos_cunit == "pix" or pixel_scale_deg is None:
            length_deg = line_length_px
            pos_cunit = "pix"
        else:
            length_deg = line_length_px * pixel_scale_deg
            if pos_cunit == "arcsec":
                length_deg *= 3600.0
            elif pos_cunit == "arcmin":
                length_deg *= 60.0
        position_origin = normalize_position_origin(position_origin)
        pos_crval = -length_deg / 2.0 if position_origin == POSITION_ORIGIN_CENTER else 0.0
        pos_ctype = 'OFFSET'
        header['PVORIGIN'] = (
            position_origin.upper(),
            'PV position-axis origin'
        )

    if num_pos_pixels > 1:
        pos_cdelt = length_deg / num_pos_pixels
    else:
        pos_cdelt = length_deg

    pos_crpix = 0.5

    # --- Define Velocity Axis ---
    vel_axis_index = 2  # Default to 3rd axis (0-indexed 2)
    vel_ctype = 'VELOCITY'
    vel_cunit = 'km/s'
    vel_crval = 0.0
    vel_cdelt = 1.0
    vel_crpix = 1.0

    if state.wcs and state.wcs.wcs.naxis >= 3:
        # Try to find spectral axis
        spec_axis = -1
        for i, ctype in enumerate(state.wcs.wcs.ctype):
            if any(x in ctype.upper() for x in ['VEL', 'FREQ', 'VRAD', 'VOPT']):
                spec_axis = i
                break

        if spec_axis != -1:
            vel_axis_index = spec_axis
            vel_ctype = state.wcs.wcs.ctype[vel_axis_index]
            vel_crval = state.wcs.wcs.crval[vel_axis_index]
            vel_cdelt = state.wcs.wcs.cdelt[vel_axis_index]
            vel_crpix = state.wcs.wcs.crpix[vel_axis_index]
            if state.wcs.wcs.cunit:
                vel_cunit = state.wcs.wcs.cunit[vel_axis_index].to_string()

    # Attempt to use pretty unit from spectral_metadata if available
    # (This handles the case where fits_loader converted units but WCS might be raw)
    if state.spectral_metadata and 'current_axis_unit' in state.spectral_metadata:
        # If we have a formatted unit string like "Velocity [km/s]" or just "km/s"
        meta_unit = state.spectral_metadata['current_axis_unit']
        # Try to extract content inside brackets if present
        match = re.search(r'\[(.*?)\]', meta_unit)
        if match:
            vel_cunit = match.group(1).strip()
        elif meta_unit.strip():
            vel_cunit = meta_unit.strip()

    # --- Assign WCS keywords based on swap state ---
    if is_swapped:
        # NAXIS1 = Velocity, NAXIS2 = Position
        header['CTYPE1'] = vel_ctype
        header['CUNIT1'] = vel_cunit
        header['CRVAL1'] = vel_crval
        header['CDELT1'] = vel_cdelt
        header['CRPIX1'] = vel_crpix

        header['CTYPE2'] = pos_ctype
        header['CUNIT2'] = pos_cunit
        header['CRVAL2'] = pos_crval
        header['CDELT2'] = pos_cdelt
        header['CRPIX2'] = pos_crpix
    else:
        # NAXIS1 = Position, NAXIS2 = Velocity
        header['CTYPE1'] = pos_ctype
        header['CUNIT1'] = pos_cunit
        header['CRVAL1'] = pos_crval
        header['CDELT1'] = pos_cdelt
        header['CRPIX1'] = pos_crpix

        header['CTYPE2'] = vel_ctype
        header['CUNIT2'] = vel_cunit
        header['CRVAL2'] = vel_crval
        header['CDELT2'] = vel_cdelt
        header['CRPIX2'] = vel_crpix

    update_datamin_datamax_if_present(header, data_to_save)

    # Write file
    hdu = fits.PrimaryHDU(data=data_to_save.astype(np.float32), header=header)
    hdu.writeto(output_path, overwrite=True)

    return output_path
