
#!/usr/bin/env python3
"""
Interactive DXF splitter / picker for spiral spring laser-cut sections.

Adds
----
- direction arrows
- start/end markers
- automatic chaining by endpoint proximity

Auto-chain behavior
-------------------
For the currently selected category, the tool can automatically extend a chain by
greedily adding the nearest unassigned entity whose endpoint matches either end of
the current chain within a user-set gap tolerance.

Recommended workflow
--------------------
1. Open DXF
2. Pick a category
3. Click one seed entity for that category
4. Press "Auto-chain current category"
5. Inspect order and directions
6. Fix with Move Up/Down or Toggle Reverse if needed
7. Generate outputs

Dependencies
------------
    pip install numpy matplotlib

Tkinter is included with most Python installs.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import zipfile
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

PROJECT_ROOT = Path(__file__).resolve().parent
MPLCONFIG_DIR = PROJECT_ROOT / ".mplconfig"
MPLCONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

DEFAULT_DXF_DIR = PROJECT_ROOT / "dxf_files"
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "splitter_configs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "splitter_outputs"


CATEGORIES = [
    "upper_edge",
    "lower_edge",
    "outer_fixture",
    "inner_transition",
    "inner_cutout",
]

CATEGORY_META = {
    "upper_edge": {
        "label": "Upper spring edge",
        "description": "One of the two main spring edges that will be split into sections.",
    },
    "lower_edge": {
        "label": "Lower spring edge",
        "description": "The other main spring edge that will be split into sections.",
    },
    "outer_fixture": {
        "label": "Outer fixture + holes",
        "description": "Entire outer-end fixture geometry. Circular holes belong here and export as one DXF.",
    },
    "inner_transition": {
        "label": "Transition region",
        "description": "Geometry bridging from the spring strip into the inner ring. Exports together with the inner ring.",
    },
    "inner_cutout": {
        "label": "Splined inner ring",
        "description": "Entire inner splined ring / arbor-end geometry. Exports together with the transition region.",
    },
}

CHAINABLE_CATEGORIES = {
    "upper_edge",
    "lower_edge",
    "outer_fixture",
    "inner_transition",
    "inner_cutout",
}

CHAINABLE_TYPES = {"LINE", "ARC", "SPLINE"}
DRAWABLE_TYPES = CHAINABLE_TYPES | {"CIRCLE"}
CATEGORY_ALLOWED_TYPES = {
    "upper_edge": CHAINABLE_TYPES,
    "lower_edge": CHAINABLE_TYPES,
    "outer_fixture": DRAWABLE_TYPES,
    "inner_transition": CHAINABLE_TYPES,
    "inner_cutout": CHAINABLE_TYPES,
}
MANUAL_KERF_CATEGORIES = {"outer_fixture", "inner_transition", "inner_cutout"}


def category_label(cat: str) -> str:
    return CATEGORY_META[cat]["label"]


@dataclass
class AssignmentItem:
    idx: int
    reverse: bool = False
    kerf_flip: bool = False

    def to_json(self):
        return {"idx": self.idx, "reverse": self.reverse, "kerf_flip": self.kerf_flip}

    @staticmethod
    def from_json(obj):
        return AssignmentItem(
            idx=int(obj["idx"]),
            reverse=bool(obj.get("reverse", False)),
            kerf_flip=bool(obj.get("kerf_flip", False)),
        )


# =========================
# DXF parsing / sampling
# =========================

def parse_entities_from_dxf_text(text: str):
    lines = text.splitlines()

    start = end = None
    for i in range(len(lines) - 3):
        if (
            lines[i].strip() == "0"
            and lines[i + 1].strip() == "SECTION"
            and lines[i + 2].strip() == "2"
            and lines[i + 3].strip() == "ENTITIES"
        ):
            start = i + 4
            break

    if start is None:
        raise ValueError("Could not find ENTITIES section in DXF.")

    for j in range(start, len(lines) - 1):
        if lines[j].strip() == "0" and lines[j + 1].strip() == "ENDSEC":
            end = j
            break

    if end is None:
        raise ValueError("Could not find ENDSEC for ENTITIES section.")

    ent_lines = lines[start:end]

    entities = []
    i = 0
    while i < len(ent_lines) - 1:
        if ent_lines[i].strip() != "0":
            i += 1
            continue

        typ = ent_lines[i + 1].strip()
        i += 2
        pairs = []
        while i < len(ent_lines) - 1 and ent_lines[i].strip() != "0":
            pairs.append((ent_lines[i].strip(), ent_lines[i + 1].strip()))
            i += 2
        entities.append((typ, pairs))

    return entities


def pairdict(pairs):
    d = {}
    for k, v in pairs:
        d.setdefault(k, []).append(v)
    return d


def sample_entity(entities, idx: int, reverse: bool = False, n_arc: int = 120):
    typ, pairs = entities[idx]
    d = pairdict(pairs)

    if typ == "LINE":
        pts = np.array([
            [float(d["10"][0]), float(d["20"][0])],
            [float(d["11"][0]), float(d["21"][0])],
        ], dtype=float)

    elif typ == "ARC":
        cx, cy = float(d["10"][0]), float(d["20"][0])
        r = float(d["40"][0])
        a1, a2 = float(d["50"][0]), float(d["51"][0])
        if a2 < a1:
            a2 += 360.0
        th = np.linspace(math.radians(a1), math.radians(a2), n_arc)
        pts = np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th)])

    elif typ == "SPLINE":
        xs = [float(x) for x in d.get("10", [])]
        ys = [float(y) for y in d.get("20", [])]
        pts = np.column_stack([xs, ys]).astype(float)
        if len(pts) < 2:
            raise ValueError(f"SPLINE entity {idx} does not contain enough sampled points.")

    elif typ == "CIRCLE":
        cx, cy = float(d["10"][0]), float(d["20"][0])
        r = float(d["40"][0])
        th = np.linspace(0.0, 2.0 * math.pi, 240)
        pts = np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th)])

    else:
        raise ValueError(f"Unsupported entity type: {typ}")

    if reverse:
        pts = pts[::-1].copy()
    return pts


def cumulative_lengths(pts: np.ndarray) -> np.ndarray:
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def slice_polyline_by_fraction(pts: np.ndarray, u0: float, u1: float) -> np.ndarray:
    s = cumulative_lengths(pts)
    L = s[-1]
    a, b = u0 * L, u1 * L

    xs = [np.interp(a, s, pts[:, 0])]
    ys = [np.interp(a, s, pts[:, 1])]

    mask = (s > a) & (s < b)
    xs.extend(pts[mask, 0].tolist())
    ys.extend(pts[mask, 1].tolist())

    xs.append(np.interp(b, s, pts[:, 0]))
    ys.append(np.interp(b, s, pts[:, 1]))

    frag = np.column_stack([xs, ys])

    keep = [0]
    for i in range(1, len(frag)):
        if np.linalg.norm(frag[i] - frag[keep[-1]]) > 1e-8:
            keep.append(i)

    return frag[keep]


def path_from_assignment_list(entities, items: List[AssignmentItem]) -> np.ndarray:
    pieces = []
    for i, item in enumerate(items):
        pts = sample_entity(entities, item.idx, reverse=item.reverse)
        if i == 0:
            pieces.append(pts)
        else:
            prev_end = pieces[-1][-1]
            if np.linalg.norm(prev_end - pts[0]) < 1e-7:
                pieces.append(pts[1:])
            else:
                pieces.append(pts)
    return np.vstack(pieces)


def paths_from_assignment_list(entities, items: List[AssignmentItem], join_tol: float = 1e-6) -> List[np.ndarray]:
    paths = []
    current = None

    for item in items:
        pts = sample_entity(entities, item.idx, reverse=item.reverse)
        if current is None:
            current = pts.copy()
            continue

        if np.linalg.norm(current[-1] - pts[0]) < join_tol:
            current = np.vstack([current, pts[1:]])
        else:
            paths.append(current)
            current = pts.copy()

    if current is not None:
        paths.append(current)

    return paths


def geometry_points_from_assignments(entities, items: List[AssignmentItem]) -> np.ndarray:
    clouds = []
    for item in items:
        try:
            clouds.append(sample_entity(entities, item.idx, reverse=item.reverse))
        except Exception:
            continue
    if not clouds:
        return np.empty((0, 2), dtype=float)
    return np.vstack(clouds)


def geometry_points_from_circles(circles: List[Tuple[float, float, float]], n: int = 120) -> np.ndarray:
    clouds = []
    for cx, cy, r in circles:
        th = np.linspace(0.0, 2.0 * math.pi, n)
        clouds.append(np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th)]))
    if not clouds:
        return np.empty((0, 2), dtype=float)
    return np.vstack(clouds)


def point_cloud_distance(point: np.ndarray, cloud: np.ndarray) -> float:
    if len(cloud) == 0:
        return float("inf")
    return float(np.min(np.linalg.norm(cloud - point, axis=1)))


def path_runs_inner_to_outer(path_pts: np.ndarray, inner_cloud: np.ndarray, outer_cloud: np.ndarray) -> bool:
    start = path_pts[0]
    end = path_pts[-1]

    natural_score = 0.0
    reversed_score = 0.0
    used_anchor = False

    if len(inner_cloud) > 0:
        natural_score += point_cloud_distance(start, inner_cloud)
        reversed_score += point_cloud_distance(end, inner_cloud)
        used_anchor = True

    if len(outer_cloud) > 0:
        natural_score += point_cloud_distance(end, outer_cloud)
        reversed_score += point_cloud_distance(start, outer_cloud)
        used_anchor = True

    if not used_anchor:
        return False

    return natural_score <= reversed_score


# =========================
# Output DXF writing
# =========================

def dxf_header(layers: List[str]) -> str:
    unique_layers = ["0"]
    for layer in layers:
        if layer not in unique_layers:
            unique_layers.append(layer)

    out = [
        "0", "SECTION",
        "2", "HEADER",
        "9", "$ACADVER",
        "1", "AC1009",
        "0", "ENDSEC",
        "0", "SECTION",
        "2", "TABLES",
        "0", "TABLE",
        "2", "LTYPE",
        "70", "1",
        "0", "LTYPE",
        "2", "CONTINUOUS",
        "70", "64",
        "3", "Solid line",
        "72", "65",
        "73", "0",
        "40", "0.0",
        "0", "ENDTAB",
        "0", "TABLE",
        "2", "LAYER",
        "70", str(len(unique_layers)),
    ]

    for i, layer in enumerate(unique_layers):
        color = 7 if i == 0 else ((i % 255) or 7)
        out.extend([
            "0", "LAYER",
            "2", layer,
            "70", "0",
            "62", str(color),
            "6", "CONTINUOUS",
        ])

    out.extend([
        "0", "ENDTAB",
        "0", "ENDSEC",
        "0", "SECTION",
        "2", "BLOCKS",
        "0", "ENDSEC",
        "0", "SECTION",
        "2", "ENTITIES",
    ])
    return "\n".join(out) + "\n"


def dxf_footer() -> str:
    return "\n".join(["0", "ENDSEC", "0", "EOF"]) + "\n"


def polyline_entity(points: np.ndarray, layer: str) -> str:
    out = [
        "0", "POLYLINE",
        "8", layer,
        "10", "0.0",
        "20", "0.0",
        "30", "0.0",
        "66", "1",
        "70", "0",
    ]
    for x, y in points:
        out += [
            "0", "VERTEX",
            "8", layer,
            "10", f"{x:.9f}",
            "20", f"{y:.9f}",
            "30", "0.0",
            "70", "0",
        ]
    out += ["0", "SEQEND"]
    return "\n".join(out) + "\n"


def circle_entity(cx: float, cy: float, r: float, layer: str) -> str:
    return "\n".join([
        "0", "CIRCLE",
        "8", layer,
        "10", f"{cx:.9f}",
        "20", f"{cy:.9f}",
        "30", "0.0",
        "40", f"{r:.9f}",
    ]) + "\n"


def build_dxf_content(
    polylines: List[Tuple[np.ndarray, str]],
    circles: Optional[List[Tuple[float, float, float, str]]] = None,
) -> str:
    layers = [layer for _, layer in polylines]
    layers.extend(layer for *_rest, layer in (circles or []))
    content = dxf_header(layers)
    for pts, layer in polylines:
        content += polyline_entity(pts, layer)
    for cx, cy, r, layer in circles or []:
        content += circle_entity(cx, cy, r, layer)
    content += dxf_footer()
    return content


def bbox_from_export_geometry(
    polylines: List[Tuple[np.ndarray, str]],
    circles: Optional[List[Tuple[float, float, float, str]]] = None,
) -> Optional[Tuple[float, float, float, float]]:
    xmins = []
    ymins = []
    xmaxs = []
    ymaxs = []

    for pts, _layer in polylines:
        if len(pts) == 0:
            continue
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        xmins.append(float(mins[0]))
        ymins.append(float(mins[1]))
        xmaxs.append(float(maxs[0]))
        ymaxs.append(float(maxs[1]))

    for cx, cy, r, _layer in circles or []:
        xmins.append(float(cx - r))
        ymins.append(float(cy - r))
        xmaxs.append(float(cx + r))
        ymaxs.append(float(cy + r))

    if not xmins:
        return None

    return min(xmins), min(ymins), max(xmaxs), max(ymaxs)


def union_bbox(bboxes: List[Optional[Tuple[float, float, float, float]]]) -> Optional[Tuple[float, float, float, float]]:
    valid = [bbox for bbox in bboxes if bbox is not None]
    if not valid:
        return None

    return (
        min(bbox[0] for bbox in valid),
        min(bbox[1] for bbox in valid),
        max(bbox[2] for bbox in valid),
        max(bbox[3] for bbox in valid),
    )


def dedupe_consecutive_points(pts: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    if len(pts) <= 1:
        return pts.copy()

    keep = [0]
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - pts[keep[-1]]) > tol:
            keep.append(i)
    return pts[keep].copy()


def normalize_vector(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([0.0, 0.0], dtype=float)
    return v / n


def cross2d(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def line_intersection(p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray) -> Optional[np.ndarray]:
    denom = cross2d(d1, d2)
    if abs(denom) < 1e-12:
        return None
    t = cross2d(p2 - p1, d2) / denom
    return p1 + t * d1


def segment_intersection(
    p0: np.ndarray,
    p1: np.ndarray,
    q0: np.ndarray,
    q1: np.ndarray,
    tol: float = 1e-12,
) -> Optional[Tuple[np.ndarray, float, float]]:
    r = p1 - p0
    s = q1 - q0
    denom = cross2d(r, s)
    if abs(denom) < tol:
        return None

    qp = q0 - p0
    t = cross2d(qp, s) / denom
    u = cross2d(qp, r) / denom
    if -tol <= t <= 1.0 + tol and -tol <= u <= 1.0 + tol:
        return p0 + t * r, float(t), float(u)
    return None


def polyline_vertex_tangents(pts: np.ndarray, closed: bool = False) -> np.ndarray:
    pts = dedupe_consecutive_points(pts)
    if len(pts) < 2:
        return np.zeros((len(pts), 2), dtype=float)

    tangents = np.zeros_like(pts, dtype=float)
    n = len(pts)
    for i in range(n):
        if closed:
            prev_pt = pts[(i - 1) % n]
            next_pt = pts[(i + 1) % n]
        elif i == 0:
            prev_pt = pts[i]
            next_pt = pts[i + 1]
        elif i == n - 1:
            prev_pt = pts[i - 1]
            next_pt = pts[i]
        else:
            prev_pt = pts[i - 1]
            next_pt = pts[i + 1]
        tangents[i] = normalize_vector(next_pt - prev_pt)

    return tangents


def offset_vertex_join(
    base_point: np.ndarray,
    prev_dir: np.ndarray,
    curr_dir: np.ndarray,
    prev_normal: np.ndarray,
    curr_normal: np.ndarray,
    distance: float,
    miter_limit: float = 8.0,
) -> np.ndarray:
    prev_line_point = base_point + distance * prev_normal
    curr_line_point = base_point + distance * curr_normal
    inter = line_intersection(prev_line_point, prev_dir, curr_line_point, curr_dir)
    if inter is not None:
        miter_len = float(np.linalg.norm(inter - base_point))
        if miter_len <= max(distance * miter_limit, distance + 1e-9):
            return inter

    avg_normal = normalize_vector(prev_normal + curr_normal)
    if np.linalg.norm(avg_normal) < 1e-12:
        avg_normal = curr_normal
    return base_point + distance * avg_normal


def trim_open_polyline_self_intersections(
    points: np.ndarray,
    tol: float = 1e-7,
) -> np.ndarray:
    pts = dedupe_consecutive_points(points, tol=tol)
    if len(pts) < 4:
        return pts

    stack = [pts[0], pts[1]]
    for point in pts[2:]:
        stack.append(point)

        while len(stack) >= 4:
            found = False
            seg_start = stack[-2]
            seg_end = stack[-1]

            for i in range(len(stack) - 3):
                inter = segment_intersection(
                    stack[i],
                    stack[i + 1],
                    seg_start,
                    seg_end,
                    tol=tol,
                )
                if inter is None:
                    continue

                inter_point, _t, _u = inter
                prefix = [p.copy() for p in stack[:i + 1]]
                if np.linalg.norm(prefix[-1] - inter_point) > tol:
                    prefix.append(inter_point)
                if np.linalg.norm(inter_point - seg_end) > tol:
                    prefix.append(seg_end)
                stack = prefix
                found = True
                break

            if found:
                continue

            end_point = stack[-1]
            for i in range(len(stack) - 3):
                if np.linalg.norm(stack[i] - end_point) <= tol:
                    stack = [p.copy() for p in stack[:i + 1]]
                    found = True
                    break

            if not found:
                break

    return dedupe_consecutive_points(np.asarray(stack, dtype=float), tol=tol)


def offset_polyline(points: np.ndarray, distance: float, sign: float, closed: bool = False) -> np.ndarray:
    pts = dedupe_consecutive_points(points)
    if len(pts) < 2 or abs(distance) < 1e-12:
        return pts.copy()

    is_closed = closed or np.linalg.norm(pts[0] - pts[-1]) < 1e-7
    base = pts[:-1].copy() if is_closed else pts.copy()
    if len(base) < 2:
        return pts.copy()

    seg_dirs = []
    seg_normals = []
    segment_count = len(base) if is_closed else len(base) - 1
    for i in range(segment_count):
        p0 = base[i]
        p1 = base[(i + 1) % len(base)]
        d = normalize_vector(p1 - p0)
        if np.linalg.norm(d) < 1e-12:
            continue
        seg_dirs.append(d)
        seg_normals.append(sign * np.array([-d[1], d[0]], dtype=float))

    if not seg_dirs:
        return pts.copy()

    if is_closed:
        out = []
        n = len(base)
        for i in range(n):
            prev_idx = (i - 1) % n
            curr_idx = i % n
            out.append(
                offset_vertex_join(
                    base[i],
                    seg_dirs[prev_idx],
                    seg_dirs[curr_idx],
                    seg_normals[prev_idx],
                    seg_normals[curr_idx],
                    distance,
                )
            )
        out = np.asarray(out, dtype=float)
        return np.vstack([out, out[0]])

    out = [base[0] + distance * seg_normals[0]]
    for i in range(1, len(base) - 1):
        out.append(
            offset_vertex_join(
                base[i],
                seg_dirs[i - 1],
                seg_dirs[i],
                seg_normals[i - 1],
                seg_normals[i],
                distance,
            )
        )
    out.append(base[-1] + distance * seg_normals[-1])
    out = np.asarray(out, dtype=float)
    # Tight inward offsets can create loops at small fillets; trim those so the exported path stays usable.
    return trim_open_polyline_self_intersections(out, tol=max(distance * 1e-3, 1e-7))


def estimate_offset_sign_away_from_reference(path_pts: np.ndarray, reference_cloud: np.ndarray) -> float:
    pts = dedupe_consecutive_points(path_pts)
    if len(pts) < 2 or len(reference_cloud) == 0:
        return 1.0

    tangents = polyline_vertex_tangents(pts)
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    sample_count = min(len(pts), 31)
    sample_indices = np.unique(np.linspace(0, len(pts) - 1, sample_count).astype(int))

    dots = []
    for idx in sample_indices:
        point = pts[idx]
        deltas = reference_cloud - point
        if len(deltas) == 0:
            continue
        nearest_idx = int(np.argmin(np.einsum("ij,ij->i", deltas, deltas)))
        nearest_vec = deltas[nearest_idx]
        if np.linalg.norm(nearest_vec) < 1e-9:
            continue
        dot = float(np.dot(normals[idx], nearest_vec))
        if abs(dot) > 1e-9:
            dots.append(dot)

    if not dots:
        return 1.0

    return -1.0 if float(np.median(dots)) > 0.0 else 1.0


def smooth_isolated_sign_outliers(signs: List[float]) -> List[float]:
    if len(signs) < 3:
        return list(signs)

    smoothed = list(signs)
    radius = 2
    for i in range(len(signs)):
        lo = max(0, i - radius)
        hi = min(len(signs), i + radius + 1)
        window = signs[lo:hi]
        pos = sum(1 for sign in window if sign > 0.0)
        neg = sum(1 for sign in window if sign < 0.0)
        if pos == neg:
            continue
        majority = 1.0 if pos > neg else -1.0
        if signs[i] != majority:
            smoothed[i] = majority

    return smoothed


def offset_polyline_away_from_reference(
    path_pts: np.ndarray,
    reference_cloud: np.ndarray,
    distance: float,
    sign_override: Optional[float] = None,
) -> np.ndarray:
    if distance <= 0.0:
        return path_pts.copy()
    sign = sign_override if sign_override is not None else estimate_offset_sign_away_from_reference(path_pts, reference_cloud)
    is_closed = np.linalg.norm(path_pts[0] - path_pts[-1]) < 1e-7
    return offset_polyline(path_pts, distance, sign=sign, closed=is_closed)


def offset_paths_away_from_reference(
    paths: List[np.ndarray],
    reference_cloud: np.ndarray,
    distance: float,
    smooth_sign_outliers: bool = False,
) -> List[np.ndarray]:
    if distance <= 0.0:
        return [pts.copy() for pts in paths]

    signs = [estimate_offset_sign_away_from_reference(pts, reference_cloud) for pts in paths]
    if smooth_sign_outliers:
        signs = smooth_isolated_sign_outliers(signs)

    return [
        offset_polyline_away_from_reference(pts, reference_cloud, distance, sign_override=sign)
        for pts, sign in zip(paths, signs)
    ]


@dataclass
class MaterialMask:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    scale: float
    blocked: np.ndarray
    material: np.ndarray
    pixel_size: float
    brush_radius_px: int

    def point_to_rc(self, point: np.ndarray) -> Optional[Tuple[int, int]]:
        col = int(round((float(point[0]) - self.xmin) * self.scale))
        row = int(round((float(point[1]) - self.ymin) * self.scale))
        if row < 0 or row >= self.blocked.shape[0] or col < 0 or col >= self.blocked.shape[1]:
            return None
        return row, col


def sample_polyline_equal_arclength(path_pts: np.ndarray, n: int) -> np.ndarray:
    pts = dedupe_consecutive_points(path_pts)
    if len(pts) <= 1 or n <= 1:
        return pts.copy()

    s = cumulative_lengths(pts)
    if s[-1] < 1e-12:
        return np.repeat(pts[:1], n, axis=0)

    u = np.linspace(0.0, s[-1], n)
    x = np.interp(u, s, pts[:, 0])
    y = np.interp(u, s, pts[:, 1])
    return np.column_stack([x, y])


def build_strip_midline_cloud(upper_edge: np.ndarray, lower_edge: np.ndarray, n: int = 500) -> np.ndarray:
    upper = sample_polyline_equal_arclength(upper_edge, n)
    lower = sample_polyline_equal_arclength(lower_edge, n)
    count = min(len(upper), len(lower))
    if count == 0:
        return np.empty((0, 2), dtype=float)
    return 0.5 * (upper[:count] + lower[:count])


def _mark_disk(mask: np.ndarray, row: int, col: int, radius: int):
    r0 = max(0, row - radius)
    r1 = min(mask.shape[0] - 1, row + radius)
    c0 = max(0, col - radius)
    c1 = min(mask.shape[1] - 1, col + radius)
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            if (rr - row) ** 2 + (cc - col) ** 2 <= radius ** 2:
                mask[rr, cc] = True


def _draw_segment_on_mask(mask: np.ndarray, p0_rc: np.ndarray, p1_rc: np.ndarray, radius: int):
    delta = p1_rc - p0_rc
    steps = max(int(math.ceil(np.max(np.abs(delta)) * 2.0)), 1)
    for t in np.linspace(0.0, 1.0, steps + 1):
        rc = p0_rc + t * delta
        _mark_disk(mask, int(round(rc[1])), int(round(rc[0])), radius)


def build_material_mask(
    boundary_paths: List[np.ndarray],
    seed_paths: List[np.ndarray],
    join_tolerance: float,
    max_dim_px: int = 1600,
    brush_radius_px: int = 2,
) -> Optional[MaterialMask]:
    clouds = [pts for pts in boundary_paths if len(pts) > 0]
    if not clouds:
        return None

    stack = np.vstack(clouds)
    xmin, ymin = stack.min(axis=0)
    xmax, ymax = stack.max(axis=0)
    dx = xmax - xmin
    dy = ymax - ymin
    pad = max(join_tolerance * 2.0, 0.05 * max(dx, dy, 1.0), 1.0)
    xmin -= pad
    ymin -= pad
    xmax += pad
    ymax += pad
    width = max(xmax - xmin, 1.0)
    height = max(ymax - ymin, 1.0)
    scale = (max_dim_px - 1) / max(width, height)
    cols = int(math.ceil(width * scale)) + 1
    rows = int(math.ceil(height * scale)) + 1

    blocked = np.zeros((rows, cols), dtype=bool)

    def to_xy_rc(points: np.ndarray) -> np.ndarray:
        x = (points[:, 0] - xmin) * scale
        y = (points[:, 1] - ymin) * scale
        return np.column_stack([x, y])

    for pts in boundary_paths:
        if len(pts) < 2:
            continue
        rc = to_xy_rc(dedupe_consecutive_points(pts))
        for i in range(len(rc) - 1):
            _draw_segment_on_mask(blocked, rc[i], rc[i + 1], brush_radius_px)

    endpoints = []
    for pts in boundary_paths:
        if len(pts) < 2:
            continue
        closed = np.linalg.norm(pts[0] - pts[-1]) < max(join_tolerance, 1e-7)
        if closed:
            continue
        endpoints.append(pts[0])
        endpoints.append(pts[-1])

    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            if np.linalg.norm(endpoints[i] - endpoints[j]) <= join_tolerance:
                rc = to_xy_rc(np.vstack([endpoints[i], endpoints[j]]))
                _draw_segment_on_mask(blocked, rc[0], rc[1], brush_radius_px)

    material = np.zeros_like(blocked, dtype=bool)
    queue = deque()

    def enqueue_seed(point: np.ndarray):
        col = int(round((float(point[0]) - xmin) * scale))
        row = int(round((float(point[1]) - ymin) * scale))
        for radius in range(0, 5):
            for rr in range(max(0, row - radius), min(rows - 1, row + radius) + 1):
                for cc in range(max(0, col - radius), min(cols - 1, col + radius) + 1):
                    if blocked[rr, cc] or material[rr, cc]:
                        continue
                    material[rr, cc] = True
                    queue.append((rr, cc))
                    return

    for seed_path in seed_paths:
        for point in seed_path:
            enqueue_seed(point)

    if not queue:
        return None

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    while queue:
        row, col = queue.popleft()
        for dr, dc in neighbors:
            rr = row + dr
            cc = col + dc
            if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                continue
            if blocked[rr, cc] or material[rr, cc]:
                continue
            material[rr, cc] = True
            queue.append((rr, cc))

    return MaterialMask(
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
        scale=scale,
        blocked=blocked,
        material=material,
        pixel_size=1.0 / scale,
        brush_radius_px=brush_radius_px,
    )


def _probe_material_side(mask: MaterialMask, point: np.ndarray, normal: np.ndarray, base_probe: float) -> Optional[bool]:
    for mul in (1.0, 2.0, 3.0):
        probe_point = point + mul * base_probe * normal
        rc = mask.point_to_rc(probe_point)
        if rc is None:
            continue
        row, col = rc
        if mask.blocked[row, col]:
            continue
        return bool(mask.material[row, col])
    return None


def estimate_offset_sign_from_material_mask(path_pts: np.ndarray, material_mask: MaterialMask) -> Optional[float]:
    pts = sample_polyline_equal_arclength(path_pts, min(max(len(path_pts), 21), 61))
    if len(pts) < 2:
        return None

    tangents = polyline_vertex_tangents(pts)
    normals = np.column_stack([-tangents[:, 1], tangents[:, 0]])
    base_probe = max(material_mask.pixel_size * (material_mask.brush_radius_px + 3), 0.15)

    votes = []
    for point, normal in zip(pts, normals):
        if np.linalg.norm(normal) < 1e-12:
            continue
        left_is_material = _probe_material_side(material_mask, point, normal, base_probe)
        right_is_material = _probe_material_side(material_mask, point, -normal, base_probe)
        if left_is_material is None or right_is_material is None:
            continue
        if left_is_material == right_is_material:
            continue
        votes.append(-1.0 if left_is_material else 1.0)

    if not votes:
        return None

    pos = sum(1 for vote in votes if vote > 0.0)
    neg = sum(1 for vote in votes if vote < 0.0)
    return 1.0 if pos > neg else -1.0


def offset_paths_using_material_mask(
    paths: List[np.ndarray],
    material_mask: Optional[MaterialMask],
    distance: float,
    fallback_reference_cloud: Optional[np.ndarray] = None,
    fallback_spatial_neighbor_count: int = 0,
) -> List[np.ndarray]:
    signs = []
    used_fallback = False
    for pts in paths:
        sign = None if material_mask is None else estimate_offset_sign_from_material_mask(pts, material_mask)
        if sign is None and fallback_reference_cloud is not None and len(fallback_reference_cloud) > 0:
            sign = estimate_offset_sign_away_from_reference(pts, fallback_reference_cloud)
            used_fallback = True
        if sign is None:
            sign = 1.0
        signs.append(sign)

    if used_fallback and fallback_spatial_neighbor_count > 0 and len(paths) > 1:
        signs = smooth_signs_by_spatial_neighbors(paths, signs, fallback_spatial_neighbor_count)

    out = []
    for pts, sign in zip(paths, signs):
        out.append(offset_polyline_away_from_reference(pts, np.empty((0, 2), dtype=float), distance, sign_override=sign))
    return out


def merge_connected_paths(paths: List[np.ndarray], join_tolerance: float) -> np.ndarray:
    if not paths:
        return np.empty((0, 2), dtype=float)

    merged = dedupe_consecutive_points(paths[0])
    for pts in paths[1:]:
        next_pts = dedupe_consecutive_points(pts)
        if len(merged) == 0:
            merged = next_pts
            continue
        if len(next_pts) == 0:
            continue

        if np.linalg.norm(merged[-1] - next_pts[0]) <= join_tolerance:
            joint = 0.5 * (merged[-1] + next_pts[0])
            merged[-1] = joint
            next_pts = next_pts.copy()
            next_pts[0] = joint
            merged = np.vstack([merged, next_pts[1:]])
        else:
            merged = np.vstack([merged, next_pts])

    if len(merged) > 2 and np.linalg.norm(merged[0] - merged[-1]) <= join_tolerance:
        merged[-1] = merged[0]

    return dedupe_consecutive_points(merged)


def effective_assignment_offset_signs(
    paths: List[np.ndarray],
    items: List[AssignmentItem],
    material_mask: Optional[MaterialMask],
    fallback_reference_cloud: Optional[np.ndarray] = None,
    fallback_spatial_neighbor_count: int = 0,
) -> List[float]:
    signs = []
    used_fallback = False
    for pts in paths:
        sign = None if material_mask is None else estimate_offset_sign_from_material_mask(pts, material_mask)
        if sign is None and fallback_reference_cloud is not None and len(fallback_reference_cloud) > 0:
            sign = estimate_offset_sign_away_from_reference(pts, fallback_reference_cloud)
            used_fallback = True
        if sign is None:
            sign = 1.0
        signs.append(sign)

    if used_fallback and fallback_spatial_neighbor_count > 0 and len(paths) > 1:
        signs = smooth_signs_by_spatial_neighbors(paths, signs, fallback_spatial_neighbor_count)

    return [(-sign if item.kerf_flip else sign) for item, sign in zip(items, signs)]


def cluster_endpoint_nodes(
    paths: List[np.ndarray],
    join_tolerance: float,
) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
    node_points: List[np.ndarray] = []
    node_members: List[List[np.ndarray]] = []
    path_nodes: List[Tuple[int, int]] = []

    def assign_node(point: np.ndarray) -> int:
        for idx, center in enumerate(node_points):
            if np.linalg.norm(point - center) <= join_tolerance:
                node_members[idx].append(point)
                node_points[idx] = np.mean(np.asarray(node_members[idx]), axis=0)
                return idx

        node_points.append(point.copy())
        node_members.append([point.copy()])
        return len(node_points) - 1

    for pts in paths:
        start_node = assign_node(pts[0])
        end_node = assign_node(pts[-1])
        path_nodes.append((start_node, end_node))

    return node_points, path_nodes


def connected_path_runs_by_sign(
    paths: List[np.ndarray],
    signs: List[float],
    join_tolerance: float,
) -> List[Tuple[List[np.ndarray], float]]:
    if not paths:
        return []

    runs: List[Tuple[List[np.ndarray], float]] = []
    by_sign: Dict[float, List[int]] = defaultdict(list)
    for idx, sign in enumerate(signs):
        by_sign[sign].append(idx)

    for run_sign, sign_indices in by_sign.items():
        sign_paths = [paths[idx] for idx in sign_indices]
        _node_points, path_nodes = cluster_endpoint_nodes(sign_paths, join_tolerance)

        adjacency: Dict[int, List[int]] = defaultdict(list)
        degree: Dict[int, int] = defaultdict(int)
        for local_idx, (node_a, node_b) in enumerate(path_nodes):
            adjacency[node_a].append(local_idx)
            adjacency[node_b].append(local_idx)
            degree[node_a] += 1
            degree[node_b] += 1

        visited = set()

        def walk(start_edge: int, start_node: int) -> List[np.ndarray]:
            run_paths: List[np.ndarray] = []
            edge_idx = start_edge
            current_node = start_node

            while True:
                if edge_idx in visited:
                    break

                visited.add(edge_idx)
                node_a, node_b = path_nodes[edge_idx]
                pts = sign_paths[edge_idx]

                if current_node == node_a:
                    oriented = pts
                    next_node = node_b
                elif current_node == node_b:
                    oriented = pts[::-1].copy()
                    next_node = node_a
                else:
                    break

                run_paths.append(oriented)
                candidates = [idx for idx in adjacency[next_node] if idx not in visited]
                if degree[next_node] != 2 or len(candidates) != 1:
                    break

                edge_idx = candidates[0]
                current_node = next_node

            return run_paths

        endpoint_nodes = sorted([node for node, node_degree in degree.items() if node_degree != 2])
        for node in endpoint_nodes:
            for edge_idx in adjacency[node]:
                if edge_idx in visited:
                    continue
                run_paths = walk(edge_idx, node)
                if run_paths:
                    runs.append((run_paths, run_sign))

        for edge_idx, (node_a, _node_b) in enumerate(path_nodes):
            if edge_idx in visited:
                continue
            run_paths = walk(edge_idx, node_a)
            if run_paths:
                runs.append((run_paths, run_sign))

    return runs


def kerf_path_join_tolerance(auto_gap: float) -> float:
    # Kerf path stitching should only snap genuinely shared endpoints.
    # Auto-chain gap can be much larger and will over-cluster tight-radius features.
    return max(1e-6, min(float(auto_gap) * 0.05, 0.005))


def smooth_signs_by_spatial_neighbors(paths: List[np.ndarray], signs: List[float], neighbor_count: int) -> List[float]:
    if len(paths) <= 1 or neighbor_count <= 0:
        return list(signs)

    centers = np.array([pts.mean(axis=0) for pts in paths], dtype=float)
    smoothed = list(signs)
    for i, center in enumerate(centers):
        distances = np.linalg.norm(centers - center, axis=1)
        neighbor_indices = [j for j in np.argsort(distances) if j != i][:neighbor_count]
        if not neighbor_indices:
            continue
        pos = sum(1 for j in neighbor_indices if signs[j] > 0.0)
        neg = sum(1 for j in neighbor_indices if signs[j] < 0.0)
        if pos == neg:
            continue
        smoothed[i] = 1.0 if pos > neg else -1.0

    return smoothed


def offset_assignment_items_using_material_mask(
    entities,
    items: List[AssignmentItem],
    distance: float,
    material_mask: Optional[MaterialMask],
    fallback_reference_cloud: Optional[np.ndarray] = None,
    fallback_spatial_neighbor_count: int = 0,
    join_tolerance: float = 0.5,
) -> List[np.ndarray]:
    if distance <= 0.0:
        return [sample_entity(entities, item.idx, reverse=item.reverse) for item in items]

    paths = [sample_entity(entities, item.idx, reverse=item.reverse) for item in items]
    final_signs = effective_assignment_offset_signs(
        paths,
        items,
        material_mask,
        fallback_reference_cloud=fallback_reference_cloud,
        fallback_spatial_neighbor_count=fallback_spatial_neighbor_count,
    )

    stitched_runs = [
        (merge_connected_paths(run_paths, join_tolerance), run_sign)
        for run_paths, run_sign in connected_path_runs_by_sign(paths, final_signs, join_tolerance)
    ]

    out = []
    for stitched_path, final_sign in stitched_runs:
        out.append(
            offset_polyline_away_from_reference(
                stitched_path,
                np.empty((0, 2), dtype=float),
                distance,
                sign_override=final_sign,
            )
        )
    return out


def compensated_circle_radius(item: AssignmentItem, radius: float, kerf_offset: float) -> float:
    if kerf_offset <= 0.0:
        return radius
    delta = kerf_offset if item.kerf_flip else -kerf_offset
    return max(radius + delta, 1e-6)


# =========================
# Direction / chaining helpers
# =========================

def safe_unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.array([0.0, 0.0])
    return v / n


def local_direction_arrow_points(pts: np.ndarray):
    if len(pts) < 2:
        return None

    start = pts[0]
    end = pts[-1]

    i2 = min(5, len(pts) - 1)
    j2 = max(len(pts) - 6, 0)

    start_dir = safe_unit(pts[i2] - start)
    end_dir = safe_unit(end - pts[j2])

    return start, start_dir, end, end_dir


def oriented_endpoints(entities, idx: int, reverse: bool) -> Tuple[np.ndarray, np.ndarray]:
    pts = sample_entity(entities, idx, reverse=reverse)
    return pts[0], pts[-1]


def all_assigned_indices(assignments: Dict[str, List[AssignmentItem]]) -> set:
    out = set()
    for items in assignments.values():
        out.update(item.idx for item in items)
    return out


def greedy_autochain(
    entities,
    seed_items: List[AssignmentItem],
    allowed_candidate_indices: List[int],
    max_gap: float,
) -> Tuple[List[AssignmentItem], List[str]]:
    """
    Greedily extend a chain from both ends based on endpoint proximity.

    For each remaining entity we consider:
      - append in forward / reverse orientation
      - prepend in forward / reverse orientation

    The best valid move under max_gap is applied repeatedly until no more
    candidates can be attached.
    """
    if len(seed_items) == 0:
        raise ValueError("greedy_autochain requires at least one seed item.")

    chain = [AssignmentItem(x.idx, x.reverse) for x in seed_items]
    remaining = [idx for idx in allowed_candidate_indices if idx not in {item.idx for item in chain}]
    log = []

    while True:
        start_pt, _ = oriented_endpoints(entities, chain[0].idx, chain[0].reverse)
        _, end_pt = oriented_endpoints(entities, chain[-1].idx, chain[-1].reverse)

        best = None
        best_gap = float("inf")

        for idx in remaining:
            for rev in (False, True):
                cand_start, cand_end = oriented_endpoints(entities, idx, rev)

                gap_append = np.linalg.norm(cand_start - end_pt)
                if gap_append < best_gap:
                    best_gap = gap_append
                    best = ("append", AssignmentItem(idx, rev), gap_append)

                gap_prepend = np.linalg.norm(cand_end - start_pt)
                if gap_prepend < best_gap:
                    best_gap = gap_prepend
                    best = ("prepend", AssignmentItem(idx, rev), gap_prepend)

        if best is None or best_gap > max_gap:
            break

        where, item, gap = best
        if where == "append":
            chain.append(item)
            log.append(f"append idx={item.idx} rev={item.reverse} gap={gap:.6g}")
        else:
            chain.insert(0, item)
            log.append(f"prepend idx={item.idx} rev={item.reverse} gap={gap:.6g}")

        remaining = [idx for idx in remaining if idx != item.idx]

    return chain, log


# =========================
# GUI app
# =========================

class DXFSplitterGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("DXF Spiral Spring Splitter")
        self.root.geometry("1600x960")
        self.root.minsize(1100, 720)

        self.project_root = PROJECT_ROOT
        self.default_dxf_dir = DEFAULT_DXF_DIR
        self.default_config_dir = DEFAULT_CONFIG_DIR
        self.default_output_dir = DEFAULT_OUTPUT_DIR
        self._ensure_workspace_dirs()

        self.dxf_path: Optional[Path] = None
        self.config_path: Optional[Path] = None
        self.entities = []
        self.assignments: Dict[str, List[AssignmentItem]] = {k: [] for k in CATEGORIES}
        self.current_category = tk.StringVar(value="upper_edge")
        self.category_help_var = tk.StringVar(value=CATEGORY_META["upper_edge"]["description"])
        self.n_sections_var = tk.IntVar(value=12)
        self.auto_gap_var = tk.DoubleVar(value=0.5)
        self.kerf_mm_var = tk.DoubleVar(value=0.0)
        self.show_kerf_overlay_var = tk.BooleanVar(value=False)
        self.show_segment_preview_var = tk.BooleanVar(value=False)
        self.show_labels_var = tk.BooleanVar(value=True)
        self.show_arrows_var = tk.BooleanVar(value=True)
        self.show_start_end_var = tk.BooleanVar(value=True)
        self.selected_entity_idx: Optional[int] = None
        self.status_var = tk.StringVar(value="Load a DXF.")

        self.figure, self.ax = plt.subplots(figsize=(9, 9))
        self.canvas: Optional[FigureCanvasTkAgg] = None
        self.nav_toolbar: Optional[NavigationToolbar2Tk] = None
        self._default_plot_limits: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None
        self.left_canvas: Optional[tk.Canvas] = None
        self.left_inner: Optional[ttk.Frame] = None
        self.left_scrollbar: Optional[ttk.Scrollbar] = None
        self.left_window_id: Optional[int] = None

        self.entity_plot_handles = {}
        self.entity_label_handles = {}

        self._build_ui()
        self.current_category.trace_add("write", self._update_category_help)
        self._bind_events()

    def _build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True)

        left_shell = ttk.Frame(main, width=420)
        left_shell.pack(side="left", fill="both", padx=6, pady=6)
        left_shell.pack_propagate(False)

        right = ttk.Frame(main)
        right.pack(side="right", fill="both", expand=True, padx=6, pady=6)

        self.left_canvas = tk.Canvas(left_shell, highlightthickness=0, borderwidth=0)
        self.left_canvas.pack(side="left", fill="both", expand=True)

        self.left_scrollbar = ttk.Scrollbar(left_shell, orient="vertical", command=self.left_canvas.yview)
        self.left_scrollbar.pack(side="right", fill="y")
        self.left_canvas.configure(yscrollcommand=self.left_scrollbar.set)

        left = ttk.Frame(self.left_canvas)
        self.left_inner = left
        self.left_window_id = self.left_canvas.create_window((0, 0), window=left, anchor="nw")
        left.bind("<Configure>", self._on_left_inner_configure)
        self.left_canvas.bind("<Configure>", self._on_left_canvas_configure)
        self.left_canvas.bind("<Enter>", self._bind_left_mousewheel)
        self.left_canvas.bind("<Leave>", self._unbind_left_mousewheel)

        plot_frame = ttk.Frame(right)
        plot_frame.pack(fill="both", expand=True)

        file_frame = ttk.LabelFrame(left, text="File")
        file_frame.pack(fill="x", pady=4)

        ttk.Button(file_frame, text="Open DXF", command=self.open_dxf).pack(fill="x", padx=4, pady=4)
        ttk.Button(file_frame, text="Save Config JSON", command=self.save_config).pack(fill="x", padx=4, pady=4)
        ttk.Button(file_frame, text="Load Config JSON", command=self.load_config).pack(fill="x", padx=4, pady=4)
        ttk.Button(file_frame, text="Generate Outputs", command=self.generate_outputs).pack(fill="x", padx=4, pady=4)

        self.file_label = ttk.Label(file_frame, text="No DXF loaded", wraplength=360)
        self.file_label.pack(fill="x", padx=4, pady=4)

        mode_frame = ttk.LabelFrame(left, text="Assign clicked entities to")
        mode_frame.pack(fill="x", pady=4)

        for cat in CATEGORIES:
            ttk.Radiobutton(
                mode_frame,
                text=category_label(cat),
                variable=self.current_category,
                value=cat
            ).pack(anchor="w", padx=4, pady=2)

        ttk.Label(
            mode_frame,
            textvariable=self.category_help_var,
            wraplength=360,
            justify="left",
        ).pack(fill="x", padx=4, pady=4)

        auto_frame = ttk.LabelFrame(left, text="Auto-chain")
        auto_frame.pack(fill="x", pady=4)

        row = ttk.Frame(auto_frame)
        row.pack(fill="x", padx=4, pady=4)
        ttk.Label(row, text="Max endpoint gap").pack(side="left")
        ttk.Entry(row, textvariable=self.auto_gap_var, width=10).pack(side="right")

        ttk.Button(
            auto_frame,
            text="Auto-chain current category",
            command=self.auto_chain_current_category
        ).pack(fill="x", padx=4, pady=4)

        ttk.Button(
            auto_frame,
            text="Seed with selected entity + auto-chain",
            command=self.seed_and_auto_chain_current_category
        ).pack(fill="x", padx=4, pady=4)

        ttk.Label(
            auto_frame,
            text=(
                "Use a small gap tolerance so the chain follows true touching endpoints.\n"
                "For most clean DXFs, 0.1 to 1.0 works well."
            ),
            wraplength=360,
            justify="left",
        ).pack(fill="x", padx=4, pady=4)

        opt_frame = ttk.LabelFrame(left, text="Display options")
        opt_frame.pack(fill="x", pady=4)

        row = ttk.Frame(opt_frame)
        row.pack(fill="x", padx=4, pady=4)
        ttk.Label(row, text="Spring segments per edge").pack(side="left")
        tk.Spinbox(row, from_=1, to=999, textvariable=self.n_sections_var, width=8).pack(side="right")

        ttk.Label(
            opt_frame,
            text="Controls how many DXFs each spring edge is split into. Default: 12.",
            wraplength=360,
            justify="left",
        ).pack(fill="x", padx=4, pady=(0, 4))

        ttk.Checkbutton(
            opt_frame,
            text="Show spring segment preview",
            variable=self.show_segment_preview_var,
            command=self.redraw_plot,
        ).pack(anchor="w", padx=4, pady=2)

        ttk.Label(
            opt_frame,
            text="Overlays the current upper/lower spring split and numbering in the plot using the current segment count.",
            wraplength=360,
            justify="left",
        ).pack(fill="x", padx=4, pady=(0, 4))

        kerf_frame = ttk.LabelFrame(left, text="Kerf compensation")
        kerf_frame.pack(fill="x", pady=4)

        row = ttk.Frame(kerf_frame)
        row.pack(fill="x", padx=4, pady=4)
        ttk.Label(row, text="Manual kerf (mm)").pack(side="left")
        tk.Spinbox(row, from_=0.0, to=5.0, increment=0.01, textvariable=self.kerf_mm_var, width=8).pack(side="right")

        ttk.Checkbutton(
            kerf_frame,
            text="Show kerf overlay in plot",
            variable=self.show_kerf_overlay_var,
            command=self.redraw_plot,
        ).pack(anchor="w", padx=4, pady=2)

        ttk.Button(
            kerf_frame,
            text="Redraw kerf overlay",
            command=self.redraw_plot,
        ).pack(fill="x", padx=4, pady=4)

        ttk.Label(
            kerf_frame,
            text=(
                "When kerf is greater than 0, exports shift all generated DXFs by kerf/2 in the compensated "
                "direction. Holes shrink, outer contours expand, and the inner-end geometry shifts toward the center opening."
            ),
            wraplength=360,
            justify="left",
        ).pack(fill="x", padx=4, pady=(0, 4))

        ttk.Checkbutton(
            opt_frame,
            text="Show entity index labels",
            variable=self.show_labels_var,
            command=self.redraw_plot,
        ).pack(anchor="w", padx=4, pady=2)

        ttk.Checkbutton(
            opt_frame,
            text="Show direction arrows",
            variable=self.show_arrows_var,
            command=self.redraw_plot,
        ).pack(anchor="w", padx=4, pady=2)

        ttk.Checkbutton(
            opt_frame,
            text="Show start/end markers",
            variable=self.show_start_end_var,
            command=self.redraw_plot,
        ).pack(anchor="w", padx=4, pady=2)

        ttk.Button(opt_frame, text="Clear all assignments", command=self.clear_assignments).pack(fill="x", padx=4, pady=4)
        ttk.Button(opt_frame, text="Redraw / auto-fit", command=lambda: self.redraw_plot(auto_fit=True)).pack(fill="x", padx=4, pady=4)

        edit_frame = ttk.LabelFrame(left, text="Assignments")
        edit_frame.pack(fill="both", expand=True, pady=4)

        self.category_tabs = ttk.Notebook(edit_frame)
        self.category_tabs.pack(fill="both", expand=True, padx=4, pady=4)

        self.listboxes = {}
        for cat in CATEGORIES:
            tab = ttk.Frame(self.category_tabs)
            self.category_tabs.add(tab, text=category_label(cat))

            lb = tk.Listbox(tab, exportselection=False, height=10)
            lb.pack(fill="both", expand=True, padx=4, pady=4)
            self.listboxes[cat] = lb

            btn_row1 = ttk.Frame(tab)
            btn_row1.pack(fill="x", padx=4, pady=2)
            ttk.Button(btn_row1, text="Toggle Reverse", command=lambda c=cat: self.toggle_reverse(c)).pack(side="left", expand=True, fill="x", padx=2)
            ttk.Button(btn_row1, text="Toggle Kerf Dir", command=lambda c=cat: self.toggle_kerf_direction(c)).pack(side="left", expand=True, fill="x", padx=2)
            ttk.Button(btn_row1, text="Remove", command=lambda c=cat: self.remove_selected_assignment(c)).pack(side="left", expand=True, fill="x", padx=2)

            btn_row2 = ttk.Frame(tab)
            btn_row2.pack(fill="x", padx=4, pady=2)
            ttk.Button(btn_row2, text="Move Up", command=lambda c=cat: self.move_assignment(c, -1)).pack(side="left", expand=True, fill="x", padx=2)
            ttk.Button(btn_row2, text="Move Down", command=lambda c=cat: self.move_assignment(c, +1)).pack(side="left", expand=True, fill="x", padx=2)

        inst = ttk.LabelFrame(left, text="How to use")
        inst.pack(fill="x", pady=4)
        ttk.Label(
            inst,
            text=(
                "Manual:\n"
                "  1. Choose a segment group\n"
                "  2. Click entities in order\n"
                "  3. Click an assigned entity again to remove a misclick\n"
                "  4. Fix direction with Toggle Reverse if needed\n"
                "  5. For fixture / transition / inner ring, use Toggle Kerf Dir if auto kerf is on the wrong side\n\n"
                "Semi-automatic:\n"
                "  1. Click one seed entity\n"
                "  2. Press 'Seed with selected entity + auto-chain'\n"
                "  3. Inspect and adjust\n\n"
                "Use the toolbar below the plot for Home / Pan / Zoom."
            ),
            justify="left",
            wraplength=360,
        ).pack(fill="x", padx=4, pady=4)

        legend = ttk.LabelFrame(left, text="Direction markers")
        legend.pack(fill="x", pady=4)
        ttk.Label(
            legend,
            text=(
                "Green dot = start\n"
                "Red dot = end\n"
                "Arrow = forward direction along the sampled path\n"
                "Outer fixture includes circular holes\n"
                "Dashed overlay = kerf-adjusted export path"
            ),
            justify="left",
            wraplength=360,
        ).pack(fill="x", padx=4, pady=4)

        status_frame = ttk.LabelFrame(left, text="Status")
        status_frame.pack(fill="x", pady=4)
        ttk.Label(status_frame, textvariable=self.status_var, wraplength=360, justify="left").pack(fill="x", padx=4, pady=4)

        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(fill="both", expand=True)

        toolbar_row = ttk.Frame(right)
        toolbar_row.pack(fill="x")
        self.nav_toolbar = NavigationToolbar2Tk(self.canvas, toolbar_row, pack_toolbar=False)
        self.nav_toolbar.update()
        self.nav_toolbar.pack(side="left", fill="x", expand=True)
        ttk.Button(toolbar_row, text="Redraw / auto-fit", command=lambda: self.redraw_plot(auto_fit=True)).pack(side="right", padx=4, pady=4)

    def _on_left_inner_configure(self, _event):
        if self.left_canvas is None:
            return
        self.left_canvas.configure(scrollregion=self.left_canvas.bbox("all"))

    def _on_left_canvas_configure(self, event):
        if self.left_canvas is None or self.left_window_id is None:
            return
        self.left_canvas.itemconfigure(self.left_window_id, width=event.width)

    def _bind_left_mousewheel(self, _event):
        self.root.bind_all("<MouseWheel>", self._on_left_mousewheel)
        self.root.bind_all("<Button-4>", self._on_left_mousewheel)
        self.root.bind_all("<Button-5>", self._on_left_mousewheel)

    def _unbind_left_mousewheel(self, _event):
        self.root.unbind_all("<MouseWheel>")
        self.root.unbind_all("<Button-4>")
        self.root.unbind_all("<Button-5>")

    def _on_left_mousewheel(self, event):
        if self.left_canvas is None:
            return
        if getattr(event, "num", None) == 4:
            step = -1
        elif getattr(event, "num", None) == 5:
            step = 1
        elif getattr(event, "delta", 0):
            step = -1 if event.delta > 0 else 1
        else:
            return
        self.left_canvas.yview_scroll(step, "units")

    def _bind_events(self):
        self.canvas.mpl_connect("pick_event", self.on_pick)

    def _update_category_help(self, *_args):
        self.category_help_var.set(CATEGORY_META[self.current_category.get()]["description"])

    def _capture_view_limits(self) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        if not self.entities:
            return None
        return (self.ax.get_xlim(), self.ax.get_ylim())

    def _ensure_workspace_dirs(self):
        self.default_dxf_dir.mkdir(parents=True, exist_ok=True)
        self.default_config_dir.mkdir(parents=True, exist_ok=True)
        self.default_output_dir.mkdir(parents=True, exist_ok=True)

    def _default_open_dir(self, preferred: Path) -> str:
        return str(preferred if preferred.exists() else self.project_root)

    def _relative_to_project(self, path: Path) -> str:
        resolved = path.resolve()
        try:
            return str(resolved.relative_to(self.project_root))
        except ValueError:
            return str(resolved)

    def _resolve_saved_path(self, raw_path: str, config_path: Path) -> Optional[Path]:
        if not raw_path:
            return None

        candidate = Path(raw_path).expanduser()
        if candidate.is_absolute():
            return candidate

        config_relative = (config_path.parent / candidate).resolve()
        if config_relative.exists():
            return config_relative

        project_relative = (self.project_root / candidate).resolve()
        if project_relative.exists():
            return project_relative

        return config_relative

    def _suggest_config_path(self) -> Path:
        stem = self.dxf_path.stem if self.dxf_path is not None else "splitter"
        return self.default_config_dir / f"{stem}_splitter_config.json"

    def _suggest_output_dir(self) -> Path:
        stem = self.dxf_path.stem if self.dxf_path is not None else "spiral_splitter_run"
        return self.default_output_dir / stem

    def _load_dxf_from_path(self, path: Path):
        text = path.read_text(errors="ignore")
        self.entities = parse_entities_from_dxf_text(text)
        self.dxf_path = path.resolve()
        self.file_label.config(text=str(self.dxf_path))

    def _split_outer_fixture_assignments(self) -> Tuple[List[AssignmentItem], List[Tuple[AssignmentItem, float, float, float]]]:
        fixture_items = []
        circles = []
        for item in self.assignments["outer_fixture"]:
            typ = self.entities[item.idx][0]
            if typ == "CIRCLE":
                d = pairdict(self.entities[item.idx][1])
                circles.append((item, float(d["10"][0]), float(d["20"][0]), float(d["40"][0])))
            else:
                fixture_items.append(item)
        return fixture_items, circles

    def _current_kerf_mm(self) -> float:
        kerf_mm = float(self.kerf_mm_var.get())
        if kerf_mm < 0.0:
            raise ValueError("Manual kerf must be >= 0.")
        return kerf_mm

    def _build_kerf_preview_geometry(self, kerf_mm: float) -> Optional[Dict[str, List]]:
        if kerf_mm <= 0.0 or not self.entities:
            return None

        if not self.assignments["upper_edge"] or not self.assignments["lower_edge"]:
            return None

        try:
            upper_edge = path_from_assignment_list(self.entities, self.assignments["upper_edge"])
            lower_edge = path_from_assignment_list(self.entities, self.assignments["lower_edge"])
        except Exception:
            return None

        offset = kerf_mm / 2.0
        preview = {
            "upper": [],
            "lower": [],
            "outer": [],
            "circles": [],
            "transition": [],
            "inner": [],
        }

        outer_fixture_items, circle_items = self._split_outer_fixture_assignments()
        outer_paths = [sample_entity(self.entities, item.idx, reverse=item.reverse) for item in outer_fixture_items]
        transition_paths = [
            sample_entity(self.entities, item.idx, reverse=item.reverse)
            for item in self.assignments["inner_transition"]
        ]
        inner_paths = [
            sample_entity(self.entities, item.idx, reverse=item.reverse)
            for item in self.assignments["inner_cutout"]
        ]

        outer_cloud = geometry_points_from_assignments(self.entities, outer_fixture_items)
        circle_cloud = geometry_points_from_circles([(cx, cy, r) for _item, cx, cy, r in circle_items])
        if len(circle_cloud) > 0:
            outer_cloud = np.vstack([outer_cloud, circle_cloud]) if len(outer_cloud) > 0 else circle_cloud
        inner_cutout_cloud = geometry_points_from_assignments(self.entities, self.assignments["inner_cutout"])

        inner_cloud = geometry_points_from_assignments(
            self.entities,
            self.assignments["inner_transition"] + self.assignments["inner_cutout"],
        )
        reference_clouds = [upper_edge, lower_edge]
        if len(inner_cloud) > 0:
            reference_clouds.append(inner_cloud)
        outer_reference_cloud = np.vstack(reference_clouds)

        circle_paths = []
        for _item, cx, cy, r in circle_items:
            th = np.linspace(0.0, 2.0 * math.pi, 240)
            circle_paths.append(np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th)]))

        midline_cloud = build_strip_midline_cloud(upper_edge, lower_edge)
        auto_gap = float(self.auto_gap_var.get())
        mask_join_tolerance = max(auto_gap, 0.5)
        path_join_tolerance = kerf_path_join_tolerance(auto_gap)
        material_mask = build_material_mask(
            [upper_edge, lower_edge] + outer_paths + transition_paths + inner_paths + circle_paths,
            [midline_cloud],
            join_tolerance=mask_join_tolerance,
        )

        preview["upper"].extend(
            offset_paths_using_material_mask([upper_edge], material_mask, offset, fallback_reference_cloud=lower_edge)
        )
        preview["lower"].extend(
            offset_paths_using_material_mask([lower_edge], material_mask, offset, fallback_reference_cloud=upper_edge)
        )
        preview["outer"].extend(
            offset_assignment_items_using_material_mask(
                self.entities,
                outer_fixture_items,
                offset,
                material_mask,
                fallback_reference_cloud=midline_cloud,
                fallback_spatial_neighbor_count=4,
                join_tolerance=path_join_tolerance,
            )
        )

        for item, cx, cy, r in circle_items:
            preview["circles"].append((cx, cy, compensated_circle_radius(item, r, offset)))

        inner_reference_clouds = [upper_edge, lower_edge]
        if len(outer_cloud) > 0:
            inner_reference_clouds.append(outer_cloud)
        inner_reference_cloud = np.vstack(inner_reference_clouds)

        preview["transition"].extend(
            offset_assignment_items_using_material_mask(
                self.entities,
                self.assignments["inner_transition"],
                offset,
                material_mask,
                fallback_reference_cloud=np.vstack([midline_cloud, inner_cutout_cloud]) if len(inner_cutout_cloud) > 0 else midline_cloud,
                fallback_spatial_neighbor_count=2,
                join_tolerance=path_join_tolerance,
            )
        )

        preview["inner"].extend(
            offset_assignment_items_using_material_mask(
                self.entities,
                self.assignments["inner_cutout"],
                offset,
                material_mask,
                fallback_reference_cloud=inner_reference_cloud,
                join_tolerance=path_join_tolerance,
            )
        )

        return preview

    def _build_spring_segment_preview_geometry(self) -> Optional[Dict[str, List[Tuple[int, np.ndarray]]]]:
        if not self.entities:
            return None

        if not self.assignments["upper_edge"] or not self.assignments["lower_edge"]:
            return None

        try:
            upper_edge = path_from_assignment_list(self.entities, self.assignments["upper_edge"])
            lower_edge = path_from_assignment_list(self.entities, self.assignments["lower_edge"])
            n_sections = int(self.n_sections_var.get())
        except Exception:
            return None

        if n_sections < 1:
            return None

        u = np.linspace(0.0, 1.0, n_sections + 1)
        upper_frags = [slice_polyline_by_fraction(upper_edge, u[i], u[i + 1]) for i in range(n_sections)]
        lower_frags = [slice_polyline_by_fraction(lower_edge, u[i], u[i + 1]) for i in range(n_sections)]

        outer_fixture_items, circle_items = self._split_outer_fixture_assignments()
        outer_cloud = geometry_points_from_assignments(self.entities, outer_fixture_items)
        circle_cloud = geometry_points_from_circles([(cx, cy, r) for _item, cx, cy, r in circle_items])
        if len(circle_cloud) > 0:
            outer_cloud = np.vstack([outer_cloud, circle_cloud]) if len(outer_cloud) > 0 else circle_cloud

        inner_cloud = geometry_points_from_assignments(
            self.entities,
            self.assignments["inner_transition"] + self.assignments["inner_cutout"],
        )

        if len(inner_cloud) > 0 or len(outer_cloud) > 0:
            upper_inner_to_outer = path_runs_inner_to_outer(upper_edge, inner_cloud, outer_cloud)
            lower_inner_to_outer = path_runs_inner_to_outer(lower_edge, inner_cloud, outer_cloud)
            upper_frags = upper_frags if upper_inner_to_outer else list(reversed(upper_frags))
            lower_frags = lower_frags if lower_inner_to_outer else list(reversed(lower_frags))

        return {
            "upper": [(seg_num, pts) for seg_num, pts in enumerate(upper_frags, start=1)],
            "lower": [(seg_num, pts) for seg_num, pts in enumerate(lower_frags, start=1)],
        }

    def refresh_assignment_lists(self):
        for cat, lb in self.listboxes.items():
            lb.delete(0, "end")
            for item in self.assignments[cat]:
                typ = self.entities[item.idx][0] if self.entities else "?"
                rev = "rev" if item.reverse else "fwd"
                kerf = "kflip" if item.kerf_flip else "kauto"
                lb.insert("end", f"{item.idx:03d} | {typ:<6} | {rev} | {kerf}")

    def clear_assignments(self):
        self.assignments = {k: [] for k in CATEGORIES}
        self.selected_entity_idx = None
        self.status_var.set("Cleared all assignments.")
        self.refresh_assignment_lists()
        self.redraw_plot()

    def open_dxf(self):
        path = filedialog.askopenfilename(
            title="Open DXF",
            initialdir=self._default_open_dir(self.default_dxf_dir),
            filetypes=[("DXF files", "*.dxf *.DXF"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            self._load_dxf_from_path(Path(path))
            self.config_path = None
            self.clear_assignments()
            drawable_count = sum(1 for typ, _ in self.entities if typ in DRAWABLE_TYPES)
            self.status_var.set(
                f"Loaded {len(self.entities)} entities ({drawable_count} drawable). "
                "Use the right-hand plot to click entities."
            )
            self.redraw_plot(auto_fit=True)
            messagebox.showinfo(
                "Loaded",
                f"Loaded {len(self.entities)} entities.\n"
                f"Drawable in plot: {drawable_count}"
            )
        except Exception as e:
            messagebox.showerror("Error loading DXF", str(e))

    def save_config(self):
        if self.dxf_path is None:
            messagebox.showwarning("No DXF", "Load a DXF first.")
            return

        suggested_path = self._suggest_config_path()
        path = filedialog.asksaveasfilename(
            title="Save Config JSON",
            defaultextension=".json",
            initialdir=str(suggested_path.parent),
            initialfile=suggested_path.name,
            filetypes=[("JSON", "*.json")]
        )
        if not path:
            return

        data = {
            "source_dxf": str(self.dxf_path),
            "source_dxf_rel": self._relative_to_project(self.dxf_path),
            "n_sections": int(self.n_sections_var.get()),
            "auto_gap": float(self.auto_gap_var.get()),
            "kerf_mm": float(self.kerf_mm_var.get()),
            "assignments": {
                cat: [item.to_json() for item in items]
                for cat, items in self.assignments.items()
            }
        }
        self.config_path = Path(path).resolve()
        self.config_path.write_text(json.dumps(data, indent=2))
        messagebox.showinfo("Saved", f"Saved config to:\n{path}")

    def load_config(self):
        path = filedialog.askopenfilename(
            title="Load Config JSON",
            initialdir=self._default_open_dir(self.default_config_dir),
            filetypes=[("JSON", "*.json"), ("All files", "*.*")]
        )
        if not path:
            return

        try:
            self.config_path = Path(path).resolve()
            obj = json.loads(self.config_path.read_text())

            source_raw = obj.get("source_dxf_rel") or obj.get("source_dxf", "")
            source_path = self._resolve_saved_path(source_raw, self.config_path)
            if source_path is not None and source_path.exists():
                if self.dxf_path is None or self.dxf_path.resolve() != source_path.resolve():
                    self._load_dxf_from_path(source_path)
            else:
                messagebox.showwarning(
                    "Load DXF first",
                    "Config loaded, but the source DXF was not found automatically.\n"
                    "Open the DXF manually if needed."
                )

            self.n_sections_var.set(int(obj.get("n_sections", 12)))
            self.auto_gap_var.set(float(obj.get("auto_gap", 0.5)))
            self.kerf_mm_var.set(float(obj.get("kerf_mm", 0.0)))
            raw_assignments = obj.get("assignments", {})
            loaded = {cat: [] for cat in CATEGORIES}
            for cat in CATEGORIES:
                loaded[cat] = [AssignmentItem.from_json(x) for x in raw_assignments.get(cat, [])]
            loaded["outer_fixture"].extend(
                AssignmentItem.from_json(x) for x in raw_assignments.get("hole_circles", [])
            )
            self.assignments = loaded

            self.refresh_assignment_lists()
            self.redraw_plot(auto_fit=True)
            self.status_var.set(f"Loaded config from {path}")
            messagebox.showinfo("Loaded", f"Loaded config from:\n{path}")
        except Exception as e:
            messagebox.showerror("Error loading config", str(e))

    def redraw_plot(self, auto_fit: bool = False):
        previous_limits = None if auto_fit else self._capture_view_limits()
        self.ax.clear()
        self.entity_plot_handles.clear()
        self.entity_label_handles.clear()
        self._default_plot_limits = None

        if not self.entities:
            self.ax.set_title("Load a DXF")
            self.canvas.draw()
            return

        assigned_lookup = {}
        for cat, items in self.assignments.items():
            for pos, item in enumerate(items):
                assigned_lookup[item.idx] = (cat, pos, item.reverse)

        colors = {
            "upper_edge": "tab:blue",
            "lower_edge": "tab:orange",
            "outer_fixture": "tab:green",
            "inner_transition": "tab:red",
            "inner_cutout": "tab:purple",
        }

        all_pts = []

        for idx, (typ, _) in enumerate(self.entities):
            try:
                pts = sample_entity(self.entities, idx, reverse=False)
            except Exception:
                continue

            all_pts.append(pts)

            if idx in assigned_lookup:
                cat, pos, rev = assigned_lookup[idx]
                color = colors[cat]
                lw = 2.6
                alpha = 1.0
                zorder = 3
            else:
                color = "0.75"
                lw = 1.0
                alpha = 0.9
                zorder = 1

            if idx == self.selected_entity_idx:
                color = "magenta"
                lw = 3.2
                zorder = 6

            line, = self.ax.plot(
                pts[:, 0], pts[:, 1],
                color=color,
                linewidth=lw,
                alpha=alpha,
                picker=5,
                zorder=zorder
            )
            line._entity_idx = idx
            self.entity_plot_handles[idx] = line

            if self.show_labels_var.get():
                mid = pts[len(pts) // 2]
                txt = self.ax.text(
                    mid[0], mid[1], str(idx),
                    fontsize=8,
                    color="black" if idx not in assigned_lookup else color,
                    ha="center", va="center",
                    zorder=7,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7)
                )
                self.entity_label_handles[idx] = txt

            arrow_data = None if typ == "CIRCLE" else local_direction_arrow_points(pts)
            if arrow_data is not None:
                start, start_dir, end, end_dir = arrow_data
                xy_scale = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]), 1.0)
                arrow_len = 0.08 * xy_scale

                if self.show_start_end_var.get():
                    self.ax.plot(start[0], start[1], marker="o", markersize=3.2, color="green", zorder=8, alpha=0.9)
                    self.ax.plot(end[0], end[1], marker="o", markersize=3.2, color="red", zorder=8, alpha=0.9)

                if self.show_arrows_var.get():
                    if np.linalg.norm(start_dir) > 0:
                        self.ax.arrow(
                            start[0], start[1],
                            arrow_len * start_dir[0], arrow_len * start_dir[1],
                            head_width=0.18 * arrow_len,
                            head_length=0.25 * arrow_len,
                            fc=color, ec=color,
                            length_includes_head=True,
                            alpha=0.85, zorder=8,
                        )
                    if np.linalg.norm(end_dir) > 0:
                        tail = end - arrow_len * end_dir
                        self.ax.arrow(
                            tail[0], tail[1],
                            arrow_len * end_dir[0], arrow_len * end_dir[1],
                            head_width=0.18 * arrow_len,
                            head_length=0.25 * arrow_len,
                            fc=color, ec=color,
                            length_includes_head=True,
                            alpha=0.65, zorder=8,
                        )

        if all_pts:
            stack = np.vstack(all_pts)
            xmin, ymin = stack.min(axis=0)
            xmax, ymax = stack.max(axis=0)
            dx = xmax - xmin
            dy = ymax - ymin
            pad = 0.05 * max(dx, dy, 1.0)
            self._default_plot_limits = (
                (xmin - pad, xmax + pad),
                (ymin - pad, ymax + pad),
            )

        segment_preview_active = False
        segment_preview_count = 0
        if self.show_segment_preview_var.get():
            try:
                segment_preview = self._build_spring_segment_preview_geometry()
            except Exception:
                segment_preview = None
            if segment_preview is not None:
                upper_segments = segment_preview["upper"]
                lower_segments = segment_preview["lower"]
                segment_preview_count = max(len(upper_segments), len(lower_segments))
                upper_cmap = plt.get_cmap("Blues")
                lower_cmap = plt.get_cmap("Oranges")
                overlay_style = dict(linewidth=3.6, alpha=0.9, zorder=5)

                def segment_color(cmap, seg_num: int, total: int):
                    frac = 0.0 if total <= 1 else (seg_num - 1) / (total - 1)
                    return cmap(0.45 + 0.45 * frac)

                for seg_num, pts in upper_segments:
                    color = segment_color(upper_cmap, seg_num, len(upper_segments))
                    self.ax.plot(pts[:, 0], pts[:, 1], color=color, **overlay_style)
                    mid = pts[len(pts) // 2]
                    self.ax.text(
                        mid[0], mid[1], f"U{seg_num}",
                        fontsize=7,
                        color=color,
                        ha="center", va="center",
                        zorder=10,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
                    )

                for seg_num, pts in lower_segments:
                    color = segment_color(lower_cmap, seg_num, len(lower_segments))
                    self.ax.plot(pts[:, 0], pts[:, 1], color=color, **overlay_style)
                    mid = pts[len(pts) // 2]
                    self.ax.text(
                        mid[0], mid[1], f"L{seg_num}",
                        fontsize=7,
                        color=color,
                        ha="center", va="center",
                        zorder=10,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8),
                    )

                segment_preview_active = True

        kerf_overlay_active = False
        kerf_mm = 0.0
        if self.show_kerf_overlay_var.get():
            try:
                kerf_mm = self._current_kerf_mm()
                preview = self._build_kerf_preview_geometry(kerf_mm)
            except Exception:
                preview = None
            if preview is not None:
                overlay_style = dict(
                    linestyle=(0, (5, 3)),
                    linewidth=1.6,
                    alpha=0.95,
                    zorder=9,
                )
                for pts in preview["upper"]:
                    self.ax.plot(pts[:, 0], pts[:, 1], color="navy", **overlay_style)
                for pts in preview["lower"]:
                    self.ax.plot(pts[:, 0], pts[:, 1], color="saddlebrown", **overlay_style)
                for pts in preview["outer"]:
                    self.ax.plot(pts[:, 0], pts[:, 1], color="darkgreen", **overlay_style)
                for cx, cy, r in preview["circles"]:
                    th = np.linspace(0.0, 2.0 * math.pi, 240)
                    self.ax.plot(cx + r * np.cos(th), cy + r * np.sin(th), color="darkgreen", **overlay_style)
                for pts in preview["transition"]:
                    self.ax.plot(pts[:, 0], pts[:, 1], color="darkred", **overlay_style)
                for pts in preview["inner"]:
                    self.ax.plot(pts[:, 0], pts[:, 1], color="indigo", **overlay_style)
                kerf_overlay_active = True

        if auto_fit or previous_limits is None:
            if self._default_plot_limits is not None:
                xlim, ylim = self._default_plot_limits
                self.ax.set_xlim(*xlim)
                self.ax.set_ylim(*ylim)
        else:
            xlim, ylim = previous_limits
            self.ax.set_xlim(*xlim)
            self.ax.set_ylim(*ylim)

        self.ax.set_aspect("equal", adjustable="box")
        title = "Click entities to assign them"
        if segment_preview_active:
            title += f" | solid overlay = spring segment preview ({segment_preview_count} per edge)"
        if kerf_overlay_active:
            title += f" | dashed overlay = exported kerf-adjusted paths ({kerf_mm:.3f} mm kerf)"
        self.ax.set_title(title)
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()

    def on_pick(self, event):
        artist = event.artist
        idx = getattr(artist, "_entity_idx", None)
        if idx is None:
            return

        self.selected_entity_idx = idx
        cat = self.current_category.get()
        typ = self.entities[idx][0]

        if typ not in CATEGORY_ALLOWED_TYPES[cat]:
            self.status_var.set(
                f"Entity {idx} has type {typ}, which does not belong in {category_label(cat)}."
            )
            self.redraw_plot()
            return

        if idx in [item.idx for item in self.assignments[cat]]:
            self.assignments[cat] = [item for item in self.assignments[cat] if item.idx != idx]
            self.refresh_assignment_lists()
            self.status_var.set(f"Removed entity {idx} from {category_label(cat)}.")
            self.redraw_plot()
            return

        if idx not in [item.idx for item in self.assignments[cat]]:
            for other_cat, items in self.assignments.items():
                if other_cat == cat:
                    continue
                if idx in [item.idx for item in items]:
                    if not messagebox.askyesno(
                        "Entity already assigned",
                        f"Entity {idx} is already assigned to '{category_label(other_cat)}'.\n"
                        f"Also add it to '{category_label(cat)}'?"
                    ):
                        self.status_var.set(f"Selected entity {idx}.")
                        self.redraw_plot()
                        return

            self.assignments[cat].append(AssignmentItem(idx=idx, reverse=False))
            self.refresh_assignment_lists()
            self.status_var.set(f"Added entity {idx} to {category_label(cat)}.")
        else:
            self.status_var.set(f"Selected entity {idx}.")

        self.redraw_plot()

    def toggle_reverse(self, cat: str):
        lb = self.listboxes[cat]
        sel = lb.curselection()
        if not sel:
            return
        i = sel[0]
        self.assignments[cat][i].reverse = not self.assignments[cat][i].reverse
        self.refresh_assignment_lists()
        lb.selection_set(i)
        self.status_var.set(f"Toggled reverse for entity {self.assignments[cat][i].idx} in {category_label(cat)}.")
        self.redraw_plot()

    def toggle_kerf_direction(self, cat: str):
        if cat not in MANUAL_KERF_CATEGORIES:
            messagebox.showwarning(
                "Not supported",
                f"Manual kerf direction overrides are only supported for {category_label('outer_fixture')}, "
                f"{category_label('inner_transition')}, and {category_label('inner_cutout')}."
            )
            return

        lb = self.listboxes[cat]
        sel = lb.curselection()
        if not sel:
            return
        i = sel[0]
        item = self.assignments[cat][i]
        item.kerf_flip = not item.kerf_flip
        self.refresh_assignment_lists()
        lb.selection_set(i)
        state = "flipped" if item.kerf_flip else "automatic"
        self.status_var.set(f"Kerf direction for entity {item.idx} in {category_label(cat)} is now {state}.")
        self.redraw_plot()

    def remove_selected_assignment(self, cat: str):
        lb = self.listboxes[cat]
        sel = lb.curselection()
        if not sel:
            return
        i = sel[0]
        removed = self.assignments[cat][i]
        del self.assignments[cat][i]
        self.refresh_assignment_lists()
        self.status_var.set(f"Removed entity {removed.idx} from {category_label(cat)}.")
        self.redraw_plot()

    def move_assignment(self, cat: str, delta: int):
        lb = self.listboxes[cat]
        sel = lb.curselection()
        if not sel:
            return
        i = sel[0]
        j = i + delta
        if j < 0 or j >= len(self.assignments[cat]):
            return
        self.assignments[cat][i], self.assignments[cat][j] = self.assignments[cat][j], self.assignments[cat][i]
        self.refresh_assignment_lists()
        lb.selection_set(j)
        self.status_var.set(f"Moved entity in {category_label(cat)}.")
        self.redraw_plot()

    def _candidate_indices_for_category(self, cat: str) -> List[int]:
        other_assigned = set()
        for other_cat, items in self.assignments.items():
            if other_cat == cat:
                continue
            other_assigned.update(item.idx for item in items)

        candidates = []
        for i, (typ, _) in enumerate(self.entities):
            if typ not in CHAINABLE_TYPES:
                continue
            if i in other_assigned:
                continue
            candidates.append(i)
        return candidates

    def auto_chain_current_category(self):
        cat = self.current_category.get()
        if cat not in CHAINABLE_CATEGORIES:
            messagebox.showwarning("Not chainable", f"{category_label(cat)} is not auto-chainable.")
            return

        if not self.entities:
            messagebox.showwarning("No DXF", "Load a DXF first.")
            return

        seed = self.assignments[cat]
        chain_seed = [item for item in seed if self.entities[item.idx][0] in CHAINABLE_TYPES]
        static_items = [item for item in seed if self.entities[item.idx][0] not in CHAINABLE_TYPES]
        if len(chain_seed) == 0:
            messagebox.showwarning(
                "Need a seed",
                "Add at least one line/arc/spline seed entity to this category,\n"
                "or use 'Seed with selected entity + auto-chain'."
            )
            return

        try:
            max_gap = float(self.auto_gap_var.get())
            candidates = self._candidate_indices_for_category(cat)
            chain, log = greedy_autochain(self.entities, chain_seed, candidates, max_gap)
            old_n = len(self.assignments[cat])
            self.assignments[cat] = chain + static_items
            self.refresh_assignment_lists()
            self.redraw_plot()
            self.status_var.set(
                f"Auto-chain finished for {category_label(cat)}: added {len(self.assignments[cat]) - old_n} entities "
                f"with max gap {max_gap}."
            )
            if log:
                print("\n".join(log))
        except Exception as e:
            messagebox.showerror("Auto-chain error", str(e))

    def seed_and_auto_chain_current_category(self):
        cat = self.current_category.get()
        if cat not in CHAINABLE_CATEGORIES:
            messagebox.showwarning("Not chainable", f"{category_label(cat)} is not auto-chainable.")
            return

        if self.selected_entity_idx is None:
            messagebox.showwarning("No selected entity", "Click an entity first.")
            return

        idx = self.selected_entity_idx
        typ = self.entities[idx][0]
        if typ not in CHAINABLE_TYPES:
            messagebox.showwarning("Wrong type", f"Entity {idx} has type {typ}, which is not chainable.")
            return

        if idx in [item.idx for item in self.assignments[cat]]:
            pass
        else:
            self.assignments[cat] = [AssignmentItem(idx=idx, reverse=False)]

        self.refresh_assignment_lists()
        self.auto_chain_current_category()

    def validate_assignments(self):
        for cat in ["upper_edge", "lower_edge"]:
            if len(self.assignments[cat]) == 0:
                raise ValueError(f"{category_label(cat)} must contain at least one entity.")

        n_sections = int(self.n_sections_var.get())
        if n_sections < 1:
            raise ValueError("Number of spring segments per edge must be >= 1.")

        self._current_kerf_mm()

    def generate_outputs(self):
        if self.dxf_path is None or not self.entities:
            messagebox.showwarning("No DXF", "Load a DXF first.")
            return

        try:
            self.validate_assignments()

            suggested_out_dir = self._suggest_output_dir()
            suggested_out_dir.mkdir(parents=True, exist_ok=True)
            out_dir = filedialog.askdirectory(
                title="Choose output folder",
                initialdir=str(suggested_out_dir),
                mustexist=True,
            )
            if not out_dir:
                return
            out_dir = Path(out_dir).resolve()

            n_sections = int(self.n_sections_var.get())
            kerf_mm = self._current_kerf_mm()
            kerf_offset = kerf_mm / 2.0

            upper_edge = path_from_assignment_list(self.entities, self.assignments["upper_edge"])
            lower_edge = path_from_assignment_list(self.entities, self.assignments["lower_edge"])

            u = np.linspace(0.0, 1.0, n_sections + 1)
            upper_frags = [slice_polyline_by_fraction(upper_edge, u[i], u[i + 1]) for i in range(n_sections)]
            lower_frags = [slice_polyline_by_fraction(lower_edge, u[i], u[i + 1]) for i in range(n_sections)]

            output_dir = out_dir / "laser_stitching_dxfs"
            output_dir.mkdir(parents=True, exist_ok=True)

            manifest_lines = [
                f"Segment numbering: 1 = arbor end, {n_sections} = outer fixture end.",
                "Placement CSV values are absolute bottom-left positions in mm after shifting the full assembled spring so its overall bottom-left sits at X=0, Y=0.",
            ]
            if kerf_mm > 0.0:
                manifest_lines.append(
                    "Manual kerf compensation: "
                    f"{kerf_mm:.3f} mm total kerf ({kerf_offset:.3f} mm offset). "
                    "Applied to all exported geometry."
                )
            output_paths = []
            export_records = []

            outer_fixture_items, circle_items = self._split_outer_fixture_assignments()
            transition_items = self.assignments["inner_transition"]
            inner_ring_items = self.assignments["inner_cutout"]
            outer_paths = [sample_entity(self.entities, item.idx, reverse=item.reverse) for item in outer_fixture_items]
            transition_paths = [sample_entity(self.entities, item.idx, reverse=item.reverse) for item in transition_items]
            inner_ring_paths = [sample_entity(self.entities, item.idx, reverse=item.reverse) for item in inner_ring_items]

            outer_cloud = geometry_points_from_assignments(self.entities, outer_fixture_items)
            circle_cloud = geometry_points_from_circles([(cx, cy, r) for _item, cx, cy, r in circle_items])
            if len(circle_cloud) > 0:
                outer_cloud = (
                    np.vstack([outer_cloud, circle_cloud])
                    if len(outer_cloud) > 0 else circle_cloud
                )
            inner_cutout_cloud = geometry_points_from_assignments(self.entities, inner_ring_items)

            inner_cloud = geometry_points_from_assignments(
                self.entities,
                transition_items + inner_ring_items,
            )

            upper_inner_to_outer = path_runs_inner_to_outer(upper_edge, inner_cloud, outer_cloud)
            lower_inner_to_outer = path_runs_inner_to_outer(lower_edge, inner_cloud, outer_cloud)

            ordered_upper_frags = upper_frags if upper_inner_to_outer else list(reversed(upper_frags))
            ordered_lower_frags = lower_frags if lower_inner_to_outer else list(reversed(lower_frags))
            export_upper_frags = ordered_upper_frags
            export_lower_frags = ordered_lower_frags

            circle_paths = []
            for _item, cx, cy, r in circle_items:
                th = np.linspace(0.0, 2.0 * math.pi, 240)
                circle_paths.append(np.column_stack([cx + r * np.cos(th), cy + r * np.sin(th)]))

            midline_cloud = build_strip_midline_cloud(upper_edge, lower_edge)
            auto_gap = float(self.auto_gap_var.get())
            mask_join_tolerance = max(auto_gap, 0.5)
            path_join_tolerance = kerf_path_join_tolerance(auto_gap)
            material_mask = build_material_mask(
                [upper_edge, lower_edge] + outer_paths + transition_paths + inner_ring_paths + circle_paths,
                [midline_cloud],
                join_tolerance=mask_join_tolerance,
            )

            if kerf_mm > 0.0:
                export_upper_frags = offset_paths_using_material_mask(
                    export_upper_frags,
                    material_mask,
                    kerf_offset,
                    fallback_reference_cloud=lower_edge,
                )
                export_lower_frags = offset_paths_using_material_mask(
                    export_lower_frags,
                    material_mask,
                    kerf_offset,
                    fallback_reference_cloud=upper_edge,
                )

            if len(inner_cloud) == 0 or len(outer_cloud) == 0:
                manifest_lines.append(
                    "Note: segment numbering was inferred with incomplete inner/outer anchor geometry; verify numbering."
                )

            for seg_num, pts in enumerate(export_upper_frags, start=1):
                p = output_dir / f"upper_segment_{seg_num}.dxf"
                export_records.append({
                    "path": p,
                    "polylines": [(pts, "UPPER_SEGMENT")],
                    "circles": [],
                    "description": f"upper spring edge, length = {cumulative_lengths(pts)[-1]:.3f}",
                })

            for seg_num, pts in enumerate(export_lower_frags, start=1):
                p = output_dir / f"lower_segment_{seg_num}.dxf"
                export_records.append({
                    "path": p,
                    "polylines": [(pts, "LOWER_SEGMENT")],
                    "circles": [],
                    "description": f"lower spring edge, length = {cumulative_lengths(pts)[-1]:.3f}",
                })

            if outer_fixture_items or circle_items:
                outer_polylines = [
                    (pts, "OUTER_FIXTURE")
                    for pts in outer_paths
                ]
                outer_reference_clouds = [upper_edge, lower_edge]
                if len(inner_cloud) > 0:
                    outer_reference_clouds.append(inner_cloud)
                outer_reference_cloud = np.vstack(outer_reference_clouds)
                if kerf_mm > 0.0:
                    compensated_outer_paths = offset_assignment_items_using_material_mask(
                        self.entities,
                        outer_fixture_items,
                        kerf_offset,
                        material_mask,
                        fallback_reference_cloud=midline_cloud,
                        fallback_spatial_neighbor_count=4,
                        join_tolerance=path_join_tolerance,
                    )
                    outer_polylines = [
                        (pts, "OUTER_FIXTURE")
                        for pts in compensated_outer_paths
                    ]
                    outer_circles = [
                        (cx, cy, compensated_circle_radius(item, r, kerf_offset), "OUTER_FIXTURE_HOLES")
                        for item, cx, cy, r in circle_items
                    ]
                else:
                    outer_circles = [(cx, cy, r, "OUTER_FIXTURE_HOLES") for _item, cx, cy, r in circle_items]
                export_records.append({
                    "path": output_dir / "outer_fixture_and_holes.dxf",
                    "polylines": outer_polylines,
                    "circles": outer_circles,
                    "description": "complete outer fixture geometry including holes",
                })
            else:
                manifest_lines.append("outer_fixture_and_holes.dxf: not generated (no outer fixture assigned)")

            if transition_items or inner_ring_items:
                transition_polylines = [
                    (pts, "TRANSITION_REGION")
                    for pts in transition_paths
                ]
                inner_ring_polylines = [
                    (pts, "SPLINED_INNER_RING")
                    for pts in inner_ring_paths
                ]
                inner_end_polylines = transition_polylines + inner_ring_polylines
                inner_reference_clouds = [upper_edge, lower_edge]
                if len(outer_cloud) > 0:
                    inner_reference_clouds.append(outer_cloud)
                inner_reference_cloud = np.vstack(inner_reference_clouds)
                if kerf_mm > 0.0:
                    compensated_transition_paths = offset_assignment_items_using_material_mask(
                        self.entities,
                        transition_items,
                        kerf_offset,
                        material_mask,
                        fallback_reference_cloud=np.vstack([midline_cloud, inner_cutout_cloud]) if len(inner_cutout_cloud) > 0 else midline_cloud,
                        fallback_spatial_neighbor_count=2,
                        join_tolerance=path_join_tolerance,
                    )
                    compensated_inner_ring_paths = offset_assignment_items_using_material_mask(
                        self.entities,
                        inner_ring_items,
                        kerf_offset,
                        material_mask,
                        fallback_reference_cloud=inner_reference_cloud,
                        join_tolerance=path_join_tolerance,
                    )
                    inner_end_polylines = (
                        [(pts, "TRANSITION_REGION") for pts in compensated_transition_paths]
                        + [(pts, "SPLINED_INNER_RING") for pts in compensated_inner_ring_paths]
                    )
                export_records.append({
                    "path": output_dir / "splined_inner_ring_and_transition_region.dxf",
                    "polylines": inner_end_polylines,
                    "circles": [],
                    "description": "complete transition region and splined inner ring",
                })
            else:
                manifest_lines.append(
                    "splined_inner_ring_and_transition_region.dxf: not generated "
                    "(no transition region or inner ring assigned)"
                )

            global_bbox = union_bbox([
                bbox_from_export_geometry(record["polylines"], record["circles"])
                for record in export_records
            ])
            if global_bbox is None:
                raise ValueError("No exportable geometry was generated.")

            positions_csv_path = out_dir / "laser_stitching_positions.csv"
            with positions_csv_path.open("w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([
                    "file_name",
                    "bottom_left_x_mm",
                    "bottom_left_y_mm",
                    "width_mm",
                    "height_mm",
                    "description",
                ])

                for record in export_records:
                    bbox = bbox_from_export_geometry(record["polylines"], record["circles"])
                    if bbox is None:
                        continue

                    xmin, ymin, xmax, ymax = bbox
                    place_x = xmin - global_bbox[0]
                    place_y = ymin - global_bbox[1]
                    width = xmax - xmin
                    height = ymax - ymin

                    record["path"].write_text(build_dxf_content(record["polylines"], record["circles"]))
                    output_paths.append(record["path"])
                    manifest_lines.append(
                        f"{record['path'].name}: {record['description']}, placement = ({place_x:.3f}, {place_y:.3f}) mm"
                    )
                    writer.writerow([
                        record["path"].name,
                        f"{place_x:.3f}",
                        f"{place_y:.3f}",
                        f"{width:.3f}",
                        f"{height:.3f}",
                        record["description"],
                    ])

            config_path = out_dir / "splitter_config.json"
            config_obj = {
                "source_dxf": str(self.dxf_path),
                "source_dxf_rel": self._relative_to_project(self.dxf_path),
                "n_sections": n_sections,
                "auto_gap": float(self.auto_gap_var.get()),
                "kerf_mm": kerf_mm,
                "assignments": {
                    cat: [item.to_json() for item in items]
                    for cat, items in self.assignments.items()
                }
            }
            config_path.write_text(json.dumps(config_obj, indent=2))

            manifest_path = out_dir / "laser_stitching_manifest.txt"
            manifest_path.write_text("\n".join(manifest_lines) + "\n")

            fig, ax = plt.subplots(figsize=(9, 9))
            for seg_num, pts in enumerate(export_upper_frags, start=1):
                ax.plot(pts[:, 0], pts[:, 1], linewidth=1.5, color="tab:blue")
                mid = pts[len(pts) // 2]
                ax.text(mid[0], mid[1], f"U{seg_num}", color="tab:blue", fontsize=7)

            for seg_num, pts in enumerate(export_lower_frags, start=1):
                ax.plot(pts[:, 0], pts[:, 1], linewidth=1.5, color="tab:orange")
                mid = pts[len(pts) // 2]
                ax.text(mid[0], mid[1], f"L{seg_num}", color="tab:orange", fontsize=7)

            for pts, _layer in outer_polylines if outer_fixture_items or circle_items else []:
                ax.plot(pts[:, 0], pts[:, 1], linewidth=2.0, color="tab:green")

            for cx, cy, r, _layer in outer_circles if outer_fixture_items or circle_items else []:
                th = np.linspace(0.0, 2.0 * math.pi, 240)
                ax.plot(cx + r * np.cos(th), cy + r * np.sin(th), linewidth=2.0, color="tab:green")

            for pts, layer in inner_end_polylines if transition_items or inner_ring_items else []:
                if layer == "TRANSITION_REGION":
                    ax.plot(pts[:, 0], pts[:, 1], linewidth=1.8, color="tab:red")
                else:
                    ax.plot(pts[:, 0], pts[:, 1], linewidth=1.6, color="tab:purple")

            ax.set_aspect("equal", adjustable="box")
            ax.set_title("Laser Stitching DXF Preview")
            ax.grid(True, alpha=0.3)
            preview_path = out_dir / "laser_stitching_preview.png"
            fig.savefig(preview_path, dpi=220, bbox_inches="tight")
            plt.close(fig)

            zip_path = out_dir / "laser_stitching_dxfs.zip"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for p in output_paths:
                    zf.write(p, arcname=p.relative_to(out_dir).as_posix())
                zf.write(config_path, arcname=config_path.name)
                zf.write(positions_csv_path, arcname=positions_csv_path.name)
                zf.write(manifest_path, arcname=manifest_path.name)
                zf.write(preview_path, arcname=preview_path.name)

            self.status_var.set(f"Generated outputs in {out_dir}")
            messagebox.showinfo(
                "Done",
                "Generated outputs:\n"
                f"{output_dir}\n\n"
                f"ZIP: {zip_path}\n"
                f"Placement CSV: {positions_csv_path}\n"
                f"Preview: {preview_path}\n"
                f"Manifest: {manifest_path}\n"
                f"Config: {config_path}"
            )

        except Exception as e:
            messagebox.showerror("Generation error", str(e))


def validate_runtime() -> None:
    if sys.platform != "darwin":
        return

    tk_version = tuple(int(part) for part in str(tk.TkVersion).split(".")[:2])
    if tk_version >= (8, 6):
        return

    raise SystemExit(
        "This Python is linked against Tk "
        f"{tk.TkVersion}, which is not usable for this GUI on your macOS install.\n"
        "Run the app with:\n"
        "  ./.venv/bin/python dxf_spiral_splitter_gui_autochain.py\n"
        "or configure your IDE to use:\n"
        f"  {PROJECT_ROOT / '.venv/bin/python'}"
    )


def main():
    validate_runtime()
    root = tk.Tk()
    app = DXFSplitterGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
