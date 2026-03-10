
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

import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DXF_DIR = PROJECT_ROOT / "dxf_files"
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "splitter_configs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "splitter_outputs"


CATEGORIES = [
    "upper_edge",
    "lower_edge",
    "outer_fixture",
    "inner_transition",
    "inner_cutout",
    "hole_circles",
]

CHAINABLE_CATEGORIES = {
    "upper_edge",
    "lower_edge",
    "outer_fixture",
    "inner_transition",
    "inner_cutout",
}

CHAINABLE_TYPES = {"LINE", "ARC", "SPLINE"}


@dataclass
class AssignmentItem:
    idx: int
    reverse: bool = False

    def to_json(self):
        return {"idx": self.idx, "reverse": self.reverse}

    @staticmethod
    def from_json(obj):
        return AssignmentItem(idx=int(obj["idx"]), reverse=bool(obj.get("reverse", False)))


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


# =========================
# Output DXF writing
# =========================

def dxf_header() -> str:
    return "\n".join([
        "0", "SECTION",
        "2", "HEADER",
        "9", "$ACADVER",
        "1", "AC1015",
        "0", "ENDSEC",
        "0", "SECTION",
        "2", "ENTITIES",
    ]) + "\n"


def dxf_footer() -> str:
    return "\n".join(["0", "ENDSEC", "0", "EOF"]) + "\n"


def lwpolyline_entity(points: np.ndarray, layer: str) -> str:
    out = ["0", "LWPOLYLINE", "8", layer, "90", str(len(points)), "70", "0"]
    for x, y in points:
        out += ["10", f"{x:.9f}", "20", f"{y:.9f}"]
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
        self.n_sections_var = tk.IntVar(value=12)
        self.auto_gap_var = tk.DoubleVar(value=0.5)
        self.show_labels_var = tk.BooleanVar(value=True)
        self.show_arrows_var = tk.BooleanVar(value=True)
        self.show_start_end_var = tk.BooleanVar(value=True)
        self.selected_entity_idx: Optional[int] = None
        self.status_var = tk.StringVar(value="Load a DXF.")

        self.figure, self.ax = plt.subplots(figsize=(9, 9))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)

        self.entity_plot_handles = {}
        self.entity_label_handles = {}

        self._build_ui()
        self._bind_events()

    def _build_ui(self):
        main = ttk.Frame(self.root)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main, width=400)
        left.pack(side="left", fill="y", padx=6, pady=6)

        right = ttk.Frame(main)
        right.pack(side="right", fill="both", expand=True, padx=6, pady=6)

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
                text=cat,
                variable=self.current_category,
                value=cat
            ).pack(anchor="w", padx=4, pady=2)

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
        ttk.Label(row, text="Number of sections").pack(side="left")
        ttk.Entry(row, textvariable=self.n_sections_var, width=8).pack(side="right")

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
        ttk.Button(opt_frame, text="Redraw / auto-fit", command=self.redraw_plot).pack(fill="x", padx=4, pady=4)

        edit_frame = ttk.LabelFrame(left, text="Assignments")
        edit_frame.pack(fill="both", expand=True, pady=4)

        self.category_tabs = ttk.Notebook(edit_frame)
        self.category_tabs.pack(fill="both", expand=True, padx=4, pady=4)

        self.listboxes = {}
        for cat in CATEGORIES:
            tab = ttk.Frame(self.category_tabs)
            self.category_tabs.add(tab, text=cat)

            lb = tk.Listbox(tab, exportselection=False, height=10)
            lb.pack(fill="both", expand=True, padx=4, pady=4)
            self.listboxes[cat] = lb

            btn_row1 = ttk.Frame(tab)
            btn_row1.pack(fill="x", padx=4, pady=2)
            ttk.Button(btn_row1, text="Toggle Reverse", command=lambda c=cat: self.toggle_reverse(c)).pack(side="left", expand=True, fill="x", padx=2)
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
                "  1. Choose a category\n"
                "  2. Click entities in order\n"
                "  3. Fix direction with Toggle Reverse\n\n"
                "Semi-automatic:\n"
                "  1. Click one seed entity\n"
                "  2. Press 'Seed with selected entity + auto-chain'\n"
                "  3. Inspect and adjust"
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
                "Arrow = forward direction along the sampled path"
            ),
            justify="left",
            wraplength=360,
        ).pack(fill="x", padx=4, pady=4)

        status_frame = ttk.LabelFrame(left, text="Status")
        status_frame.pack(fill="x", pady=4)
        ttk.Label(status_frame, textvariable=self.status_var, wraplength=360, justify="left").pack(fill="x", padx=4, pady=4)

        canvas_widget = self.canvas.get_tk_widget()
        canvas_widget.pack(in_=right, fill="both", expand=True)

        toolbar_row = ttk.Frame(right)
        toolbar_row.pack(fill="x")
        ttk.Button(toolbar_row, text="Redraw", command=self.redraw_plot).pack(side="left", padx=4, pady=4)

    def _bind_events(self):
        self.canvas.mpl_connect("pick_event", self.on_pick)

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

    def refresh_assignment_lists(self):
        for cat, lb in self.listboxes.items():
            lb.delete(0, "end")
            for item in self.assignments[cat]:
                typ = self.entities[item.idx][0] if self.entities else "?"
                rev = "rev" if item.reverse else "fwd"
                lb.insert("end", f"{item.idx:03d} | {typ:<6} | {rev}")

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
            self.status_var.set(f"Loaded {len(self.entities)} entities.")
            self.redraw_plot()
            messagebox.showinfo("Loaded", f"Loaded {len(self.entities)} entities.")
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
            loaded = {}
            for cat in CATEGORIES:
                loaded[cat] = [AssignmentItem.from_json(x) for x in obj.get("assignments", {}).get(cat, [])]
            self.assignments = loaded

            self.refresh_assignment_lists()
            self.redraw_plot()
            self.status_var.set(f"Loaded config from {path}")
            messagebox.showinfo("Loaded", f"Loaded config from:\n{path}")
        except Exception as e:
            messagebox.showerror("Error loading config", str(e))

    def redraw_plot(self):
        self.ax.clear()
        self.entity_plot_handles.clear()
        self.entity_label_handles.clear()

        if not self.entities:
            self.ax.set_title("Load a DXF")
            self.canvas.draw_idle()
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
            "hole_circles": "tab:brown",
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

            arrow_data = local_direction_arrow_points(pts)
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
            self.ax.set_xlim(xmin - pad, xmax + pad)
            self.ax.set_ylim(ymin - pad, ymax + pad)

        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_title("Click entities to assign them")
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw_idle()

    def on_pick(self, event):
        artist = event.artist
        idx = getattr(artist, "_entity_idx", None)
        if idx is None:
            return

        self.selected_entity_idx = idx
        cat = self.current_category.get()

        if cat == "hole_circles" and self.entities[idx][0] != "CIRCLE":
            self.status_var.set(f"Selected entity {idx}, but it is not a CIRCLE.")
            self.redraw_plot()
            return

        if idx not in [item.idx for item in self.assignments[cat]]:
            for other_cat, items in self.assignments.items():
                if other_cat == cat:
                    continue
                if idx in [item.idx for item in items]:
                    if not messagebox.askyesno(
                        "Entity already assigned",
                        f"Entity {idx} is already assigned to '{other_cat}'.\n"
                        f"Also add it to '{cat}'?"
                    ):
                        self.status_var.set(f"Selected entity {idx}.")
                        self.redraw_plot()
                        return

            self.assignments[cat].append(AssignmentItem(idx=idx, reverse=False))
            self.refresh_assignment_lists()
            self.status_var.set(f"Added entity {idx} to {cat}.")
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
        self.status_var.set(f"Toggled reverse for entity {self.assignments[cat][i].idx} in {cat}.")
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
        self.status_var.set(f"Removed entity {removed.idx} from {cat}.")
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
        self.status_var.set(f"Moved entity in {cat}.")
        self.redraw_plot()

    def _candidate_indices_for_category(self, cat: str) -> List[int]:
        if cat == "hole_circles":
            return [i for i, (typ, _) in enumerate(self.entities) if typ == "CIRCLE"]

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
            messagebox.showwarning("Not chainable", f"Category '{cat}' is not auto-chainable.")
            return

        if not self.entities:
            messagebox.showwarning("No DXF", "Load a DXF first.")
            return

        seed = self.assignments[cat]
        if len(seed) == 0:
            messagebox.showwarning(
                "Need a seed",
                "Add at least one seed entity to this category,\n"
                "or use 'Seed with selected entity + auto-chain'."
            )
            return

        try:
            max_gap = float(self.auto_gap_var.get())
            candidates = self._candidate_indices_for_category(cat)
            chain, log = greedy_autochain(self.entities, seed, candidates, max_gap)
            old_n = len(self.assignments[cat])
            self.assignments[cat] = chain
            self.refresh_assignment_lists()
            self.redraw_plot()
            self.status_var.set(
                f"Auto-chain finished for {cat}: added {len(chain) - old_n} entities "
                f"with max gap {max_gap}."
            )
            if log:
                print("\n".join(log))
        except Exception as e:
            messagebox.showerror("Auto-chain error", str(e))

    def seed_and_auto_chain_current_category(self):
        cat = self.current_category.get()
        if cat not in CHAINABLE_CATEGORIES:
            messagebox.showwarning("Not chainable", f"Category '{cat}' is not auto-chainable.")
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
                raise ValueError(f"Category '{cat}' must contain at least one entity.")

        n_sections = int(self.n_sections_var.get())
        if n_sections < 1:
            raise ValueError("Number of sections must be >= 1.")

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

            upper_edge = path_from_assignment_list(self.entities, self.assignments["upper_edge"])
            lower_edge = path_from_assignment_list(self.entities, self.assignments["lower_edge"])

            u = np.linspace(0.0, 1.0, n_sections + 1)
            upper_frags = [slice_polyline_by_fraction(upper_edge, u[i], u[i + 1]) for i in range(n_sections)]
            lower_frags = [slice_polyline_by_fraction(lower_edge, u[i], u[i + 1]) for i in range(n_sections)]

            section_dir = out_dir / "spiral_open_section_paths"
            section_dir.mkdir(parents=True, exist_ok=True)

            manifest_lines = []
            section_paths = []

            circles = []
            for item in self.assignments["hole_circles"]:
                d = pairdict(self.entities[item.idx][1])
                circles.append((float(d["10"][0]), float(d["20"][0]), float(d["40"][0])))

            for i in range(n_sections):
                p = section_dir / f"spiral_open_section_{i+1:02d}.dxf"
                content = dxf_header()

                content += lwpolyline_entity(upper_frags[i], "SPRING_UPPER")
                content += lwpolyline_entity(lower_frags[i], "SPRING_LOWER")

                if i == 0:
                    for item in self.assignments["outer_fixture"]:
                        content += lwpolyline_entity(
                            sample_entity(self.entities, item.idx, reverse=item.reverse),
                            "OUTER_FIXTURE"
                        )
                    for cx, cy, r in circles:
                        content += circle_entity(cx, cy, r, "OUTER_FIXTURE_HOLES")

                if i == n_sections - 1:
                    for item in self.assignments["inner_transition"]:
                        content += lwpolyline_entity(
                            sample_entity(self.entities, item.idx, reverse=item.reverse),
                            "INNER_TRANSITION"
                        )
                    for item in self.assignments["inner_cutout"]:
                        content += lwpolyline_entity(
                            sample_entity(self.entities, item.idx, reverse=item.reverse),
                            "INNER_CUTOUT"
                        )

                content += dxf_footer()
                p.write_text(content)
                section_paths.append(p)

                parts = [
                    f"Section {i+1:02d}: upper length = {cumulative_lengths(upper_frags[i])[-1]:.3f}",
                    f"lower length = {cumulative_lengths(lower_frags[i])[-1]:.3f}",
                ]
                if i == 0 and (self.assignments["outer_fixture"] or circles):
                    parts.append("includes OUTER_FIXTURE / holes")
                if i == n_sections - 1 and (self.assignments["inner_transition"] or self.assignments["inner_cutout"]):
                    parts.append("includes INNER_TRANSITION / INNER_CUTOUT")
                manifest_lines.append(", ".join(parts))

            outer_file = None
            if self.assignments["outer_fixture"] or circles:
                outer_file = section_dir / "outer_fixture_with_holes_only.dxf"
                content = dxf_header()
                for item in self.assignments["outer_fixture"]:
                    content += lwpolyline_entity(
                        sample_entity(self.entities, item.idx, reverse=item.reverse),
                        "OUTER_FIXTURE"
                    )
                for cx, cy, r in circles:
                    content += circle_entity(cx, cy, r, "OUTER_FIXTURE_HOLES")
                content += dxf_footer()
                outer_file.write_text(content)

            inner_file = None
            if self.assignments["inner_transition"] or self.assignments["inner_cutout"]:
                inner_file = section_dir / "inner_transition_and_cutout_only.dxf"
                content = dxf_header()
                for item in self.assignments["inner_transition"]:
                    content += lwpolyline_entity(
                        sample_entity(self.entities, item.idx, reverse=item.reverse),
                        "INNER_TRANSITION"
                    )
                for item in self.assignments["inner_cutout"]:
                    content += lwpolyline_entity(
                        sample_entity(self.entities, item.idx, reverse=item.reverse),
                        "INNER_CUTOUT"
                    )
                content += dxf_footer()
                inner_file.write_text(content)

            config_path = out_dir / "splitter_config.json"
            config_obj = {
                "source_dxf": str(self.dxf_path),
                "source_dxf_rel": self._relative_to_project(self.dxf_path),
                "n_sections": n_sections,
                "auto_gap": float(self.auto_gap_var.get()),
                "assignments": {
                    cat: [item.to_json() for item in items]
                    for cat, items in self.assignments.items()
                }
            }
            config_path.write_text(json.dumps(config_obj, indent=2))

            manifest_path = out_dir / "spiral_open_section_paths_manifest.txt"
            manifest_path.write_text("\n".join(manifest_lines) + "\n")

            fig, ax = plt.subplots(figsize=(9, 9))
            for i in range(n_sections):
                ax.plot(upper_frags[i][:, 0], upper_frags[i][:, 1], linewidth=1.5)
                ax.plot(lower_frags[i][:, 0], lower_frags[i][:, 1], linewidth=1.5)

            for item in self.assignments["outer_fixture"]:
                pts = sample_entity(self.entities, item.idx, reverse=item.reverse)
                ax.plot(pts[:, 0], pts[:, 1], linewidth=2.0)

            for cx, cy, r in circles:
                th = np.linspace(0.0, 2.0 * math.pi, 240)
                ax.plot(cx + r * np.cos(th), cy + r * np.sin(th), linewidth=2.0)

            for item in self.assignments["inner_transition"]:
                pts = sample_entity(self.entities, item.idx, reverse=item.reverse)
                ax.plot(pts[:, 0], pts[:, 1], linewidth=1.8)

            for item in self.assignments["inner_cutout"]:
                pts = sample_entity(self.entities, item.idx, reverse=item.reverse)
                ax.plot(pts[:, 0], pts[:, 1], linewidth=1.6)

            ax.set_aspect("equal", adjustable="box")
            ax.set_title("Spiral Spring Section Paths Preview")
            ax.grid(True, alpha=0.3)
            preview_path = out_dir / "spiral_open_section_paths_preview.png"
            fig.savefig(preview_path, dpi=220, bbox_inches="tight")
            plt.close(fig)

            zip_path = out_dir / "spiral_open_section_paths.zip"
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for p in section_paths:
                    zf.write(p, arcname=p.relative_to(out_dir).as_posix())
                if outer_file is not None:
                    zf.write(outer_file, arcname=outer_file.relative_to(out_dir).as_posix())
                if inner_file is not None:
                    zf.write(inner_file, arcname=inner_file.relative_to(out_dir).as_posix())
                zf.write(config_path, arcname=config_path.name)
                zf.write(manifest_path, arcname=manifest_path.name)
                zf.write(preview_path, arcname=preview_path.name)

            self.status_var.set(f"Generated outputs in {out_dir}")
            messagebox.showinfo(
                "Done",
                "Generated outputs:\n"
                f"{section_dir}\n\n"
                f"ZIP: {zip_path}\n"
                f"Preview: {preview_path}\n"
                f"Manifest: {manifest_path}\n"
                f"Config: {config_path}"
            )

        except Exception as e:
            messagebox.showerror("Generation error", str(e))


def main():
    root = tk.Tk()
    app = DXFSplitterGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
