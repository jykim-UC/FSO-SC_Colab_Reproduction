"""Regenerate measured v27 gallery figures offline; no checkpoints or image assets."""

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULATIONS = ("QPSK", "8QAM", "16QAM")
METHODS = ("fso_sc", "bpg", "bpg_ldpc")
METHOD_LABEL = {"fso_sc": "FSO-SC", "bpg": "BPG", "bpg_ldpc": "BPG + LDPC"}
METHOD_COLOR = {"fso_sc": "#185483", "bpg": "#517455", "bpg_ldpc": "#a3412a"}
METHOD_MARKER = {"fso_sc": "o", "bpg": "^", "bpg_ldpc": "s"}
DEFAULT_CHANNEL = {
    "modulation": "QPSK",
    "launch_power_dbm": 36.0,
    "zenith_angle_deg": 0.0,
    "pointing_error_urad": 20.0,
    "scattered_power_ratio": 0.95,
    "background_light_radiance_w_m2_sr_um": 1e-4,
    "weather": "clear",
}
POWER = tuple(range(28, 37))
ANGLE = tuple(range(0, 81, 10))
POINTING = tuple(range(18, 23))
ZENITH = (0, 30, 60)
RADIANCE = (1e-4, 1e-2, 1.0)
RADIANCE_LABELS = (r"$10^{-4}$", r"$10^{-2}$", "1")
WEATHER = ("dust", "rain", "snow", "fog")
PALETTE = ("#185483", "#a3412a", "#7057a1", "#517455", "#9a6c24", "#296e73")
NOTE = ("v27 historical simulation · 200 substitute holdout images · shading: 95% image bootstrap CI "
        "· v27 BPG decode failures score 1")
CSV_FIELDS = ("figure", "panel", "x_axis", "x_value", "method", "receivers", "combining", *DEFAULT_CHANNEL,
              "v27_lpips_alex", "v27_ci95_lo", "v27_ci95_hi", "paper_lpips_approx", "paper_reference_kind")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def case_key(method, receivers, combining, channel):
    if channel is None:
        return method, receivers, combining, None
    if channel.keys() != DEFAULT_CHANNEL.keys():
        raise ValueError(f"Unexpected channel keys: {sorted(channel)}")
    return method, receivers, combining, tuple(channel[k] for k in DEFAULT_CHANNEL)


class Sweep:
    """Strictly select recorded rows by method, branch configuration, and channel."""

    def __init__(self, aggregate):
        if aggregate["experiment"] != "paper_sweep_v27" or aggregate["images"] != 200:
            raise ValueError("Expected the published v27 aggregate with 200 images")
        self.rows = {}
        self.used = set()
        self.points = []
        for row in aggregate["points"]:
            key = case_key(row["method"], row["receivers"], row["combining"], row["channel"])
            if key in self.rows or row["unique_images"] != aggregate["images"]:
                raise ValueError(f"Duplicate case or inconsistent image count: {key}")
            ci = row["lpips_alex_ci95"]
            if len(ci) != 2 or not all(map(math.isfinite, [*ci, row["lpips_alex"]])) or ci[0] > ci[1]:
                raise ValueError(f"Malformed LPIPS or confidence interval: {key}")
            self.rows[key] = row
        self.clean = {case_key(method, 1, "MRC", None) for method in ("fso_sc", "bpg")}
        if len(self.rows) != aggregate["case_count"] or not self.clean <= self.rows.keys():
            raise ValueError("Unexpected v27 case count or missing clean references")

    def series(self, method, x_axis, x_values, receivers=1, combining="MRC", **channel):
        records = []
        for x in x_values:
            args = DEFAULT_CHANNEL | channel | {x_axis: x}
            key = case_key(method, receivers, combining, args)
            if key not in self.rows:
                raise ValueError(f"No measured v27 row for {key}")
            records.append(self.rows[key])
            self.used.add(key)
        return records

    def verify_complete(self, extra_raster_points):
        if self.used != self.rows.keys() - self.clean or len(self.points) != 783:
            raise ValueError(
                f"Incomplete v27 figure mapping: {len(self.used)} cases, {len(self.points)} points"
            )
        for kind, expected in (("vector_marker_approx", 531), ("raster_ceiling_approx", 3),
                               ("raster_approx", extra_raster_points)):
            if sum(point["paper_reference_kind"] == kind for point in self.points) != expected:
                raise ValueError(f"Expected {expected} author references of type {kind}")

    def add_point(self, figure, panel, axis, x, row, paper_y, paper_kind):
        lo, hi = row["lpips_alex_ci95"]
        self.points.append({
            "figure": figure, "panel": panel, "x_axis": axis, "x_value": x,
            "method": row["method"], "receivers": row["receivers"], "combining": row["combining"],
            **row["channel"], "v27_lpips_alex": row["lpips_alex"], "v27_ci95_lo": lo, "v27_ci95_hi": hi,
            "paper_lpips_approx": paper_y, "paper_reference_kind": paper_kind if paper_y is not None else "",
        })


def plot_series(sweep, ax, x, rows, *, location, label, color, marker, linestyle="-",
                paper=(), paper_axis=None, paper_kind="vector_marker_approx"):
    means = [row["lpips_alex"] for row in rows]
    lo, hi = zip(*(row["lpips_alex_ci95"] for row in rows))
    ax.fill_between(x, lo, hi, color=color, alpha=0.10, linewidth=0)
    ax.plot(x, means, label=label, color=color, marker=marker, ms=4, lw=1.55, ls=linestyle)
    paper_y = {row[paper_axis]: row["lpips"] for row in paper}
    kinds = {row[paper_axis]: row.get("paper_kind", paper_kind) for row in paper}
    if paper_y:
        if (len(paper_y) != len(paper) or not paper_y.keys() <= set(x)
                or not all(math.isfinite(y) and -0.025 <= y <= 1.075 for y in paper_y.values())):
            raise ValueError(f"Malformed, duplicate, or out-of-grid paper points for {label}")
        if paper_kind.startswith("raster"):
            for kind, color_ref, marker_ref in (("raster_approx", color, marker), ("raster_ceiling_approx", "#22262d", "D")):
                positions = [point for point, provenance in kinds.items() if provenance == kind]
                if positions:
                    ax.scatter(positions, [paper_y[point] for point in positions], s=37, marker=marker_ref,
                               edgecolor=color_ref, facecolor="white", zorder=5)
        else:
            # NaNs keep unverified gaps empty. A line never spans an unverified marker.
            ax.plot(x, [paper_y.get(point, math.nan) for point in x], color=color, ls="--",
                    lw=1.1, marker=marker, ms=5, mfc="white", mew=1, alpha=0.95)
    for value, row in zip(x, rows):
        sweep.add_point(*location, value, row, paper_y.get(value), kinds.get(value))


def set_panel(ax, title, x_ticks):
    ax.set_title(title, fontsize=11, loc="left", pad=5, fontweight="bold")
    ax.set_xticks(x_ticks)
    ax.set_ylim(-0.025, 1.075)
    ax.set_yticks((0, 0.2, 0.4, 0.6, 0.8, 1))
    ax.grid(color="#a0aeba", alpha=0.32, linewidth=0.65)
    ax.set_axisbelow(True)


def begin(plt, rows, cols, size, title, xlabel, detail):
    fig, axes = plt.subplots(rows, cols, figsize=size, sharey=True, squeeze=False)
    fig.subplots_adjust(left=0.075, right=0.984, bottom=0.216 if rows == 1 else 0.164,
                        top=0.879, wspace=0.22, hspace=0.40)
    fig.suptitle(title, fontsize=17, fontweight="bold", y=0.976)
    fig.supxlabel(xlabel, y=0.105, fontsize=11)
    fig.supylabel("LPIPS (lower is better)", x=0.017, fontsize=11)
    fig.text(0.5, 0.021, NOTE + "\n" + detail, ha="center", fontsize=9, color="#3f4850")
    return fig, axes


def paper_legend_handle():
    from matplotlib.lines import Line2D

    return Line2D([], [], label="Paper digitization (approx.)", c="#495159",
                  marker="o", mfc="white", ls="--")


def method_legend(fig, methods, paper=False):
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], label=METHOD_LABEL[m], c=METHOD_COLOR[m], marker=METHOD_MARKER[m]) for m in methods]
    if paper:
        handles.append(paper_legend_handle())
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.061), ncols=len(handles),
               frameon=False, fontsize=9, handlelength=2.3)


def plot_fig4(plt, sweep, references):
    fig, axes = begin(plt, 3, 3, (14, 10), "Fig. 4 · Zenith-angle sensitivity", "Zenith angle (°)",
                      "Rows: published power settings. Open markers show available author readings; LPIPS backbone unverified.")
    for i, (letter, power) in enumerate(zip("abc", (30, 33, 36))):
        paper_panel = references["figure4"][letter]["series"]
        if not paper_panel or not paper_panel.keys() <= {f"{m}_{mod.lower()}" for m in METHODS for mod in MODULATIONS}:
            raise ValueError(f"Unexpected author series for Fig. 4({letter})")
        for j, mod in enumerate(MODULATIONS):
            ax = axes[i, j]
            for method in METHODS:
                rows = sweep.series(method, "zenith_angle_deg", ANGLE, modulation=mod, launch_power_dbm=power)
                paper = paper_panel.get(f"{method}_{mod.lower()}", [])
                plot_series(sweep, ax, ANGLE, rows, location=("fig4", letter, "zenith_angle_deg"),
                            label=METHOD_LABEL[method], color=METHOD_COLOR[method], marker=METHOD_MARKER[method],
                            paper=paper, paper_axis="zenith_deg")
            set_panel(ax, f"({letter})  {power} dBm  /  {mod}", ANGLE)
    method_legend(fig, METHODS, paper=True)
    return fig


def plot_fig6(plt, sweep, anchors, *, pointing, raster=None):
    raster = {} if raster is None else raster["figure6"]
    letters = "abc" if pointing else "def"
    has_raster = any(raster.get(letter, {}).get("series", {}) for letter in letters)
    if pointing:
        detail = ("Open markers: selected approximate author raster readings, not full curves." if has_raster else
                  "No numerical author readings are overlaid for these rasterized 3-D panels.")
    else:
        detail = ("Open markers: selected approximate author raster readings; diamonds are three Fig. 6(d) ceiling anchors."
                  if has_raster else "Open diamonds: only three approximate Fig. 6(d) raster ceiling anchors; no other author values digitized.")
    name, xlabel = (("Fig. 6(a–c) · Pointing error", "Pointing error (µrad)") if pointing else
                    ("Fig. 6(d–f) · Background radiance", "Launch optical power (dBm)"))
    if not pointing:
        detail += " BGL: W/(m²·sr·µm)."
        if anchors["method"] != "bpg_ldpc" or anchors["channel"] != {
            "modulation": "QPSK", "weather": "clear", "zenith_angle_deg": 0,
            "pointing_error_urad": 19, "background_light_radiance_w_m2_sr_um": 1.0
        }:
            raise ValueError("Fig. 6(d) author anchors do not match the plotted condition")
    fig, axes = begin(plt, 1, 3, (14, 4.9), name, xlabel, detail)
    x_axis, x_ticks = ("pointing_error_urad", POINTING) if pointing else ("launch_power_dbm", POWER)
    for ax, letter, mod in zip(axes.flat, letters, MODULATIONS):
        raster_series = raster.get(letter, {}).get("series", {})
        expected = set()
        for k, value in enumerate(ZENITH if pointing else RADIANCE):
            args = ({"zenith_angle_deg": value} if pointing else {
                "pointing_error_urad": 19, "background_light_radiance_w_m2_sr_um": value
            })
            condition = f"z={value}°" if pointing else f"BGL {RADIANCE_LABELS[k]}"
            for method in ("fso_sc", "bpg_ldpc"):
                rows = sweep.series(method, x_axis, x_ticks, modulation=mod, **args)
                paper_axis = "pointing_urad" if pointing else "power_dbm"
                condition_key = f"{method}_{'z' if pointing else 'bgl'}{value:g}"
                expected.add(condition_key)
                raster_points = raster_series.get(condition_key, ())
                ceiling_points = anchors["points"] if letter == "d" and value == 1.0 and method == "bpg_ldpc" else ()
                paper = ([dict(p, paper_kind="raster_approx") for p in raster_points] +
                         [dict(p, paper_kind="raster_ceiling_approx") for p in ceiling_points])
                plot_series(sweep, ax, x_ticks, rows, location=("fig6", letter, x_axis),
                            label=f"{condition} · {METHOD_LABEL[method]}",
                            color=PALETTE[k], marker=METHOD_MARKER[method],
                            linestyle="-" if method == "fso_sc" else "--", paper=paper, paper_axis=paper_axis,
                            paper_kind="raster_approx")
        if not raster_series.keys() <= expected:
            raise ValueError(f"Unexpected author raster series for Fig. 6({letter})")
        set_panel(ax, f"({letter})  {mod}", x_ticks)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.062), ncols=6,
               frameon=False, fontsize=8)
    return fig


def plot_fig7(plt, sweep, references):
    fig, axes = begin(plt, 2, 2, (11, 7.9), "Fig. 7 · Weather and launch power (QPSK)", "Launch optical power (dBm)",
                      "One published panel split by weather for readability. Open markers: available author readings; backbone unverified.")
    paper_series = references["figure7"]["series"]
    if not paper_series or not paper_series.keys() <= {f"{m}_{w}" for m in ("fso_sc", "bpg_ldpc") for w in WEATHER}:
        raise ValueError("Unexpected author series for Fig. 7")
    for ax, weather in zip(axes.flat, WEATHER):
        for method in ("fso_sc", "bpg_ldpc"):
            rows = sweep.series(method, "launch_power_dbm", POWER, weather=weather, pointing_error_urad=15)
            plot_series(sweep, ax, POWER, rows, location=("fig7", "a", "launch_power_dbm"),
                        label=METHOD_LABEL[method], color=METHOD_COLOR[method],
                        marker=METHOD_MARKER[method], paper=paper_series.get(f"{method}_{weather}", []),
                        paper_axis="power_dbm")
        set_panel(ax, weather.title(), POWER)
    method_legend(fig, ("fso_sc", "bpg_ldpc"), paper=True)
    return fig


def plot_fig8(plt, sweep, references):
    fig, axes = begin(plt, 2, 2, (13.1, 8.4), "Fig. 8 · Receive diversity (16-QAM)", "Launch optical power (dBm)",
                      "Open markers: approximate author readings. Independent branches, ideal relative branch CSI. "
                      "In (d), LDPC follows the legend and text.")
    panels = (("a", .95, False), ("b", .95, True), ("c", .75, True), ("d", .596, True))
    for ax, (letter, rho, fec) in zip(axes.flat, panels):
        paper_series = references["figure8"][letter]["series"]
        baseline = "bpg_ldpc" if fec else "bpg"
        configs = (("fso_sc", 1, "MRC"), (baseline, 1, "MRC"), (baseline, 2, "EGC"),
                   (baseline, 4, "EGC"), (baseline, 2, "MRC"), (baseline, 4, "MRC"))
        expected_keys = set()
        for j, (method, receivers, combining) in enumerate(configs):
            rows = sweep.series(method, "launch_power_dbm", POWER, receivers, combining, modulation="16QAM",
                                pointing_error_urad=19, scattered_power_ratio=rho)
            branch = f"{receivers} Rx" if receivers == 1 else f"{receivers} Rx {combining}"
            label = f"FSO-SC · {branch}" if j == 0 else f"BPG baseline · {branch}"
            paper_key = f"{method}_{receivers}rx" + ("" if receivers == 1 else f"_{combining.lower()}")
            expected_keys.add(paper_key)
            plot_series(sweep, ax, POWER, rows, location=("fig8", letter, "launch_power_dbm"),
                        label=label, color=PALETTE[j], marker="os^DvP"[j],
                        paper=paper_series.get(paper_key, []), paper_axis="power_dbm")
        if not paper_series or not paper_series.keys() <= expected_keys:
            raise ValueError(f"Unexpected author series for Fig. 8({letter})")
        set_panel(ax, f"({letter})  ρ={rho:g}  /  baseline {'with' if fec else 'without'} LDPC", POWER)
    handles, _ = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles=handles + [paper_legend_handle()], loc="lower center",
               bbox_to_anchor=(0.5, 0.061), ncols=7, frameon=False, fontsize=7.6)
    return fig


def render_gallery(aggregate_path, reference_path, anchor_path, output, points_path, raster_path=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sweep = Sweep(read_json(aggregate_path))
    references, anchors = read_json(reference_path), read_json(anchor_path)
    raster = read_json(raster_path) if raster_path is not None else None
    n_raster = 0
    if raster is not None:
        if not raster["figure6"].keys() <= set("abcdef"):
            raise ValueError("Unexpected panel in Fig. 6 author raster reference")
        n_raster = sum(len(series) for panel in raster["figure6"].values() for series in panel["series"].values())
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.facecolor": "white"})
    figures = (
        ("v27_fig4.png", plot_fig4(plt, sweep, references)),
        ("v27_fig6_pointing.png", plot_fig6(plt, sweep, anchors, pointing=True, raster=raster)),
        ("v27_fig6_background.png", plot_fig6(plt, sweep, anchors, pointing=False, raster=raster)),
        ("v27_fig7.png", plot_fig7(plt, sweep, references)),
        ("v27_fig8.png", plot_fig8(plt, sweep, references)),
    )
    sweep.verify_complete(n_raster)
    output.mkdir(parents=True, exist_ok=True)
    for name, fig in figures:
        fig.savefig(output / name, dpi=160)
        plt.close(fig)
        print(output / name)
    points_path.parent.mkdir(parents=True, exist_ok=True)
    with points_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(sweep.points)
    print(points_path)
    print(f"Verified {len(sweep.used)} physical cases, {len(sweep.clean)} clean references, "
          f"and {len(sweep.points)} plotted measurements.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, default=ROOT / "results/paper_sweep_v27_aggregate.json")
    parser.add_argument("--paper-reference", type=Path, default=ROOT / "config/paper_gallery_reference.json")
    parser.add_argument("--fig6d-anchors", type=Path, default=ROOT / "config/paper_gallery_fig6d_anchors.json")
    default_raster = ROOT / "config/paper_gallery_fig6_raster_reference.json"
    parser.add_argument("--fig6-raster", type=Path, default=default_raster if default_raster.is_file() else None)
    parser.add_argument("--output", type=Path, default=ROOT / "results/figures")
    parser.add_argument("--points", type=Path, default=ROOT / "results/paper_gallery_points.csv")
    args = parser.parse_args()
    render_gallery(args.aggregate, args.paper_reference, args.fig6d_anchors, args.output, args.points, args.fig6_raster)
