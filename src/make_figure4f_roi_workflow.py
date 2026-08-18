"""Render a three-panel, data-faithful ROI/W50 workflow schematic.

Panel a uses the reference-grid NIR display image and the stored X01-X05
coordinates. Panel b enlarges X02 and draws the stored local sampling geometry.
Panel c is explicitly illustrative: it contains no measured profile values.
"""

from pathlib import Path
import csv
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Polygon
import matplotlib.patheffects as pe
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

IMAGE_PATH = ROOT / "data" / "NIR_976-T_display.tif"
ROI_PATH = ROOT / "config" / "M4_5X_ROIs.csv"

CYAN = "#009FCD"
CYAN_FILL = "#8FD7E9"
INK = "#1D1D1F"
GRAY = "#6D6D6D"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7.0,
    "axes.linewidth": 0.8,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_rgb(path: Path) -> np.ndarray:
    """Load RGBA/RGB raster, compositing any transparency onto white."""
    im = Image.open(path).convert("RGBA")
    rgba = np.asarray(im).astype(float) / 255.0
    alpha = rgba[..., 3:4]
    return rgba[..., :3] * alpha + (1.0 - alpha)


def read_rois(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    selected = []
    for row in rows:
        if row["selected"].strip().lower() == "true":
            selected.append({
                "id": row["selected_ID"],
                "x": float(row["center_x"]),
                "y": float(row["center_y"]),
                "normal": float(row["normal_angle_deg"]),
                "parent": row["parent_ID"],
            })
    return selected


def unit_vectors(normal_deg: float):
    """Return normal/tangent directions in reference-image coordinates."""
    theta = math.radians(normal_deg)
    normal = np.array([math.cos(theta), math.sin(theta)])
    tangent = np.array([-math.sin(theta), math.cos(theta)])
    return normal, tangent


def rectangle_vertices(center, normal, tangent, half_normal=22.0, half_tangent=5.5):
    center = np.asarray(center)
    return np.vstack([
        center - half_normal * normal - half_tangent * tangent,
        center + half_normal * normal - half_tangent * tangent,
        center + half_normal * normal + half_tangent * tangent,
        center - half_normal * normal + half_tangent * tangent,
    ])


def add_roi_band(ax, roi, profile_lines=False, label=None, label_offset=(0, 0)):
    center = np.array([roi["x"], roi["y"]])
    normal, tangent = unit_vectors(roi["normal"])
    vertices = rectangle_vertices(center, normal, tangent)
    ax.add_patch(Polygon(vertices, closed=True, facecolor=CYAN_FILL, edgecolor=CYAN,
                         linewidth=1.1, alpha=0.34, zorder=3))
    # Local profile-normal direction.
    endpoints = np.vstack([center - 17 * normal, center + 17 * normal])
    ax.plot(endpoints[:, 0], endpoints[:, 1], color=CYAN, lw=1.8,
            solid_capstyle="round", zorder=4)
    if profile_lines:
        for offset in np.arange(-5, 6, 1):
            c = center + offset * tangent
            line = np.vstack([c - 22 * normal, c + 22 * normal])
            ax.plot(line[:, 0], line[:, 1], color=CYAN,
                    lw=0.65 if offset else 1.35, alpha=0.96, zorder=5)
    ax.scatter([center[0]], [center[1]], s=34, facecolors="white",
               edgecolors=CYAN, linewidths=1.5, zorder=6)
    if label:
        x_text, y_text = center + np.asarray(label_offset)
        text = ax.text(x_text, y_text, label, color=INK, fontsize=7.2,
                       fontweight="bold", zorder=7)
        text.set_path_effects([pe.withStroke(linewidth=2.3, foreground="white")])


def style_image_axis(ax, width, height, xlabel=True, ylabel=True):
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.tick_params(direction="out", length=3, width=0.8, labelsize=6.2)
    if xlabel:
        ax.set_xlabel("Reference-grid X (px)", fontsize=7.2, labelpad=2)
    if ylabel:
        ax.set_ylabel("Reference-grid Y (px)", fontsize=7.2, labelpad=2)


def panel_label(ax, text):
    ax.text(-0.12, 1.04, text, transform=ax.transAxes, fontsize=10,
            fontweight="bold", va="bottom", ha="left")


def build():
    image = load_rgb(IMAGE_PATH)
    h, w = image.shape[:2]
    rois = read_rois(ROI_PATH)
    roi_by_id = {r["id"]: r for r in rois}

    # Contract: image plate + quantitative-method schematic, 183 mm-wide working size.
    fig = plt.figure(figsize=(7.20, 3.82), constrained_layout=False)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.07, 0.98, 1.12],
                          left=0.072, right=0.986, top=0.83, bottom=0.25,
                          wspace=0.42)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    fig.suptitle("ROI selection and transverse-width measurement workflow",
                 x=0.52, y=0.96, fontsize=9.0, fontweight="bold")

    # (a) Reference-grid locations.
    ax_a.imshow(image, origin="upper", extent=(0, w, h, 0), interpolation="nearest")
    label_offsets = {
        "T01": (10, 12), "T02": (15, -13), "T03": (12, 12),
        "T04": (15, -1), "T05": (14, 16),
    }
    for roi in rois:
        add_roi_band(ax_a, roi, label="X" + roi["id"][-2:],
                     label_offset=label_offsets[roi["id"]])
    style_image_axis(ax_a, w, h)
    ax_a.set_title("ROI locations", fontsize=7.0, fontweight="bold", pad=8)
    ax_a.text(0.5, 1.008, "same coordinates in UV and NIR", transform=ax_a.transAxes,
              fontsize=5.8, ha="center", va="bottom", color=CYAN, fontweight="bold")
    panel_label(ax_a, "a")

    # (b) X02 zoom, with the full set of stored transverse profiles.
    r2 = roi_by_id["T02"]
    x0, x1, y0, y1 = 54, 115, 204, 260
    ax_b.imshow(image, origin="upper", extent=(0, w, h, 0), interpolation="nearest")
    ax_b.set_xlim(x0, x1)
    ax_b.set_ylim(y1, y0)
    ax_b.set_aspect("equal")
    add_roi_band(ax_b, r2, profile_lines=True, label="X02", label_offset=(10, -10))
    ax_b.set_xlabel("Reference-grid X (px)", fontsize=7.2, labelpad=2)
    ax_b.set_ylabel("Reference-grid Y (px)", fontsize=7.2, labelpad=2)
    ax_b.tick_params(direction="out", length=3, width=0.8, labelsize=6.2)
    ax_b.set_title("Representative ROI", fontsize=7.0, fontweight="bold", pad=8)
    ax_b.annotate("11 parallel profiles", xy=(69, 212), xytext=(56, 207),
                  fontsize=5.8, ha="left", va="top", color=INK,
                  arrowprops=dict(arrowstyle="->", color=INK, lw=0.65))
    ax_b.annotate("normal to ridge", xy=(86, 230), xytext=(95, 248),
                  fontsize=5.8, ha="left", va="bottom", color=INK,
                  arrowprops=dict(arrowstyle="->", color=INK, lw=0.65))
    panel_label(ax_b, "b")

    # (c) Purely illustrative profile geometry (not measured data).
    xx = np.linspace(-3.2, 3.2, 401)
    yy = np.exp(-(xx / 1.12) ** 2)
    ax_c.plot(xx, yy, color=CYAN, lw=1.8)
    ax_c.axhline(0.5, color=GRAY, lw=0.7, ls=(0, (3, 2)))
    half = 1.12 * math.sqrt(math.log(2))
    ax_c.vlines([-half, half], 0, 0.5, colors=CYAN, lw=0.8, linestyles=(0, (2, 2)))
    ax_c.scatter([-half, half], [0.5, 0.5], s=14, color=CYAN, zorder=3)
    ax_c.add_patch(FancyArrowPatch((-half, 0.18), (half, 0.18), arrowstyle="<->",
                                   mutation_scale=9, color=CYAN, lw=1.1))
    ax_c.text(0, 0.23, "W50 (FWHM)", fontsize=6.5, color=CYAN,
              ha="center", va="bottom", fontweight="bold")
    ax_c.text(3.05, 0.52, "half maximum", fontsize=5.8, color=GRAY, ha="right", va="bottom")
    ax_c.set_xlim(-3.35, 3.35)
    ax_c.set_ylim(0, 1.10)
    ax_c.set_xlabel("Transverse position (px)", fontsize=7.2, labelpad=2)
    ax_c.set_ylabel("Normalized a* intensity", fontsize=7.2, labelpad=2)
    ax_c.set_xticks([-3, 0, 3])
    ax_c.set_yticks([0, 0.5, 1.0])
    ax_c.tick_params(direction="out", length=3, width=0.8, labelsize=6.2)
    ax_c.set_title("Averaged profile and W50", fontsize=7.0, fontweight="bold", pad=8)
    ax_c.text(0.5, -0.35, "11 profiles  →  averaged profile  →  W50",
              transform=ax_c.transAxes, ha="center", va="top", fontsize=6.2, color=INK)
    panel_label(ax_c, "c")

    # Shared legend, using only the requested visual vocabulary.
    handles = [
        Line2D([0], [0], color=CYAN, lw=1.8, label="Local profile-normal direction"),
        plt.Rectangle((0, 0), 1, 1, facecolor=CYAN_FILL, edgecolor=CYAN, alpha=0.34, label="ROI"),
        Line2D([0], [0], marker="o", markersize=5.7, markerfacecolor="white", markeredgewidth=1.3,
               markeredgecolor=CYAN, lw=0, label="Transect centre"),
        Line2D([0], [0], color=CYAN, lw=0.65, label="Individual transverse profiles"),
    ]
    fig.legend(handles=handles, ncols=4, loc="lower center", bbox_to_anchor=(0.53, 0.065),
               fontsize=6.25, frameon=False, handlelength=1.8, handletextpad=0.45,
               columnspacing=1.15)

    stem = OUT / "Figure4f_ROI_selection_transverse_width_workflow"
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    build()
