"""Build publication-ready composite figures for Supplementary Figures S9-S12.

The script is intentionally table-driven: it reads only the regenerated tables
under ``results/tables`` and the frozen transect order under ``config``. In this
release, ``reference px`` are the registered-image pixels of the common grid.
No figure embedded in a manuscript document is used as a plotting source.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import measurement_core as core

CFG = ROOT / "config"
TAB = ROOT / "results" / "tables"
OUT = ROOT / "results" / "figures"

UV = "#D55E00"
NIR = "#0072B2"
GRID = "#D9D9D9"
TEXT = "#222222"
MARKERS = ["o", "s", "^", "D", "P"]
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 2))]
PROFILE_HALF_LENGTH_REF = core.PROFILE_HALF_LENGTH_REF
BACKGROUND_FRACTION = core.BACKGROUND_FRACTION
PRIMARY_SIGMA_REF = core.PRIMARY_SIGMA_REF
MIN_PNR_CRITERION = core.MIN_CNR


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.5,
            "axes.titlesize": 8.0,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "legend.frameon": False,
            "figure.dpi": 120,
            "savefig.dpi": 450,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def polish_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.tick_params(direction="out", length=3, width=0.8)


def save(fig: plt.Figure, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.png"
    fig.savefig(path, facecolor="white", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Wrote {path.relative_to(ROOT)}")
    return path


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    rois = pd.read_csv(CFG / "transects.csv")
    measurements = pd.read_csv(TAB / "measurements.csv")
    profiles = pd.read_csv(TAB / "profiles.csv")
    sensitivity = pd.read_csv(TAB / "smoothing_sensitivity.csv")
    summary = pd.read_csv(TAB / "summary.csv").iloc[0]
    return rois, measurements, profiles, sensitivity, summary


def validate_inputs(
    rois: pd.DataFrame,
    measurements: pd.DataFrame,
    profiles: pd.DataFrame,
    sensitivity: pd.DataFrame,
    summary: pd.Series,
) -> list[str]:
    order = rois["ROI_ID"].astype(str).tolist()
    if len(order) != 5 or len(set(order)) != 5:
        raise RuntimeError("Publication figures require exactly five unique frozen transects")
    expected_pairs = pd.MultiIndex.from_product([["UV", "NIR"], order])
    observed_pairs = pd.MultiIndex.from_frame(measurements[["condition", "ROI_ID"]].astype(str))
    if len(measurements) != 10 or set(observed_pairs) != set(expected_pairs):
        raise RuntimeError("measurements.csv does not contain the expected ten condition-transect rows")
    if not measurements["valid"].astype(str).str.lower().eq("true").all():
        raise RuntimeError("Publication figures require all ten primary measurements to be valid")
    if len(sensitivity) != 40 or not sensitivity["valid"].astype(str).str.lower().eq("true").all():
        raise RuntimeError("Publication figures require all 40 smoothing-sensitivity rows to be valid")

    widths = (measurements["x50_right"] - measurements["x50_left"]).abs()
    np.testing.assert_allclose(widths, measurements["FWHM"], rtol=0.0, atol=1e-12)

    profile_pairs = set(pd.MultiIndex.from_frame(profiles[["condition", "ROI_ID"]].astype(str)))
    if profile_pairs != set(expected_pairs):
        raise RuntimeError("profiles.csv does not contain all ten primary profiles")

    primary = sensitivity[np.isclose(sensitivity["sigma_ref_px"], PRIMARY_SIGMA_REF)].copy()
    joined = measurements.merge(
        primary[["condition", "ROI_ID", "FWHM", "peak_to_noise"]],
        on=["condition", "ROI_ID"],
        suffixes=("_measurement", "_sensitivity"),
        validate="one_to_one",
    )
    if len(joined) != 10:
        raise RuntimeError("The sigma=0.75 sensitivity rows do not match the primary measurements")
    np.testing.assert_allclose(
        joined["FWHM_measurement"], joined["FWHM_sensitivity"], rtol=0.0, atol=1e-12
    )
    np.testing.assert_allclose(
        joined["peak_to_noise_measurement"],
        joined["peak_to_noise_sensitivity"],
        rtol=0.0,
        atol=1e-12,
    )

    uv_mean = float(measurements.loc[measurements.condition.eq("UV"), "FWHM"].mean())
    nir_mean = float(measurements.loc[measurements.condition.eq("NIR"), "FWHM"].mean())
    np.testing.assert_allclose(uv_mean, float(summary["UV_mean_W50_reference_px"]), atol=1e-12)
    np.testing.assert_allclose(nir_mean, float(summary["NIR_mean_W50_reference_px"]), atol=1e-12)
    np.testing.assert_allclose(uv_mean / nir_mean, float(summary["Gx_ratio_of_means"]), atol=1e-12)
    return order


def condition_profile(
    profiles: pd.DataFrame, measurements: pd.DataFrame, condition: str, roi_id: str
) -> tuple[pd.DataFrame, pd.Series]:
    q = profiles[(profiles.condition == condition) & (profiles.ROI_ID == roi_id)].sort_values(
        "distance_reference_px"
    )
    mm = measurements[(measurements.condition == condition) & (measurements.ROI_ID == roi_id)].iloc[0]
    if q.empty:
        raise RuntimeError(f"Missing profile for {condition}/{roi_id}")
    return q, mm


def make_s9(
    order: list[str], measurements: pd.DataFrame, profiles: pd.DataFrame, summary: pd.Series
) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.75), sharex=True, sharey=True)
    axes_flat = axes.ravel()
    colors = {"UV": UV, "NIR": NIR}

    for index, roi_id in enumerate(order):
        ax = axes_flat[index]
        for condition in ["UV", "NIR"]:
            q, mm = condition_profile(profiles, measurements, condition, roi_id)
            x = q["distance_reference_px"].to_numpy(float)
            y = q["normalized"].to_numpy(float)
            color = colors[condition]
            ax.plot(x, y, color=color, linewidth=1.35, zorder=2)
            ax.plot(
                [mm.x50_left, mm.x50_left, mm.x50_right, mm.x50_right],
                [-0.70, 0.5, 0.5, -0.70],
                color=color,
                linestyle=":",
                linewidth=0.8,
                alpha=0.85,
            )
            ax.scatter([mm.x50_left, mm.x50_right], [0.5, 0.5], s=10, color=color, zorder=4)
            ax.text(
                0.98,
                0.94 if condition == "UV" else 0.84,
                f"{condition} {mm.FWHM:.2f} px",
                color=color,
                ha="right",
                va="top",
                transform=ax.transAxes,
                fontsize=6.2,
            )
        ax.axhline(0.5, color="#666666", linestyle="--", linewidth=0.8, zorder=1)
        ax.set_title(f"X{index + 1:02d}", loc="left", fontweight="bold")
        ax.set_xlim(-22, 22)
        ax.set_ylim(-0.72, 1.12)
        polish_axis(ax)

    key = axes_flat[-1]
    key.axis("off")
    handles = [
        Line2D([0], [0], color=UV, lw=1.8, label="248 nm UV"),
        Line2D([0], [0], color=NIR, lw=1.8, label="976 nm NIR"),
        Line2D([0], [0], color="#666666", lw=0.8, ls="--", label="Half maximum"),
    ]
    key.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.04, 0.98))
    # The manuscript notation G_lat maps to the repository field Gx_ratio_of_means.
    key.text(
        0.04,
        0.58,
        "Mean apparent W50\n"
        f"UV   {float(summary['UV_mean_W50_reference_px']):.2f} reference px\n"
        f"NIR  {float(summary['NIR_mean_W50_reference_px']):.2f} reference px\n"
        f"G$_{{\\mathrm{{lat}}}}$  {float(summary['Gx_ratio_of_means']):.3f}",
        transform=key.transAxes,
        va="top",
        color=TEXT,
        linespacing=1.45,
    )
    fig.supxlabel("Distance along profile normal (reference px)", y=0.015)
    fig.supylabel("Background-normalized Lab a* signal", x=0.012)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.12, top=0.96, wspace=0.18, hspace=0.24)
    return save(fig, "Figure_S9_profiles_all")


def monotonic_envelopes(x: np.ndarray, y: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    center = np.abs(x) <= PROFILE_HALF_LENGTH_REF * 0.42
    peak_idx = int(np.flatnonzero(center)[np.argmax(y[center])])
    left = np.maximum.accumulate(y[: peak_idx + 1])
    right = np.maximum.accumulate(y[peak_idx:][::-1])[::-1]
    return peak_idx, left, right


def make_s10(order: list[str], measurements: pd.DataFrame, profiles: pd.DataFrame) -> Path:
    roi_id = order[3]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharex=True, sharey=True)
    colors = {"UV": UV, "NIR": NIR}

    for panel, (ax, condition) in enumerate(zip(axes, ["UV", "NIR"])):
        q, mm = condition_profile(profiles, measurements, condition, roi_id)
        x = q["distance_reference_px"].to_numpy(float)
        filtered_norm = q["normalized"].to_numpy(float)
        background = float(mm.background)
        center = np.abs(x) <= PROFILE_HALF_LENGTH_REF * 0.42
        amplitude = float(np.max(q.loc[center, "filtered"].to_numpy(float)) - background)
        if amplitude <= 0:
            raise RuntimeError(f"Non-positive profile amplitude for {condition}/{roi_id}")
        raw_norm = (q["raw_mean"].to_numpy(float) - background) / amplitude
        raw_sd_norm = q["profile_stack_SD"].to_numpy(float) / amplitude
        peak_idx, left_env, right_env = monotonic_envelopes(x, filtered_norm)
        n_bg = max(5, int(BACKGROUND_FRACTION * len(x)))
        color = colors[condition]

        ax.axvspan(x[0], x[n_bg - 1], color="#A6A6A6", alpha=0.16, linewidth=0, zorder=0)
        ax.axvspan(x[-n_bg], x[-1], color="#A6A6A6", alpha=0.16, linewidth=0, zorder=0)
        ax.fill_between(
            x,
            raw_norm - raw_sd_norm,
            raw_norm + raw_sd_norm,
            color="#8A8A8A",
            alpha=0.18,
            linewidth=0,
            zorder=1,
        )
        ax.plot(x, raw_norm, color="#8A8A8A", linewidth=0.75, alpha=0.9, zorder=2)
        ax.plot(x, filtered_norm, color=color, linewidth=1.7, zorder=3)
        ax.plot(x[: peak_idx + 1], left_env, color="#222222", ls="--", lw=0.95, zorder=4)
        ax.plot(x[peak_idx:], right_env, color="#222222", ls="--", lw=0.95, zorder=4)
        ax.axhline(0.5, color="#666666", linestyle=":", linewidth=0.9, zorder=1)
        for crossing in [float(mm.x50_left), float(mm.x50_right)]:
            ax.vlines(crossing, -0.38, 0.5, color=color, linestyle=":", linewidth=0.9)
        ax.scatter([mm.x50_left, mm.x50_right], [0.5, 0.5], color=color, s=18, zorder=5)
        ax.annotate(
            "",
            xy=(float(mm.x50_left), 0.59),
            xytext=(float(mm.x50_right), 0.59),
            arrowprops={"arrowstyle": "<->", "color": color, "lw": 1.1},
        )
        ax.text(
            (float(mm.x50_left) + float(mm.x50_right)) / 2,
            0.63,
            f"W50 = {float(mm.FWHM):.2f} px",
            color=color,
            ha="center",
            va="bottom",
            fontsize=6.5,
            fontweight="bold",
        )
        ax.set_title(f"{condition} ({'248 nm' if condition == 'UV' else '976 nm'})")
        ax.text(-0.12, 1.03, f"({chr(97 + panel)})", transform=ax.transAxes, fontweight="bold")
        ax.set_xlim(-22, 22)
        ax.set_ylim(-0.40, 1.18)
        polish_axis(ax)

    handles = [
        Line2D([0], [0], color="#8A8A8A", lw=0.8, label="Raw 11-profile mean +/- SD"),
        Line2D([0], [0], color=TEXT, lw=1.7, label="Gaussian-filtered profile"),
        Line2D([0], [0], color=TEXT, lw=0.95, ls="--", label="Monotonic flank envelope"),
        Patch(facecolor="#A6A6A6", alpha=0.20, label="Background region"),
        Line2D([0], [0], color=TEXT, marker="o", lw=0, ms=4, label="0.5 crossings"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.04), ncol=3)
    fig.supxlabel("Transverse position (reference px)", y=0.02)
    fig.supylabel("Background-normalized Lab a* signal", x=0.012)
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.15, top=0.80, wspace=0.14)
    return save(fig, "Figure_S10_representative_W50_X04_diagnostic")


def make_s11(
    order: list[str], measurements: pd.DataFrame, sensitivity: pd.DataFrame
) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharey=True)
    colors = {"UV": UV, "NIR": NIR}
    offsets = {"UV": -0.08, "NIR": 0.08}
    x_primary = np.arange(1, len(order) + 1, dtype=float)

    ax = axes[0]
    for condition in ["UV", "NIR"]:
        q = measurements[measurements.condition.eq(condition)].set_index("ROI_ID").loc[order]
        y = q["peak_to_noise"].to_numpy(float)
        x = x_primary + offsets[condition]
        ax.plot(x, y, color=colors[condition], linewidth=0.9, alpha=0.38, zorder=1)
        for i, (xx, yy) in enumerate(zip(x, y)):
            ax.scatter(
                xx,
                yy,
                marker=MARKERS[i],
                s=28,
                facecolor=colors[condition],
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
            )
    ax.set_xticks(x_primary, [f"X{i:02d}" for i in range(1, 6)])
    ax.set_ylabel("Peak-to-noise ratio")
    ax.set_title("Primary profiles ($\\sigma$ = 0.75 reference px)")

    ax = axes[1]
    sigmas = np.sort(sensitivity["sigma_ref_px"].unique().astype(float))
    for condition in ["UV", "NIR"]:
        for i, roi_id in enumerate(order):
            q = sensitivity[
                sensitivity.condition.eq(condition) & sensitivity.ROI_ID.eq(roi_id)
            ].sort_values("sigma_ref_px")
            ax.plot(
                q["sigma_ref_px"],
                q["peak_to_noise"],
                color=colors[condition],
                linewidth=0.65,
                alpha=0.22,
                marker=MARKERS[i],
                markersize=3.2,
                markerfacecolor="white",
                markeredgewidth=0.65,
            )
        minimum = (
            sensitivity[sensitivity.condition.eq(condition)]
            .groupby("sigma_ref_px", sort=True)["peak_to_noise"]
            .min()
            .reindex(sigmas)
        )
        ax.plot(sigmas, minimum, color=colors[condition], linewidth=2.0, marker="o", markersize=3.2)
        ax.text(
            sigmas[-1] + 0.025,
            minimum.iloc[-1],
            f"{condition} min",
            color=colors[condition],
            va="center",
            fontsize=6.2,
        )
    ax.set_xlim(-0.03, 1.17)
    ax.set_xticks(sigmas)
    ax.set_xlabel("Gaussian $\\sigma$ (reference px)")
    ax.set_title("All smoothing settings")

    for panel, ax in enumerate(axes):
        ax.axhline(MIN_PNR_CRITERION, color="#555555", linestyle="--", linewidth=1.0, zorder=0)
        ax.text(
            0.02,
            MIN_PNR_CRITERION + 0.28,
            "minimum PNR criterion = 3",
            color="#555555",
            fontsize=6.0,
            transform=ax.get_yaxis_transform(),
        )
        ax.text(-0.12, 1.03, f"({chr(97 + panel)})", transform=ax.transAxes, fontweight="bold")
        polish_axis(ax)
    axes[0].set_ylim(2.25, max(18.4, float(sensitivity.peak_to_noise.max()) + 0.5))

    handles = [
        Line2D([0], [0], color=UV, lw=1.8, label="UV"),
        Line2D([0], [0], color=NIR, lw=1.8, label="NIR"),
    ] + [
        Line2D([0], [0], color="#555555", marker=marker, lw=0, markersize=4, label=f"X{i:02d}")
        for i, marker in enumerate(MARKERS, 1)
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=7)
    fig.subplots_adjust(left=0.09, right=0.95, bottom=0.17, top=0.79, wspace=0.16)
    return save(fig, "Figure_S11_PNR_validity_publication")


def make_s12(order: list[str], sensitivity: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), sharex=True, sharey=True)
    colors = {"UV": UV, "NIR": NIR}
    changes: list[float] = []

    for panel, (ax, condition) in enumerate(zip(axes, ["UV", "NIR"])):
        for i, roi_id in enumerate(order):
            q = sensitivity[
                sensitivity.condition.eq(condition) & sensitivity.ROI_ID.eq(roi_id)
            ].sort_values("sigma_ref_px")
            baseline = q.loc[np.isclose(q.sigma_ref_px, PRIMARY_SIGMA_REF), "FWHM"]
            if len(baseline) != 1:
                raise RuntimeError(f"Missing unique sigma=0.75 baseline for {condition}/{roi_id}")
            y = 100.0 * (q["FWHM"].to_numpy(float) / float(baseline.iloc[0]) - 1.0)
            changes.extend(y.tolist())
            ax.plot(
                q["sigma_ref_px"],
                y,
                color=colors[condition],
                linewidth=1.05,
                linestyle=LINESTYLES[i],
                marker=MARKERS[i],
                markersize=4.2,
                markerfacecolor="white",
                markeredgecolor=colors[condition],
                markeredgewidth=0.9,
                label=f"X{i + 1:02d}",
            )
        # The band is a visual highlight of the prespecified setting, not an uncertainty interval.
        ax.axvspan(0.70, 0.80, color="#9A9A9A", alpha=0.16, linewidth=0, zorder=0)
        ax.axvline(PRIMARY_SIGMA_REF, color="#666666", linestyle="--", linewidth=0.9)
        ax.axhline(0.0, color="#777777", linewidth=0.8)
        ax.text(
            PRIMARY_SIGMA_REF,
            0.96,
            "primary $\\sigma$",
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            color="#555555",
            fontsize=6.0,
        )
        ax.set_title(f"{condition} ({'248 nm' if condition == 'UV' else '976 nm'})")
        ax.text(-0.12, 1.03, f"({chr(97 + panel)})", transform=ax.transAxes, fontweight="bold")
        ax.set_xticks([0.0, 0.5, 0.75, 1.0])
        ax.set_xlim(-0.03, 1.03)
        polish_axis(ax)

    span = max(abs(float(np.min(changes))), abs(float(np.max(changes))))
    axes[0].set_ylim(-max(27.0, span * 1.08), max(15.0, span * 0.62))
    axes[0].set_ylabel("Change from $\\sigma$ = 0.75 (%)")
    fig.supxlabel("Gaussian $\\sigma$ (reference px)", y=0.055)
    handles = [
        Line2D(
            [0],
            [0],
            color="#555555",
            linestyle=LINESTYLES[i],
            marker=MARKERS[i],
            markerfacecolor="white",
            markersize=4.2,
            label=f"X{i + 1:02d}",
        )
        for i in range(5)
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=5)
    fig.text(
        0.5,
        0.012,
        "Shading identifies the primary setting; it is not an uncertainty interval.",
        ha="center",
        va="bottom",
        fontsize=5.8,
        color="#555555",
    )
    fig.subplots_adjust(left=0.09, right=0.99, bottom=0.20, top=0.79, wspace=0.12)
    return save(fig, "Figure_S12_smoothing_combined")


def main() -> None:
    setup_style()
    rois, measurements, profiles, sensitivity, summary = load_inputs()
    order = validate_inputs(rois, measurements, profiles, sensitivity, summary)
    make_s9(order, measurements, profiles, summary)
    make_s10(order, measurements, profiles)
    make_s11(order, measurements, sensitivity)
    make_s12(order, sensitivity)


if __name__ == "__main__":
    main()
