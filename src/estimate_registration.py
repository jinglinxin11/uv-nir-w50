"""Reconstruct and validate the frozen registration before width analysis.

The high-resolution NIR similarity transform is independently re-estimated
from released image inputs. The historical UV rotation/translation cannot be
independently reconstructed because its upstream control points were not
released, so its dimension-derived scale is reconstructed and the frozen
mapping is validated with feature/RANSAC and phase-correlation diagnostics.
Quantitative analysis continues only when all checks pass.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measurement_core as core


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CFG = ROOT / "config" / "registration.json"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

NIR_PARAMETER_TOLERANCES = {
    "scale_abs": 1e-3,
    "rotation_deg_abs": 0.02,
    "tx_native_px_abs": 0.25,
    "ty_native_px_abs": 0.25,
}
MAX_FEATURE_MEDIAN_RESIDUAL_REF_PX = 3.0
MAX_PHASE_RESIDUAL_REF_PX = 1.0
MIN_RANSAC_INLIERS = 6
MIN_MEDIAN_STRUCTURAL_NCC = 0.40


def uv_on_common_grid(uv_rgb: np.ndarray, mapping: dict[str, Any]) -> np.ndarray:
    h = int(mapping["common_grid_height_px"])
    w = int(mapping["common_grid_width_px"])
    yy, xx = np.mgrid[:h, :w].astype(float)
    ux, uy = core.common_to_uv_native(xx, yy, mapping, uv_rgb.shape[:2])
    return np.stack(
        [core.sample_array(uv_rgb[..., channel], ux, uy) for channel in range(3)],
        axis=-1,
    )


def method_passes(record: dict[str, Any]) -> bool:
    residual = record.get("median_residual_common_px")
    return bool(
        record.get("ransac_valid", False)
        and int(record.get("ransac_inliers", 0)) >= MIN_RANSAC_INLIERS
        and residual is not None
        and float(residual) <= MAX_FEATURE_MEDIAN_RESIDUAL_REF_PX
    )


def validation_summary(record: dict[str, Any]) -> dict[str, Any]:
    methods = [name for name in ("SIFT", "ORB") if method_passes(record.get(name, {}))]
    phase = record["phase_correlation_residual"]
    phase_norm = float(np.hypot(phase["shift_x_common_px"], phase["shift_y_common_px"]))
    return {
        "passing_feature_methods": methods,
        "feature_validation_pass": bool(methods),
        "phase_residual_norm_reference_px": phase_norm,
        "phase_validation_pass": phase_norm <= MAX_PHASE_RESIDUAL_REF_PX,
    }


def structural_correlations(reference: np.ndarray, moving: np.ndarray) -> dict[str, float]:
    ref_maps = core.structure_maps(reference)
    moving_maps = core.structure_maps(moving)
    return {name: core.ncc(ref_maps[name], moving_maps[name]) for name in ref_maps}


def canonicalize_report(value: Any) -> Any:
    """Remove platform-level floating-point jitter from the audit artifact."""
    if isinstance(value, dict):
        return {key: canonicalize_report(item) for key, item in value.items()}
    if isinstance(value, list):
        return [canonicalize_report(item) for item in value]
    if isinstance(value, tuple):
        return [canonicalize_report(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), 10)
    return value


def contrast_stretch(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(x, [1, 99])
    return np.clip((x - lo) * 255.0 / max(float(hi - lo), core.EPS), 0, 255).astype(np.uint8)


def overlay(reference: np.ndarray, moving: np.ndarray) -> np.ndarray:
    ref_gray = contrast_stretch(cv2.cvtColor(np.clip(reference, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY))
    mov_gray = contrast_stretch(cv2.cvtColor(np.clip(moving, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY))
    out = np.zeros((*ref_gray.shape, 3), dtype=np.float32)
    out[..., 0] = ref_gray / 255.0
    out[..., 1] = mov_gray / 255.0
    return out


def make_qc_figure(
    reference: np.ndarray,
    nir_common: np.ndarray,
    uv_common: np.ndarray,
    report: dict[str, Any],
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 8.0), layout="constrained")
    rows = [
        ("Reconstructed NIR", nir_common, report["nir_reconstruction"]["validation_summary"]),
        ("Structurally checked frozen UV", uv_common, report["uv_mapping_validation"]["validation_summary"]),
    ]
    for row, (label, moving, summary) in enumerate(rows):
        axes[row, 0].imshow(contrast_stretch(reference))
        axes[row, 0].set_title("Common-grid NIR reference")
        axes[row, 1].imshow(contrast_stretch(moving))
        axes[row, 1].set_title(label)
        axes[row, 2].imshow(overlay(reference, moving))
        methods = ", ".join(summary["passing_feature_methods"]) or "none"
        axes[row, 2].set_title(
            f"Red=reference, green=mapped\n"
            f"RANSAC: {methods}; phase residual={summary['phase_residual_norm_reference_px']:.3f} px"
        )
        for ax in axes[row]:
            ax.axis("off")
    fig.suptitle("Registration reconstruction and validation (contrast-stretched display)", fontsize=14)
    fig.savefig(FIGURES / "Registration_QC.png", dpi=250, facecolor="white")
    plt.close(fig)


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    cv2.setRNGSeed(core.SEED)

    frozen = json.loads(CFG.read_text(encoding="utf-8"))
    uv_rgb = core.load_rgb(DATA / "UV_248T.tif")
    reference_rgb = core.load_rgb(DATA / "NIR_976-T_display.tif")
    high_nir_rgb = core.load_rgb(DATA / "NIR_highres_unsharpened_original.jpg")
    common_h, common_w = reference_rgb.shape[:2]
    expected_shape = (int(frozen["common_grid_height_px"]), int(frozen["common_grid_width_px"]))
    if (common_h, common_w) != expected_shape:
        raise RuntimeError(f"Common-grid reference shape changed: {(common_h, common_w)} != {expected_shape}")

    coarse, coarse_top = core.coarse_template_registration(high_nir_rgb, reference_rgb, "legacy_NIR")
    initial = core.similarity_from_coarse(coarse, reference_rgb.shape[:2])
    estimated, refinement = core.refine_similarity(high_nir_rgb, reference_rgb, initial)
    nir_common = core.warp_high_to_common(high_nir_rgb, estimated, reference_rgb.shape[:2])
    cv2.setRNGSeed(core.SEED)
    nir_features = core.feature_validation(reference_rgb, nir_common)
    nir_validation = validation_summary(nir_features)

    frozen_nir = frozen["nir_similarity_common_to_original_native"]
    parameter_delta = {
        "scale_abs": abs(estimated.scale - float(frozen_nir["scale"])),
        "rotation_deg_abs": abs(estimated.rotation_deg - float(frozen_nir["rotation_deg"])),
        "tx_native_px_abs": abs(estimated.tx - float(frozen_nir["tx"])),
        "ty_native_px_abs": abs(estimated.ty - float(frozen_nir["ty"])),
    }
    parameter_checks = {
        key: value <= NIR_PARAMETER_TOLERANCES[key]
        for key, value in parameter_delta.items()
    }

    uv_mapping = {
        **frozen["uv_mapping"],
        "common_grid_width_px": common_w,
        "common_grid_height_px": common_h,
    }
    reconstructed_uv_scale = {
        "resize_scale_x": common_w / uv_rgb.shape[1],
        "resize_scale_y": common_h / uv_rgb.shape[0],
    }
    uv_scale_checks = {
        key: bool(np.isclose(value, float(frozen["uv_mapping"][key]), rtol=0, atol=1e-12))
        for key, value in reconstructed_uv_scale.items()
    }
    uv_common = uv_on_common_grid(uv_rgb, uv_mapping)
    cv2.setRNGSeed(core.SEED)
    uv_features = core.feature_validation(reference_rgb, uv_common)
    uv_validation = validation_summary(uv_features)
    uv_correlations = structural_correlations(reference_rgb, uv_common)
    uv_median_correlation = float(np.median(list(uv_correlations.values())))

    checks = {
        "nir_optimizer_success": bool(refinement["optimizer_success"]),
        "nir_parameter_reconstruction": all(parameter_checks.values()),
        "nir_feature_ransac_validation": nir_validation["feature_validation_pass"],
        "nir_phase_validation": nir_validation["phase_validation_pass"],
        "uv_dimension_scale_reconstruction": all(uv_scale_checks.values()),
        "uv_structural_similarity_validation": uv_median_correlation >= MIN_MEDIAN_STRUCTURAL_NCC,
        "uv_phase_validation": uv_validation["phase_validation_pass"],
    }
    report = {
        "seed": core.SEED,
        "common_grid_shape_yx": [common_h, common_w],
        "registration_policy": {
            "nir": "independently re-estimated, compared with frozen transform, then validated",
            "uv": "dimension-derived scale reconstructed; frozen historical rotation/translation checked by structural and phase diagnostics",
            "analysis": "verified frozen transforms retained for exact released-result reproduction",
        },
        "thresholds": {
            "nir_parameter_tolerances": NIR_PARAMETER_TOLERANCES,
            "minimum_ransac_inliers": MIN_RANSAC_INLIERS,
            "maximum_feature_median_residual_reference_px": MAX_FEATURE_MEDIAN_RESIDUAL_REF_PX,
            "maximum_phase_residual_reference_px": MAX_PHASE_RESIDUAL_REF_PX,
            "minimum_median_structural_ncc": MIN_MEDIAN_STRUCTURAL_NCC,
        },
        "nir_reconstruction": {
            "coarse_best": coarse,
            "coarse_top_candidates": coarse_top,
            "initial_similarity": initial.__dict__,
            "estimated_similarity": estimated.__dict__,
            "frozen_similarity": frozen_nir,
            "absolute_parameter_delta": parameter_delta,
            "parameter_checks": parameter_checks,
            "continuous_refinement": refinement,
            "feature_ransac_and_phase_validation": nir_features,
            "validation_summary": nir_validation,
        },
        "uv_mapping_validation": {
            "reconstruction_scope": "scale only; upstream control points for historical rotation/translation are not released",
            "dimension_derived_scale": reconstructed_uv_scale,
            "frozen_mapping": frozen["uv_mapping"],
            "scale_checks": uv_scale_checks,
            "feature_ransac_and_phase_validation": uv_features,
            "feature_ransac_interpretation": "diagnostic only; a passing residual transform is not required because upstream UV control points are unavailable",
            "structural_correlations": uv_correlations,
            "median_structural_correlation": uv_median_correlation,
            "validation_summary": uv_validation,
        },
        "checks": checks,
        "overall_pass": all(checks.values()),
    }
    report = canonicalize_report(report)
    core.write_json(TABLES / "registration_reconstruction.json", report)
    make_qc_figure(reference_rgb, nir_common, uv_common, report)

    print(json.dumps({
        "overall_pass": report["overall_pass"],
        "nir_estimated_similarity": estimated.__dict__,
        "nir_absolute_parameter_delta": parameter_delta,
        "nir_passing_ransac_methods": nir_validation["passing_feature_methods"],
        "uv_passing_ransac_methods": uv_validation["passing_feature_methods"],
        "uv_median_structural_correlation": uv_median_correlation,
        "nir_phase_residual_reference_px": nir_validation["phase_residual_norm_reference_px"],
        "uv_phase_residual_reference_px": uv_validation["phase_residual_norm_reference_px"],
    }, indent=2))
    if not report["overall_pass"]:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"Registration reconstruction/validation failed: {failed}")


if __name__ == "__main__":
    main()
