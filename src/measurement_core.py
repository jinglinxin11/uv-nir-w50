"""Shared numerical core for the released fixed-transect UV/NIR analysis.

The geometric transform is estimated exclusively from low-frequency structure.
No FWHM, UV/NIR ratio, sharpening, deconvolution, or generative processing enters
registration or ROI selection.  Reported distances are common-reference-grid
pixels because no sample-level physical calibration is available.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage, optimize
from skimage import color, exposure, restoration


SEED = 20260816
EPS = 1e-9
PRIMARY_SIGMA_REF = 0.75
PRIMARY_HALF_LEVEL = 0.50
PRIMARY_ORIENTATION_TOL = 15.0
PROFILE_HALF_LENGTH_REF = 22.0
PROFILE_STEP_REF = 0.20
PROFILE_COUNT = 11
PROFILE_SPACING_REF = 1.0
BACKGROUND_FRACTION = 0.18
MIN_CNR = 3.0


@dataclass(frozen=True)
class Similarity:
    scale: float
    rotation_deg: float
    tx: float
    ty: float

    def matrix(self) -> np.ndarray:
        a = math.radians(self.rotation_deg)
        c, s = math.cos(a), math.sin(a)
        return np.array(
            [[self.scale * c, -self.scale * s, self.tx],
             [self.scale * s, self.scale * c, self.ty]], dtype=float
        )

    def map(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m = self.matrix()
        return m[0, 0] * x + m[0, 1] * y + m[0, 2], m[1, 0] * x + m[1, 1] * y + m[1, 2]


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or isinstance(value, str):
        return value
    return str(value)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_ready(value), ensure_ascii=False, indent=2), encoding="utf-8")


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def robust_standardize(a: np.ndarray, clip: float = 6.0) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    med = float(np.nanmedian(a))
    mad = float(np.nanmedian(np.abs(a - med)))
    z = (a - med) / max(1.4826 * mad, 1e-6)
    return np.clip(z, -clip, clip).astype(np.float32)


def rgb_channel_maps(rgb: np.ndarray, local_sigma_native: float) -> dict[str, np.ndarray]:
    x = np.clip(rgb.astype(np.float32), 0, 255)
    r, g, b = x[..., 0], x[..., 1], x[..., 2]
    gray = cv2.cvtColor(x.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    try:
        lab_a = color.rgb2lab(x / 255.0)[..., 1].astype(np.float32)
    except ValueError:
        # OpenCV's 8-bit Lab-a is an affine encoding of the same opponent-
        # color axis. ROI normalization makes its offset/scale irrelevant to
        # W50, while avoiding a rare skimage failure on large JPEG round-trips.
        lab_a = cv2.cvtColor(
            np.ascontiguousarray(x.astype(np.uint8)), cv2.COLOR_RGB2LAB
        )[..., 1].astype(np.float32)
    denom = np.maximum(r + g + b, 1.0)
    darkness = 255.0 - gray
    smooth_gray = ndimage.gaussian_filter(gray, max(local_sigma_native, 0.5), mode="nearest")
    local_darkness = smooth_gray - gray
    bg_gray = float(np.percentile(gray, 90))
    optical_density = -np.log(np.maximum(gray, 1.0) / max(bg_gray, 1.0))
    chroma = x / denom[..., None]
    return {
        "ExR": (2.0 * r - g - b).astype(np.float32),
        "Lab_a": lab_a,
        "red_fraction": (r / denom).astype(np.float32),
        "darkness": darkness,
        "local_darkness": local_darkness.astype(np.float32),
        "optical_density": optical_density.astype(np.float32),
        "chroma_r": chroma[..., 0].astype(np.float32),
        "chroma_g": chroma[..., 1].astype(np.float32),
        "chroma_b": chroma[..., 2].astype(np.float32),
    }


def structure_maps(rgb: np.ndarray, work_factor: float = 1.0) -> dict[str, np.ndarray]:
    if work_factor != 1.0:
        rgb = cv2.resize(rgb, None, fx=work_factor, fy=work_factor, interpolation=cv2.INTER_AREA)
    x = np.clip(rgb, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lab_a = cv2.cvtColor(x, cv2.COLOR_RGB2LAB)[..., 1].astype(np.float32)
    darkness = 255.0 - gray
    sigma_small = max(0.8, 2.0 * work_factor)
    sigma_large = max(6.0, 22.0 * work_factor)
    dog_dark = cv2.GaussianBlur(darkness, (0, 0), sigma_small) - cv2.GaussianBlur(darkness, (0, 0), sigma_large)
    dog_a = cv2.GaussianBlur(lab_a, (0, 0), sigma_small) - cv2.GaussianBlur(lab_a, (0, 0), sigma_large)
    base = cv2.GaussianBlur(gray, (0, 0), max(0.8, 1.5 * work_factor))
    gx = cv2.Scharr(base, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(base, cv2.CV_32F, 0, 1)
    grad = cv2.GaussianBlur(cv2.magnitude(gx, gy), (0, 0), max(0.8, 1.2 * work_factor))
    return {"darkness_DoG": robust_standardize(dog_dark), "Lab_a_DoG": robust_standardize(dog_a), "Scharr": robust_standardize(grad)}


def ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    aa, bb = np.asarray(a, float), np.asarray(b, float)
    good = np.isfinite(aa) & np.isfinite(bb)
    if mask is not None:
        good &= mask
    if good.sum() < 200:
        return float("nan")
    av, bv = aa[good], bb[good]
    av -= av.mean(); bv -= bv.mean()
    den = math.sqrt(float(np.dot(av, av) * np.dot(bv, bv)))
    return float(np.dot(av, bv) / den) if den > EPS else float("nan")


def mutual_information(a: np.ndarray, b: np.ndarray, bins: int = 48) -> float:
    aa, bb = np.asarray(a).ravel(), np.asarray(b).ravel()
    good = np.isfinite(aa) & np.isfinite(bb)
    aa, bb = aa[good], bb[good]
    if len(aa) < 200:
        return float("nan")
    h, _, _ = np.histogram2d(aa, bb, bins=bins)
    p = h / max(h.sum(), 1.0)
    px, py = p.sum(1), p.sum(0)
    expected = px[:, None] * py[None, :]
    nz = p > 0
    return float(np.sum(p[nz] * np.log(p[nz] / np.maximum(expected[nz], EPS))))


def sample_array(image: np.ndarray, x: np.ndarray, y: np.ndarray, order: int = 1) -> np.ndarray:
    return ndimage.map_coordinates(image, [y, x], order=order, mode="nearest", prefilter=False)


def warp_high_to_common(image: np.ndarray, transform: Similarity, shape: tuple[int, int]) -> np.ndarray:
    yy, xx = np.mgrid[:shape[0], :shape[1]].astype(float)
    hx, hy = transform.map(xx, yy)
    if image.ndim == 2:
        return sample_array(image, hx, hy)
    return np.stack([sample_array(image[..., c], hx, hy) for c in range(image.shape[2])], axis=-1)


def coarse_template_registration(parent_rgb: np.ndarray, ref_rgb: np.ndarray, label: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    factor = 0.25
    parent_work = cv2.resize(parent_rgb, (int(round(parent_rgb.shape[1] * factor)), int(round(parent_rgb.shape[0] * factor))), interpolation=cv2.INTER_AREA)
    p_maps = structure_maps(parent_work)
    records: list[dict[str, Any]] = []
    for angle in np.arange(-3.0, 3.01, 0.5):
        # Derive a dimension-compatible scale grid so the same structural
        # registration works for native high-resolution inputs and for
        # faithfully downsampled exports.  This range is fixed before any
        # width measurement and uses image dimensions only.
        max_fit_scale = min(
            parent_rgb.shape[1] / ref_rgb.shape[1],
            parent_rgb.shape[0] / ref_rgb.shape[0],
        )
        min_fit_scale = max(0.5, 0.55 * max_fit_scale)
        n_scale_steps = max(25, int(math.ceil((max_fit_scale - min_fit_scale) / 0.08)) + 1)
        scale_grid = np.linspace(min_fit_scale, 0.998 * max_fit_scale, n_scale_steps)
        for scale in scale_grid:
            tw = int(round(ref_rgb.shape[1] * scale * factor))
            th = int(round(ref_rgb.shape[0] * scale * factor))
            if tw < 80 or th < 100 or tw >= parent_work.shape[1] or th >= parent_work.shape[0]:
                continue
            resized = cv2.resize(ref_rgb, (tw, th), interpolation=cv2.INTER_CUBIC)
            r_maps = structure_maps(resized)
            center = ((tw - 1) / 2.0, (th - 1) / 2.0)
            rot = cv2.getRotationMatrix2D(center, angle, 1.0)
            wy, wx = np.hanning(th).astype(np.float32), np.hanning(tw).astype(np.float32)
            taper = np.sqrt(np.outer(wy, wx))
            channel_scores: dict[str, float] = {}
            best_locs: list[tuple[int, int]] = []
            for name in p_maps:
                templ = cv2.warpAffine(r_maps[name], rot, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) * taper
                response = cv2.matchTemplate(p_maps[name], templ, cv2.TM_CCOEFF_NORMED)
                _, mv, _, loc = cv2.minMaxLoc(response)
                channel_scores[name] = float(mv)
                best_locs.append(loc)
            loc = tuple(np.median(np.asarray(best_locs), axis=0).astype(int))
            combined = float(np.median(list(channel_scores.values())))
            records.append({
                "reference": label, "scale": float(scale), "rotation_deg": float(angle),
                "x_work": int(loc[0]), "y_work": int(loc[1]), "width_work": tw, "height_work": th,
                "work_factor": factor, "combined_score": combined, **{f"score_{k}": v for k, v in channel_scores.items()},
            })
    records.sort(key=lambda r: r["combined_score"], reverse=True)
    return records[0], records[:40]


def similarity_from_coarse(row: dict[str, Any], ref_shape: tuple[int, int]) -> Similarity:
    f = float(row["work_factor"])
    x0, y0 = row["x_work"] / f, row["y_work"] / f
    w, h = row["width_work"] / f, row["height_work"] / f
    scale, angle = float(row["scale"]), float(row["rotation_deg"])
    cx, cy = (ref_shape[1] - 1) / 2.0, (ref_shape[0] - 1) / 2.0
    a = math.radians(angle); c, s = math.cos(a), math.sin(a)
    mapped_cx, mapped_cy = scale * (c * cx - s * cy), scale * (s * cx + c * cy)
    return Similarity(scale, angle, x0 + (w - 1) / 2.0 - mapped_cx, y0 + (h - 1) / 2.0 - mapped_cy)


def refine_similarity(parent_rgb: np.ndarray, ref_rgb: np.ndarray, initial: Similarity) -> tuple[Similarity, dict[str, Any]]:
    work = 0.5
    parent = cv2.resize(parent_rgb, None, fx=work, fy=work, interpolation=cv2.INTER_AREA)
    p_maps = structure_maps(parent)
    r_maps = structure_maps(ref_rgb)
    h, w = ref_rgb.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(float)
    margin = 14
    mask = np.zeros((h, w), bool); mask[margin:-margin, margin:-margin] = True

    def unpack(p: np.ndarray) -> Similarity:
        return Similarity(float(math.exp(p[0])), float(p[1]), float(p[2]), float(p[3]))

    def objective(p: np.ndarray) -> float:
        t = unpack(p)
        hx, hy = t.map(xx, yy)
        hx *= work; hy *= work
        scores = []
        for name in p_maps:
            sampled = sample_array(p_maps[name], hx, hy)
            scores.append(ncc(r_maps[name], sampled, mask))
        finite = [s for s in scores if np.isfinite(s)]
        return -float(np.median(finite)) if finite else 1.0

    p0 = np.array([math.log(initial.scale), initial.rotation_deg, initial.tx, initial.ty], float)
    bounds = [
        (math.log(initial.scale * 0.88), math.log(initial.scale * 1.12)),
        (initial.rotation_deg - 4.0, initial.rotation_deg + 4.0),
        (initial.tx - 220.0, initial.tx + 220.0),
        (initial.ty - 160.0, initial.ty + 160.0),
    ]
    result = optimize.minimize(objective, p0, method="Powell", bounds=bounds, options={"xtol": 1e-4, "ftol": 1e-5, "maxiter": 120})
    refined = unpack(result.x)
    warped_maps = {name: warp_high_to_common(m, Similarity(refined.scale * work, refined.rotation_deg, refined.tx * work, refined.ty * work), (h, w)) for name, m in p_maps.items()}
    correlations = {name: ncc(r_maps[name], warped_maps[name], mask) for name in r_maps}
    mi = {name: mutual_information(r_maps[name][mask], warped_maps[name][mask]) for name in r_maps}
    return refined, {"optimizer_success": bool(result.success), "optimizer_message": str(result.message), "objective": float(result.fun), "iterations": int(result.nit), "correlations": correlations, "mutual_information": mi}


def feature_validation(ref_rgb: np.ndarray, warped_high_rgb: np.ndarray) -> dict[str, Any]:
    ref = exposure.rescale_intensity(structure_maps(ref_rgb)["darkness_DoG"], out_range=(0, 255)).astype(np.uint8)
    mov = exposure.rescale_intensity(structure_maps(warped_high_rgb)["darkness_DoG"], out_range=(0, 255)).astype(np.uint8)
    out: dict[str, Any] = {}
    for method in ["SIFT", "ORB"]:
        detector = cv2.SIFT_create(nfeatures=1000, contrastThreshold=0.01) if method == "SIFT" else cv2.ORB_create(nfeatures=1500, fastThreshold=5)
        k1, d1 = detector.detectAndCompute(ref, None); k2, d2 = detector.detectAndCompute(mov, None)
        if d1 is None or d2 is None or len(k1) < 4 or len(k2) < 4:
            out[method] = {"keypoints_reference": len(k1), "keypoints_warped": len(k2), "matches": 0}
            continue
        norm = cv2.NORM_L2 if method == "SIFT" else cv2.NORM_HAMMING
        pairs = cv2.BFMatcher(norm).knnMatch(d1, d2, k=2)
        good = [a for a, b in pairs if a.distance < 0.78 * b.distance]
        if len(good) < 4:
            out[method] = {"keypoints_reference": len(k1), "keypoints_warped": len(k2), "matches": len(good)}
            continue
        p1 = np.float32([k1[m.queryIdx].pt for m in good]); p2 = np.float32([k2[m.trainIdx].pt for m in good])
        mat, inliers = cv2.estimateAffinePartial2D(p1, p2, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=5000)
        if mat is None:
            out[method] = {"keypoints_reference": len(k1), "keypoints_warped": len(k2), "matches": len(good), "ransac_inliers": 0}
            continue
        pred = cv2.transform(p1[:, None, :], mat)[:, 0, :]
        residual = np.linalg.norm(pred - p2, axis=1)
        inlier_mask = inliers.ravel().astype(bool)
        singular_values = np.linalg.svd(mat[:, :2], compute_uv=False)
        inlier_span = np.ptp(p1[inlier_mask], axis=0) if inlier_mask.any() else np.zeros(2)
        residual_rotation_deg = float(math.degrees(math.atan2(mat[1, 0], mat[0, 0])))
        residual_translation_norm = float(np.linalg.norm(mat[:, 2]))
        degenerate = bool(
            singular_values.min() < 0.5
            or singular_values.max() > 2.0
            or inlier_span.min() < 10.0
        )
        near_identity = bool(
            np.max(np.abs(singular_values - 1.0)) <= 0.05
            and abs(residual_rotation_deg) <= 1.0
            and residual_translation_norm <= 3.0
        )
        out[method] = {
            "keypoints_reference": len(k1), "keypoints_warped": len(k2), "matches": len(good),
            "ransac_inliers": int(inlier_mask.sum()), "median_residual_common_px": float(np.median(residual[inlier_mask])) if inlier_mask.any() else None,
            "p95_residual_common_px": float(np.percentile(residual[inlier_mask], 95)) if inlier_mask.any() else None,
            "residual_affine_reference_to_warped": mat.tolist(),
            "affine_singular_values": singular_values.tolist(),
            "residual_rotation_deg": residual_rotation_deg,
            "residual_translation_norm_common_px": residual_translation_norm,
            "inlier_span_reference_px": inlier_span.tolist(),
            "degenerate_transform": degenerate,
            "residual_transform_near_identity": near_identity,
            "ransac_valid": bool(inlier_mask.sum() >= 4 and not degenerate and near_identity),
        }
    shift, response = cv2.phaseCorrelate(ref.astype(np.float32), mov.astype(np.float32))
    out["phase_correlation_residual"] = {"shift_x_common_px": float(shift[0]), "shift_y_common_px": float(shift[1]), "response": float(response)}
    return out


def common_to_uv_native(x: np.ndarray, y: np.ndarray, baseline_registration: dict[str, Any], uv_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of baseline resize -> rotate -> shift, with pixel-center scaling."""
    sx = float(baseline_registration["resize_scale_x"])
    sy = float(baseline_registration["resize_scale_y"])
    angle = math.radians(float(baseline_registration["angle_deg"]))
    shift_x = float(baseline_registration["shift_x_px"])
    shift_y = float(baseline_registration["shift_y_px"])
    common_h, common_w = int(baseline_registration["common_grid_height_px"]), int(baseline_registration["common_grid_width_px"])
    cx, cy = (common_w - 1) / 2.0, (common_h - 1) / 2.0
    qx, qy = x - shift_x - cx, y - shift_y - cy
    c, s = math.cos(-angle), math.sin(-angle)
    rx, ry = c * qx - s * qy + cx, s * qx + c * qy + cy
    ux = (rx + 0.5) / sx - 0.5
    uy = (ry + 0.5) / sy - 0.5
    return np.clip(ux, 0, uv_shape[1] - 1), np.clip(uy, 0, uv_shape[0] - 1)


def profile_coordinates(cx: float, cy: float, normal_deg: float, tangent_offset: float, distance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = math.radians(normal_deg)
    nx, ny = math.cos(a), math.sin(a)
    tx, ty = -ny, nx
    return cx + tangent_offset * tx + distance * nx, cy + tangent_offset * ty + distance * ny


def sample_profile_from_common(
    image: np.ndarray,
    source: str,
    cx: float,
    cy: float,
    normal_deg: float,
    tangent_offset: float,
    distance: np.ndarray,
    high_transform: Similarity,
    baseline_reg: dict[str, Any],
) -> np.ndarray:
    x, y = profile_coordinates(cx, cy, normal_deg, tangent_offset, distance)
    if source == "high":
        sx, sy = high_transform.map(x, y)
    elif source == "uv":
        sx, sy = common_to_uv_native(x, y, baseline_reg, image.shape)
    elif source == "old_nir":
        sx, sy = x, y
    else:
        raise ValueError(source)
    return sample_array(image, sx, sy)


def monotonic_crossing(x: np.ndarray, y: np.ndarray, threshold: float) -> float:
    yy = np.maximum.accumulate(np.asarray(y, float))
    if not np.isfinite(yy).any() or threshold < np.nanmin(yy) or threshold > np.nanmax(yy):
        return float("nan")
    idx = int(np.searchsorted(yy, threshold, side="left"))
    if idx <= 0:
        return float(x[0])
    if idx >= len(yy):
        return float(x[-1])
    y0, y1 = yy[idx - 1], yy[idx]
    if y1 == y0:
        return float(x[idx])
    return float(x[idx - 1] + (threshold - y0) * (x[idx] - x[idx - 1]) / (y1 - y0))


def filter_profile(raw: np.ndarray, distance: np.ndarray, method: str, sigma_ref: float) -> np.ndarray:
    step = float(np.median(np.diff(distance)))
    if method == "none" or sigma_ref <= 0:
        return raw.copy()
    if method == "gaussian":
        return ndimage.gaussian_filter1d(raw, sigma_ref / step, mode="nearest")
    if method == "median":
        size = max(3, int(round(2 * sigma_ref / step)) | 1)
        return ndimage.median_filter(raw, size=size, mode="nearest")
    if method == "bilateral":
        radius = max(2, int(math.ceil(3 * sigma_ref / step)))
        out = np.empty_like(raw)
        robust_noise = max(1.4826 * np.median(np.abs(raw - np.median(raw))), EPS)
        for i in range(len(raw)):
            lo, hi = max(0, i - radius), min(len(raw), i + radius + 1)
            j = np.arange(lo, hi)
            ws = np.exp(-0.5 * ((j - i) * step / max(sigma_ref, step)) ** 2)
            wr = np.exp(-0.5 * ((raw[j] - raw[i]) / max(1.5 * robust_noise, EPS)) ** 2)
            out[i] = np.sum(raw[j] * ws * wr) / max(np.sum(ws * wr), EPS)
        return out
    if method == "tv":
        scale = max(float(np.nanstd(raw)), EPS)
        return restoration.denoise_tv_chambolle((raw - np.nanmin(raw)) / scale, weight=0.035, channel_axis=None) * scale + np.nanmin(raw)
    if method == "wavelet":
        try:
            return restoration.denoise_wavelet(raw, method="BayesShrink", mode="soft", rescale_sigma=True, channel_axis=None)
        except Exception:
            return ndimage.gaussian_filter1d(raw, max(0.5, sigma_ref / step), mode="nearest")
    raise ValueError(method)


def analyze_mean_profile(
    distance: np.ndarray,
    raw: np.ndarray,
    sigma_ref: float = PRIMARY_SIGMA_REF,
    half_level: float = PRIMARY_HALF_LEVEL,
    filter_method: str = "gaussian",
) -> dict[str, Any]:
    raw = np.asarray(raw, float)
    filtered = filter_profile(raw, distance, filter_method, sigma_ref)
    n_bg = max(5, int(BACKGROUND_FRACTION * len(raw)))
    bg = np.r_[raw[:n_bg], raw[-n_bg:]]
    baseline = float(np.median(bg))
    mad = float(np.median(np.abs(bg - baseline)))
    noise = max(float(np.std(bg, ddof=1)), 1.4826 * mad, EPS)
    center_mask = np.abs(distance) <= PROFILE_HALF_LENGTH_REF * 0.42
    center_idx = np.flatnonzero(center_mask)
    peak_idx = int(center_idx[np.argmax(filtered[center_mask])])
    peak = float(filtered[peak_idx]); amplitude = peak - baseline
    ptn = amplitude / noise
    norm = (filtered - baseline) / amplitude if amplitude > EPS else np.full_like(filtered, np.nan)
    invalid = {"valid": False, "FWHM": np.nan, "edge_width": np.nan, "CNR": np.nan,
               "peak_to_noise": ptn, "background": baseline, "background_SD": noise,
               "raw": raw, "filtered": filtered, "normalized": norm}
    if amplitude <= 0 or ptn < MIN_CNR:
        return {**invalid, "flag": f"insufficient peak-to-noise ({ptn:.2f})"}
    lx, ly = distance[:peak_idx + 1], norm[:peak_idx + 1]
    rx, ry = distance[peak_idx:][::-1], norm[peak_idx:][::-1]
    levels = sorted(set([0.1, 0.5, 0.9, float(half_level)]))
    left = {v: monotonic_crossing(lx, ly, v) for v in levels}
    right = {v: monotonic_crossing(rx, ry, v) for v in levels}
    if not all(np.isfinite(v) for v in [*left.values(), *right.values()]):
        return {**invalid, "flag": "one or more crossings absent"}
    fwhm = abs(right[float(half_level)] - left[float(half_level)])
    edge = float(np.mean([abs(left[0.9] - left[0.1]), abs(right[0.9] - right[0.1])]))
    core = raw[norm >= 0.5]
    cnr = float(abs(np.mean(core) - np.mean(bg)) / math.sqrt(max((np.var(core, ddof=1) + np.var(bg, ddof=1)) / 2, EPS))) if len(core) > 1 else np.nan
    grad = float(np.nanmax(np.abs(np.gradient(norm, distance))))
    return {
        "valid": True, "flag": "", "FWHM": float(fwhm), "edge_width": edge, "CNR": cnr,
        "peak_to_noise": ptn, "background": baseline, "background_SD": noise, "peak": peak,
        "x50_left": left.get(0.5), "x50_right": right.get(0.5), "xhalf_left": left[float(half_level)],
        "xhalf_right": right[float(half_level)], "max_gradient": grad,
        "raw": raw, "filtered": filtered, "normalized": norm,
    }


def roi_profile(
    image: np.ndarray,
    source: str,
    roi: dict[str, Any],
    distance: np.ndarray,
    high_transform: Similarity,
    baseline_reg: dict[str, Any],
    sigma_ref: float = PRIMARY_SIGMA_REF,
    half_level: float = PRIMARY_HALF_LEVEL,
    filter_method: str = "gaussian",
) -> tuple[dict[str, Any], np.ndarray]:
    offsets = np.arange(-(PROFILE_COUNT // 2), PROFILE_COUNT // 2 + 1) * PROFILE_SPACING_REF
    stack = np.stack([
        sample_profile_from_common(image, source, roi["center_x"], roi["center_y"], roi["normal_angle_deg"], float(off), distance, high_transform, baseline_reg)
        for off in offsets
    ])
    mean = np.mean(stack, axis=0)
    result = analyze_mean_profile(distance, mean, sigma_ref=sigma_ref, half_level=half_level, filter_method=filter_method)
    result["n_profiles_averaged"] = int(len(stack))
    result["within_stack_SD_median"] = float(np.median(np.std(stack, axis=0, ddof=1)))
    return result, stack


def direction_from_normal(normal_deg: float, tolerance: float = PRIMARY_ORIENTATION_TOL) -> str:
    a = normal_deg % 180.0
    if min(a, 180.0 - a) <= tolerance:
        return "X"
    if abs(a - 90.0) <= tolerance:
        return "Y"
    return "Diagonal"


def measure_rois(
    rois: list[dict[str, Any]],
    uv_map: np.ndarray,
    nir_map: np.ndarray,
    nir_source: str,
    distance: np.ndarray,
    transform: Similarity,
    baseline_reg: dict[str, Any],
    sigma_ref: float = PRIMARY_SIGMA_REF,
    half_level: float = PRIMARY_HALF_LEVEL,
    filter_method: str = "gaussian",
    channel: str = "ExR",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    roi_rows: list[dict[str, Any]] = []
    profile_rows: list[dict[str, Any]] = []
    cache: dict[tuple[str, str], dict[str, Any]] = {}
    for roi in rois:
        for condition, image, source in [("UV", uv_map, "uv"), ("NIR", nir_map, nir_source)]:
            res, stack = roi_profile(image, source, roi, distance, transform, baseline_reg, sigma_ref=sigma_ref, half_level=half_level, filter_method=filter_method)
            cache[(condition, roi["ROI_ID"])] = {**res, "stack": stack}
            roi_rows.append({
                "condition": condition, "ROI_ID": roi["ROI_ID"], "candidate_ID": roi["candidate_ID"],
                "center_x": roi["center_x"], "center_y": roi["center_y"], "tangent_angle_deg": roi["tangent_angle_deg"],
                "normal_angle_deg": roi["normal_angle_deg"], "Direction": direction_from_normal(roi["normal_angle_deg"]),
                "channel": channel, "filter_method": filter_method, "smoothing_sigma_ref_px": sigma_ref, "half_level": half_level,
                "valid": res["valid"], "flag": res.get("flag", ""), "FWHM": res.get("FWHM"), "edge_width": res.get("edge_width"),
                "CNR": res.get("CNR"), "peak_to_noise": res.get("peak_to_noise"), "background": res.get("background"),
                "background_SD": res.get("background_SD"), "max_gradient": res.get("max_gradient"),
                "x50_left": res.get("x50_left"), "x50_right": res.get("x50_right"),
                "n_profiles_averaged": PROFILE_COUNT, "within_stack_SD_median": res.get("within_stack_SD_median"),
                "unit": "common-reference-grid apparent pixels",
            })
            for i, d in enumerate(distance):
                profile_rows.append({"condition": condition, "ROI_ID": roi["ROI_ID"], "channel": channel,
                                     "distance_reference_px": float(d), "raw_mean": float(res["raw"][i]),
                                     "filtered": float(res["filtered"][i]), "normalized": float(res["normalized"][i]) if np.isfinite(res["normalized"][i]) else np.nan,
                                     "profile_stack_SD": float(np.std(stack[:, i], ddof=1))})
    return roi_rows, profile_rows, cache
