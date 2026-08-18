"""Directional UV/NIR apparent-FWHM analysis on native image grids.

The geometric transform is estimated exclusively from low-frequency structure.
No FWHM, UV/NIR ratio, sharpening, deconvolution, or generative processing enters
registration or ROI selection.  Reported distances are common-reference-grid
pixels because no sample-level physical calibration is available.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from PIL import Image
from scipy import ndimage, optimize, special
from skimage import color, exposure, feature, restoration


SEED = 20260816
RNG = np.random.default_rng(SEED)
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
MIN_COHERENCE = 0.22
MIN_BORDER_REF = 23.0
MAX_PER_DIRECTION = 8
MIN_ROI_SEPARATION_REF = 15.0


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


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


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


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def file_metadata(path: Path) -> dict[str, Any]:
    with Image.open(path) as im:
        qtables = getattr(im, "quantization", None)
        return {
            "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size,
            "format": im.format, "width": im.width, "height": im.height,
            "mode": im.mode, "dpi": list(im.info.get("dpi", [])),
            "compression": im.info.get("compression"), "icc_profile_present": bool(im.info.get("icc_profile")),
            "exif_bytes": len(im.info.get("exif", b"")),
            "jpeg_quantization_tables": qtables,
        }


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
        out[method] = {
            "keypoints_reference": len(k1), "keypoints_warped": len(k2), "matches": len(good),
            "ransac_inliers": int(inlier_mask.sum()), "median_residual_common_px": float(np.median(residual[inlier_mask])) if inlier_mask.any() else None,
            "p95_residual_common_px": float(np.percentile(residual[inlier_mask], 95)) if inlier_mask.any() else None,
            "residual_affine_reference_to_warped": mat.tolist(),
        }
    shift, response = cv2.phaseCorrelate(ref.astype(np.float32), mov.astype(np.float32))
    out["phase_correlation_residual"] = {"shift_x_common_px": float(shift[0]), "shift_y_common_px": float(shift[1]), "response": float(response)}
    return out


def jpeg_blockiness(gray: np.ndarray) -> dict[str, float]:
    gray = np.asarray(gray, float)
    xb = np.arange(8, gray.shape[1], 8); yb = np.arange(8, gray.shape[0], 8)
    xi = np.arange(5, gray.shape[1], 8); yi = np.arange(5, gray.shape[0], 8)
    v_boundary = np.mean(np.abs(gray[:, xb] - gray[:, xb - 1]))
    h_boundary = np.mean(np.abs(gray[yb, :] - gray[yb - 1, :]))
    v_internal = np.mean(np.abs(gray[:, xi] - gray[:, xi - 1]))
    h_internal = np.mean(np.abs(gray[yi, :] - gray[yi - 1, :]))
    boundary, internal = (v_boundary + h_boundary) / 2, (v_internal + h_internal) / 2
    return {"boundary_difference": float(boundary), "internal_difference": float(internal), "boundary_to_internal_ratio": float(boundary / max(internal, EPS))}


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


def validate_uv_inverse(uv_rgb: np.ndarray, baseline_registered_rgb: np.ndarray, reg: dict[str, Any]) -> dict[str, float]:
    h, w = baseline_registered_rgb.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(float)
    ux, uy = common_to_uv_native(xx, yy, reg, uv_rgb.shape[:2])
    rebuilt = np.stack([sample_array(uv_rgb[..., c], ux, uy) for c in range(3)], axis=-1)
    rmse = float(np.sqrt(np.mean((rebuilt - baseline_registered_rgb) ** 2)))
    corr = ncc(cv2.cvtColor(rebuilt.astype(np.uint8), cv2.COLOR_RGB2GRAY), cv2.cvtColor(baseline_registered_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY))
    return {"RMSE_RGB_0_255": rmse, "grayscale_correlation": corr}


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


def axial_distance(angle_deg: float, axis_deg: float) -> float:
    return abs(((angle_deg - axis_deg + 90.0) % 180.0) - 90.0)


def direction_from_normal(normal_deg: float, tolerance: float = PRIMARY_ORIENTATION_TOL) -> str:
    a = normal_deg % 180.0
    if min(a, 180.0 - a) <= tolerance:
        return "X"
    if abs(a - 90.0) <= tolerance:
        return "Y"
    return "Diagonal"


def local_orientation_coherence(image: np.ndarray, x: float, y: float, radius: int = 7) -> float:
    y0, y1 = max(0, int(y) - radius), min(image.shape[0], int(y) + radius + 1)
    x0, x1 = max(0, int(x) - radius), min(image.shape[1], int(x) + radius + 1)
    patch = image[y0:y1, x0:x1]
    if patch.size < 25:
        return 0.0
    gx = cv2.Sobel(patch.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    jxx, jyy, jxy = float(np.mean(gx * gx)), float(np.mean(gy * gy)), float(np.mean(gx * gy))
    disc = math.sqrt(max((jxx - jyy) ** 2 + 4 * jxy * jxy, 0.0))
    l1, l2 = (jxx + jyy + disc) / 2, (jxx + jyy - disc) / 2
    return float((l1 - l2) / max(l1 + l2, EPS))


def build_roi_candidates(old_nir_rgb: np.ndarray, high_common_rgb: np.ndarray) -> list[dict[str, Any]]:
    old_map = robust_standardize(rgb_channel_maps(old_nir_rgb, 8.0)["local_darkness"])
    high_map = robust_standardize(rgb_channel_maps(high_common_rgb, 8.0)["local_darkness"])
    structural = 0.5 * old_map + 0.5 * high_map
    lo, hi = np.percentile(structural, [2, 98])
    display = exposure.rescale_intensity(structural, in_range=(float(lo), float(hi)), out_range=(0, 255)).astype(np.uint8)
    edges = cv2.Canny(display, 38, 105, L2gradient=True)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=18, minLineLength=18, maxLineGap=5)
    raw: list[dict[str, Any]] = []
    if lines is not None:
        # OpenCV may return either (N, 1, 4) or, for a singleton/variant
        # build, an already squeezed (N, 4) array.  Normalizing the shape
        # leaves the detected coordinates unchanged and avoids a data-
        # dependent indexing failure on very low-contrast images.
        line_rows = np.asarray(lines).reshape(-1, 4)
        for idx, line in enumerate(line_rows):
            x1, y1, x2, y2 = map(float, line)
            length = float(math.hypot(x2 - x1, y2 - y1))
            if length < 18:
                continue
            tangent = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180.0
            n_points = max(1, int(length // 18))
            for k, f in enumerate(np.linspace(0.25, 0.75, n_points)):
                cx, cy = x1 + f * (x2 - x1), y1 + f * (y2 - y1)
                raw.append({"candidate_source": "Hough", "source_line": idx, "center_x": cx, "center_y": cy,
                            "tangent_angle_deg": tangent, "normal_angle_deg": (tangent + 90.0) % 180.0,
                            "line_length_ref_px": length})
    frozen = [
        (26.0, 238.0, 90.0, "legacy_R01"), (254.75, 225.0, 90.0, "legacy_R02"),
        (115.0, 331.0, 0.0, "legacy_R03"), (174.79, 316.0, 2.0, "legacy_R04"),
        (88.25, 192.0, 62.0, "legacy_R05"), (189.91, 203.44, 125.0, "legacy_R06"),
    ]
    for x, y, tangent, label in frozen:
        raw.append({"candidate_source": label, "source_line": -1, "center_x": x, "center_y": y,
                    "tangent_angle_deg": tangent, "normal_angle_deg": (tangent + 90.0) % 180.0,
                    "line_length_ref_px": 24.0})
    candidates: list[dict[str, Any]] = []
    for row in sorted(raw, key=lambda r: r["line_length_ref_px"], reverse=True):
        if any(math.hypot(row["center_x"] - q["center_x"], row["center_y"] - q["center_y"]) < 7.0 and axial_distance(row["normal_angle_deg"], q["normal_angle_deg"]) < 12.0 for q in candidates):
            continue
        row = dict(row)
        row["coherence"] = local_orientation_coherence(structural, row["center_x"], row["center_y"])
        row["direction_15deg"] = direction_from_normal(row["normal_angle_deg"])
        candidates.append(row)
    for i, row in enumerate(candidates, 1):
        row["candidate_ID"] = f"C{i:03d}"
    return candidates


def screen_candidates(
    candidates: list[dict[str, Any]],
    uv_exr: np.ndarray,
    high_exr: np.ndarray,
    distance: np.ndarray,
    high_transform: Similarity,
    baseline_reg: dict[str, Any],
    common_shape: tuple[int, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    assessed: list[dict[str, Any]] = []
    for row in candidates:
        r = dict(row); reasons: list[str] = []
        x, y = r["center_x"], r["center_y"]
        margin = min(x, y, common_shape[1] - 1 - x, common_shape[0] - 1 - y)
        r["common_border_margin_px"] = float(margin)
        if margin < MIN_BORDER_REF:
            reasons.append("insufficient common-grid background margin")
        if r["coherence"] < MIN_COHERENCE:
            reasons.append("low local orientation coherence")
        uv_res, _ = roi_profile(uv_exr, "uv", r, distance, high_transform, baseline_reg)
        hi_res, _ = roi_profile(high_exr, "high", r, distance, high_transform, baseline_reg)
        r.update({
            "UV_profile_valid": bool(uv_res["valid"]), "NIR_profile_valid": bool(hi_res["valid"]),
            "UV_CNR": uv_res.get("CNR"), "NIR_CNR": hi_res.get("CNR"),
            "UV_peak_to_noise": uv_res.get("peak_to_noise"), "NIR_peak_to_noise": hi_res.get("peak_to_noise"),
        })
        finite_cnr = [v for v in [r["UV_CNR"], r["NIR_CNR"]] if v is not None and np.isfinite(v)]
        r["minimum_pair_CNR"] = float(min(finite_cnr)) if len(finite_cnr) == 2 else np.nan
        if not uv_res["valid"]:
            reasons.append("UV profile invalid")
        if not hi_res["valid"]:
            reasons.append("NIR profile invalid")
        if len(finite_cnr) < 2 or min(finite_cnr) < MIN_CNR:
            reasons.append("pair CNR below frozen threshold")
        r["quality_score_no_gain"] = float(
            1.2 * min(max(r.get("coherence", 0), 0), 1)
            + 0.8 * math.log1p(max(r.get("minimum_pair_CNR", 0) if np.isfinite(r.get("minimum_pair_CNR", np.nan)) else 0, 0))
            + 0.3 * min(r["line_length_ref_px"] / 35.0, 2.0)
        )
        r["preselection_pass"] = len(reasons) == 0
        r["rejection_reason"] = "; ".join(reasons)
        assessed.append(r)
    selected: list[dict[str, Any]] = []
    for direction in ["X", "Y", "Diagonal"]:
        pool = sorted([r for r in assessed if r["preselection_pass"] and r["direction_15deg"] == direction], key=lambda r: r["quality_score_no_gain"], reverse=True)
        for row in pool:
            if len([q for q in selected if q["direction_15deg"] == direction]) >= MAX_PER_DIRECTION:
                row["rejection_reason"] = "direction quota reached"
                continue
            if any(math.hypot(row["center_x"] - q["center_x"], row["center_y"] - q["center_y"]) < MIN_ROI_SEPARATION_REF for q in selected):
                row["rejection_reason"] = "spatially redundant with higher-quality selected ROI"
                continue
            selected.append(row)
    selected_ids = {r["candidate_ID"] for r in selected}
    for i, row in enumerate(sorted(selected, key=lambda r: (r["direction_15deg"], r["center_y"], r["center_x"])), 1):
        row["ROI_ID"] = f"R{i:02d}"
        row["selected"] = True
    rejected = []
    for row in assessed:
        if row["candidate_ID"] not in selected_ids:
            row["selected"] = False
            if not row["rejection_reason"]:
                row["rejection_reason"] = "not retained after blinded quality ranking"
            rejected.append(row)
    return assessed, selected, rejected


def gaussian_model(x: np.ndarray, b: float, a: float, mu: float, sigma: float) -> np.ndarray:
    return b + a * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def lorentzian_model(x: np.ndarray, b: float, a: float, mu: float, gamma: float) -> np.ndarray:
    return b + a / (1.0 + ((x - mu) / gamma) ** 2)


def generalized_gaussian_model(x: np.ndarray, b: float, a: float, mu: float, alpha: float, beta: float) -> np.ndarray:
    return b + a * np.exp(-(np.abs(x - mu) / alpha) ** beta)


def voigt_model(x: np.ndarray, b: float, a: float, mu: float, sigma: float, gamma: float) -> np.ndarray:
    v = special.voigt_profile(x - mu, sigma, gamma)
    return b + a * v / max(float(special.voigt_profile(0.0, sigma, gamma)), EPS)


def rect_psf_model(x: np.ndarray, b: float, a: float, mu: float, width: float, sigma: float) -> np.ndarray:
    xl, xr = mu - width / 2.0, mu + width / 2.0
    z = math.sqrt(2.0) * sigma
    return b + 0.5 * a * (special.erf((x - xl) / z) - special.erf((x - xr) / z))


def fit_profile_models(distance: np.ndarray, raw: np.ndarray) -> list[dict[str, Any]]:
    result = analyze_mean_profile(distance, raw)
    if not result["valid"]:
        return []
    y = result["normalized"]
    good = np.isfinite(y) & (np.abs(distance) <= PROFILE_HALF_LENGTH_REF * 0.95)
    x, yy = distance[good], y[good]
    mu0 = float(distance[np.nanargmax(y)]); width0 = max(float(result["FWHM"]), 1.0)
    specs: list[tuple[str, Callable, list[float], tuple[list[float], list[float]], Callable[[np.ndarray], float]]] = [
        ("Gaussian", gaussian_model, [0, 1, mu0, width0 / 2.355], ([-0.5, 0.2, -8, 0.15], [0.5, 2, 8, 30]), lambda p: 2.354820045 * p[3]),
        ("Lorentzian", lorentzian_model, [0, 1, mu0, width0 / 2], ([-0.5, 0.2, -8, 0.15], [0.5, 2, 8, 30]), lambda p: 2 * p[3]),
        ("GeneralizedGaussian", generalized_gaussian_model, [0, 1, mu0, width0 / 2, 2], ([-0.5, 0.2, -8, 0.15, 0.5], [0.5, 2, 8, 30, 8]), lambda p: 2 * p[3] * (math.log(2)) ** (1 / p[4])),
        ("Voigt", voigt_model, [0, 1, mu0, width0 / 4, width0 / 4], ([-0.5, 0.2, -8, 0.10, 0.10], [0.5, 2, 8, 20, 20]), lambda p: 0.5346 * 2 * p[4] + math.sqrt(0.2166 * (2 * p[4]) ** 2 + (2.354820045 * p[3]) ** 2)),
        ("FiniteLineGaussianPSF", rect_psf_model, [0, 1, mu0, width0 * 0.6, max(width0 * 0.15, 0.3)], ([-0.5, 0.2, -8, 0.05, 0.10], [0.5, 2, 8, 35, 15]), lambda p: numerical_fwhm(lambda z: rect_psf_model(z, *p), -30, 30)),
    ]
    rows = []
    for name, func, p0, bounds, width_fn in specs:
        try:
            popt, _ = optimize.curve_fit(func, x, yy, p0=p0, bounds=bounds, maxfev=30000)
            pred = func(x, *popt)
            resid = yy - pred
            row = {"estimator": name, "fit_success": True, "fitted_FWHM": float(width_fn(popt)),
                   "RMSE": float(np.sqrt(np.mean(resid ** 2))), "MAE": float(np.mean(np.abs(resid))),
                   "parameters": json.dumps([float(v) for v in popt])}
            if name == "FiniteLineGaussianPSF":
                row["apparent_intrinsic_like_width"] = float(popt[3])
                row["apparent_blur_sigma"] = float(popt[4])
            rows.append(row)
        except Exception as exc:
            rows.append({"estimator": name, "fit_success": False, "fitted_FWHM": np.nan, "RMSE": np.nan, "MAE": np.nan, "parameters": "", "fit_error": str(exc)})
    return rows


def numerical_fwhm(func: Callable[[np.ndarray], np.ndarray], lo: float, hi: float) -> float:
    x = np.linspace(lo, hi, 6001); y = np.asarray(func(x), float)
    b = float(min(y[0], y[-1])); target = b + 0.5 * (float(np.max(y)) - b)
    above = np.flatnonzero(y >= target)
    if len(above) < 2:
        return float("nan")
    return float(x[above[-1]] - x[above[0]])


def summarize_values(values: np.ndarray) -> dict[str, Any]:
    v = np.asarray(values, float); v = v[np.isfinite(v)]
    if not len(v):
        return {"n": 0, "mean": np.nan, "SD": np.nan, "median": np.nan, "IQR": np.nan, "min": np.nan, "max": np.nan}
    q1, q3 = np.percentile(v, [25, 75])
    return {"n": len(v), "mean": float(v.mean()), "SD": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "median": float(np.median(v)), "IQR": float(q3 - q1), "min": float(v.min()), "max": float(v.max())}


def directional_summary(roi_df: pd.DataFrame, orientation_tolerance: float = PRIMARY_ORIENTATION_TOL, value_col: str = "FWHM") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    work = roi_df.copy()
    work["Direction_current"] = work["normal_angle_deg"].map(lambda a: direction_from_normal(float(a), orientation_tolerance))
    for direction in ["X", "Y", "Diagonal", "All"]:
        sub = work if direction == "All" else work[work["Direction_current"] == direction]
        pivot = sub.pivot_table(index="ROI_ID", columns="condition", values=value_col, aggfunc="first")
        if not {"UV", "NIR"}.issubset(pivot.columns):
            uv = np.array([]); nir = np.array([])
        else:
            complete = pivot[["UV", "NIR"]].dropna()
            uv, nir = complete["UV"].to_numpy(float), complete["NIR"].to_numpy(float)
        us, ns = summarize_values(uv), summarize_values(nir)
        gain = us["mean"] / ns["mean"] if np.isfinite(us["mean"]) and np.isfinite(ns["mean"]) and ns["mean"] > 0 else np.nan
        rows.append({
            "Direction": direction, "orientation_tolerance_deg": orientation_tolerance, "N_ROI": int(min(us["n"], ns["n"])),
            "UV_mean_FWHM": us["mean"], "UV_SD": us["SD"], "UV_median": us["median"], "UV_IQR": us["IQR"], "UV_min": us["min"], "UV_max": us["max"],
            "NIR_mean_FWHM": ns["mean"], "NIR_SD": ns["SD"], "NIR_median": ns["median"], "NIR_IQR": ns["IQR"], "NIR_min": ns["min"], "NIR_max": ns["max"],
            "Gain": gain, "Reduction_percent": (1 - 1 / gain) * 100 if np.isfinite(gain) and gain > 0 else np.nan,
            "replicate_level": "within-image spatial ROI; not independent experimental replicate",
            "unit": "common-reference-grid apparent pixels",
        })
    return rows


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


def gains_from_rows(rows: list[dict[str, Any]], tolerance: float = PRIMARY_ORIENTATION_TOL) -> tuple[float, float, int, int]:
    if not rows:
        return np.nan, np.nan, 0, 0
    summary = directional_summary(pd.DataFrame(rows), tolerance)
    by = {r["Direction"]: r for r in summary}
    return by["X"]["Gain"], by["Y"]["Gain"], by["X"]["N_ROI"], by["Y"]["N_ROI"]


def plot_registration(
    out: Path,
    high_rgb: np.ndarray,
    old_nir_rgb: np.ndarray,
    high_common_rgb: np.ndarray,
    transform: Similarity,
    identity_scores: dict[str, Any],
) -> None:
    h, w = old_nir_rgb.shape[:2]
    corners_x = np.array([0, w - 1, w - 1, 0, 0], float); corners_y = np.array([0, 0, h - 1, h - 1, 0], float)
    hx, hy = transform.map(corners_x, corners_y)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    axes[0, 0].imshow(high_rgb.astype(np.uint8)); axes[0, 0].plot(hx, hy, "c-", lw=1.5)
    axes[0, 0].set_title("(a) 4096×3000 JPEG; fitted corresponding field"); axes[0, 0].axis("off")
    axes[0, 1].imshow(old_nir_rgb.astype(np.uint8)); axes[0, 1].set_title("(b) Legacy 976-nm TIFF reference"); axes[0, 1].axis("off")
    axes[1, 0].imshow(high_common_rgb.astype(np.uint8)); axes[1, 0].set_title("(c) High-resolution field sampled on reference grid (display only)"); axes[1, 0].axis("off")
    overlay = np.zeros((h, w, 3), np.float32)
    overlay[..., 0] = cv2.cvtColor(old_nir_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
    overlay[..., 1] = cv2.cvtColor(high_common_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY) / 255.0
    axes[1, 1].imshow(np.clip(overlay, 0, 1)); axes[1, 1].set_title("(d) Red=legacy NIR; green=high-resolution NIR"); axes[1, 1].axis("off")
    fig.suptitle(f"Structure-frozen similarity: s={transform.scale:.4f}, θ={transform.rotation_deg:.3f}°; identity score NIR={identity_scores['legacy_NIR']['combined_score']:.3f}, UV={identity_scores['legacy_UV']['combined_score']:.3f}")
    fig.savefig(out, dpi=220, facecolor="white"); plt.close(fig)


def plot_directional_rois(out: Path, reference_rgb: np.ndarray, all_candidates: list[dict[str, Any]], selected: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 7.5), layout="constrained")
    # The legacy NIR reference is used only as a legible common-grid locator.
    # Quantitative profiles remain sampled from native UV/high-resolution data.
    ax.imshow(reference_rgb.astype(np.uint8))
    for r in all_candidates:
        ax.plot(r["center_x"], r["center_y"], ".", color="0.55", ms=0.7, alpha=0.14)
    colors = {"X": "#0072B2", "Y": "#D55E00", "Diagonal": "#009E73"}
    for r in selected:
        a = math.radians(r["normal_angle_deg"]); dx, dy = 11 * math.cos(a), 11 * math.sin(a)
        d = direction_from_normal(r["normal_angle_deg"])
        ax.plot([r["center_x"] - dx, r["center_x"] + dx], [r["center_y"] - dy, r["center_y"] + dy], color=colors[d], lw=1.8)
        ax.text(r["center_x"] + 3, r["center_y"] - 3, r["ROI_ID"], fontsize=6, color="black", bbox={"fc": "white", "ec": colors[d], "alpha": 0.8, "pad": 0.8})
    for d, c in colors.items():
        ax.plot([], [], color=c, lw=2, label=f"{d} normal")
    ax.legend(frameon=True, fontsize=7, loc="lower right")
    ax.set_title(
        "Frozen directional ROIs\nEnhanced image used only for candidate localization; widths use raw images",
        fontsize=10,
    )
    ax.axis("off"); fig.savefig(out, dpi=300, facecolor="white"); plt.close(fig)


def plot_profile_examples(
    out: Path,
    selected: list[dict[str, Any]],
    cache: dict[tuple[str, str], dict[str, Any]],
    distance: np.ndarray,
    channel_label: str,
    title_note: str,
) -> None:
    examples = []
    example_limits = {"X": 2, "Y": 1, "Diagonal": 1}
    for direction in ["X", "Y", "Diagonal"]:
        valid = [r for r in selected if direction_from_normal(r["normal_angle_deg"]) == direction and cache[("UV", r["ROI_ID"])]["valid"] and cache[("NIR", r["ROI_ID"])]["valid"]]
        examples.extend(valid[:example_limits[direction]])
    if not examples:
        fig, ax = plt.subplots(figsize=(8, 3.2), layout="constrained")
        ax.text(0.5, 0.56, f"No complete UV/NIR profile pairs for {channel_label}", ha="center", va="center", fontsize=13)
        ax.text(0.5, 0.40, title_note, ha="center", va="center", fontsize=9, color="0.35", wrap=True)
        ax.set_axis_off()
        fig.savefig(out, dpi=300, facecolor="white"); plt.close(fig)
        return
    fig, axes = plt.subplots(math.ceil(len(examples) / 2), 2, figsize=(8, 2.7 * math.ceil(len(examples) / 2)), layout="constrained", squeeze=False)
    for ax, roi in zip(axes.ravel(), examples):
        for condition, color_value, ls in [("UV", "#D55E00", "--"), ("NIR", "#0072B2", "-")]:
            r = cache[(condition, roi["ROI_ID"])]
            label = f"{condition}: {r['FWHM']:.2f} px" if r["valid"] else f"{condition}: invalid"
            ax.plot(distance, r["normalized"], color=color_value, ls=ls, lw=1.6, label=label)
            if r["valid"]:
                ax.scatter([r["x50_left"], r["x50_right"]], [0.5, 0.5], s=12, color=color_value, zorder=3)
        ax.axhline(0.5, color="0.5", lw=0.7); ax.set_ylim(-0.25, 1.25)
        ax.set_title(f"{roi['ROI_ID']} | {direction_from_normal(roi['normal_angle_deg'])}")
        ax.set_xlabel("Distance (reference-grid px)"); ax.set_ylabel(f"Normalized {channel_label}"); ax.legend(fontsize=7)
    for ax in axes.ravel()[len(examples):]: ax.axis("off")
    fig.suptitle(title_note, fontsize=10)
    fig.savefig(out, dpi=300, facecolor="white"); plt.close(fig)


def transform_about_center(base: Similarity, common_shape: tuple[int, int], scale_factor: float = 1.0, rotation_delta: float = 0.0, shift_x: float = 0.0, shift_y: float = 0.0) -> Similarity:
    cx, cy = (common_shape[1] - 1) / 2.0, (common_shape[0] - 1) / 2.0
    hx, hy = base.map(np.array([cx]), np.array([cy]))
    new = Similarity(base.scale * scale_factor, base.rotation_deg + rotation_delta, 0.0, 0.0)
    nx, ny = new.map(np.array([cx]), np.array([cy]))
    return Similarity(new.scale, new.rotation_deg, float(hx[0] - nx[0] + shift_x), float(hy[0] - ny[0] + shift_y))


def append_robustness(rows: list[dict[str, Any]], label: str, analysis_rows: list[dict[str, Any]], tolerance: float, category: str, detail: str = "") -> None:
    gx, gy, nx, ny = gains_from_rows(analysis_rows, tolerance)
    rows.append({"Method": label, "Category": category, "Detail": detail, "orientation_tolerance_deg": tolerance,
                 "Gx": gx, "Gy": gy, "N_x": nx, "N_y": ny,
                 "Gx_ge_2p3": bool(np.isfinite(gx) and gx >= 2.3), "Gy_ge_2p3": bool(np.isfinite(gy) and gy >= 2.3)})


def make_pca_chromatic_maps(uv_rgb: np.ndarray, high_rgb: np.ndarray, baseline_registered_rgb: np.ndarray, high_common_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    def chroma(x: np.ndarray) -> np.ndarray:
        denom = np.maximum(x.sum(axis=-1, keepdims=True), 1.0)
        return x / denom
    pooled = np.concatenate([chroma(baseline_registered_rgb).reshape(-1, 3), chroma(high_common_rgb).reshape(-1, 3)], axis=0)
    if len(pooled) > 120000:
        pooled = pooled[RNG.choice(len(pooled), 120000, replace=False)]
    center = pooled.mean(0); _, vec, vals = cv2.PCACompute2(pooled.astype(np.float32), mean=center[None, :].astype(np.float32))
    component = vec[0].astype(float)
    exr_vector = np.array([2.0, -1.0, -1.0])
    if np.dot(component, exr_vector) < 0:
        component *= -1
    uv_map = np.tensordot(chroma(uv_rgb) - center, component, axes=([-1], [0])).astype(np.float32)
    high_map = np.tensordot(chroma(high_rgb) - center, component, axes=([-1], [0])).astype(np.float32)
    return uv_map, high_map, {"center": center.tolist(), "first_component": component.tolist(), "explained_variance": float(vals[0, 0] / max(vals.sum(), EPS)), "sign_aligned_to_ExR": True}


def mean_by_direction(summary_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {r["Direction"]: r for r in summary_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="High-resolution directional apparent-FWHM analysis")
    parser.add_argument("--workspace", type=Path, default=Path(r"C:\Users\Admin\Desktop\xiaoyu"))
    parser.add_argument("--uv", type=Path, default=None)
    parser.add_argument("--legacy-nir", type=Path, default=None)
    parser.add_argument("--high-nir", type=Path, default=Path(r"C:\Users\Admin\AppData\Local\Temp\codex-clipboard-d6737a5c-e05c-45ea-a6b3-cab206ddff91.jpg"))
    parser.add_argument(
        "--candidate-nir",
        type=Path,
        default=None,
        help="Optional whole-image deterministic enhancement used only for ROI candidate discovery; quantitative profiles remain sampled from --high-nir.",
    )
    parser.add_argument(
        "--candidate-csv",
        type=Path,
        default=None,
        help="Optional precomputed candidate geometry CSV. It may be selected only by image-structure/quality criteria, never by FWHM or gain.",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=None,
        help="Existing baseline results directory. Defaults to OUTPUT/baseline_workspace/results.",
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    workspace, out = args.workspace.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    uv_path = (args.uv or workspace / "248T.tif").resolve()
    legacy_nir_path = (args.legacy_nir or workspace / "976-T.tif").resolve()
    high_path = args.high_nir.resolve()
    candidate_path = (args.candidate_nir or high_path).resolve()
    baseline_root = args.baseline_root.resolve() if args.baseline_root is not None else out / "baseline_workspace" / "results"
    baseline_json_path = baseline_root / "analysis_results.json"
    baseline_roi_path = baseline_root / "tables" / "roi_level_metrics.csv"
    baseline_registered_path = baseline_root / "diagnostics" / "before_registered_common_grid.tif"
    required_paths = [uv_path, legacy_nir_path, high_path, candidate_path, baseline_json_path, baseline_roi_path, baseline_registered_path]
    if not all(p.exists() for p in required_paths):
        missing = [str(p) for p in required_paths if not p.exists()]
        raise FileNotFoundError(f"Missing required inputs: {missing}")

    log_lines = [f"started_utc={datetime.now(timezone.utc).isoformat()}", f"seed={SEED}"]
    baseline = json.loads(baseline_json_path.read_text(encoding="utf-8"))
    baseline_reg = baseline["registration"]
    uv_rgb, legacy_nir_rgb, high_rgb = load_rgb(uv_path), load_rgb(legacy_nir_path), load_rgb(high_path)
    candidate_rgb = load_rgb(candidate_path)
    if candidate_rgb.shape != high_rgb.shape:
        raise ValueError(
            f"Candidate-discovery image must match raw high-NIR dimensions exactly: {candidate_rgb.shape} != {high_rgb.shape}"
        )
    baseline_registered_rgb = load_rgb(baseline_registered_path)
    common_shape = legacy_nir_rgb.shape[:2]
    metadata = {
        "UV": file_metadata(uv_path),
        "legacy_NIR": file_metadata(legacy_nir_path),
        "high_resolution_candidate": file_metadata(high_path),
        "candidate_discovery_image": file_metadata(candidate_path),
        "candidate_discovery_scope": "ROI candidate localization only; registration validation, quality gates, and all quantitative profiles use the raw high-resolution NIR input",
    }
    metadata["high_resolution_candidate"]["jpeg_blockiness"] = jpeg_blockiness(cv2.cvtColor(high_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY))
    write_json(out / "input_metadata.json", metadata)

    coarse_nir, coarse_nir_records = coarse_template_registration(high_rgb, legacy_nir_rgb, "legacy_NIR")
    coarse_uv, coarse_uv_records = coarse_template_registration(high_rgb, uv_rgb, "legacy_UV")
    identity_scores = {"legacy_NIR": coarse_nir, "legacy_UV": coarse_uv}
    initial = similarity_from_coarse(coarse_nir, legacy_nir_rgb.shape[:2])
    transform, refinement = refine_similarity(high_rgb, legacy_nir_rgb, initial)
    high_common_rgb = warp_high_to_common(high_rgb, transform, common_shape)
    candidate_common_rgb = warp_high_to_common(candidate_rgb, transform, common_shape)
    feature_checks = feature_validation(legacy_nir_rgb, high_common_rgb)
    inverse_validation = validate_uv_inverse(uv_rgb, baseline_registered_rgb, baseline_reg)
    classification = "NIR-supported" if coarse_nir["combined_score"] > coarse_uv["combined_score"] else "identity-ambiguous"
    registration_record = {
        "primary_transform_type": "similarity (translation + rotation + isotropic scale)",
        "mapping": "legacy 976-nm common-reference coordinates -> 4096x3000 JPEG coordinates",
        "initial_from_multiscale_NCC": initial.__dict__, "final_similarity": transform.__dict__,
        "matrix_2x3": transform.matrix().tolist(), "scale_high_px_per_reference_px": transform.scale,
        "rotation_deg": transform.rotation_deg, "translation_x_high_px": transform.tx, "translation_y_high_px": transform.ty,
        "coarse_identity_scores": identity_scores, "coarse_NIR_top40": coarse_nir_records,
        "coarse_UV_top40": coarse_uv_records, "refinement": refinement,
        "landmark_and_phase_validation": feature_checks, "UV_native_inverse_mapping_validation": inverse_validation,
        "classification": classification,
        "scale_selection_statement": "Scale was frozen from structural NCC/ECC-like continuous optimization and feature/phase validation before any FWHM calculation.",
        "physical_calibration_available": False, "reported_unit": "common-reference-grid apparent pixels",
    }
    write_json(out / "registration_parameters.json", registration_record)
    plot_registration(out / "registration_overlay.png", high_rgb, legacy_nir_rgb, high_common_rgb, transform, identity_scores)

    uv_scale_native_per_ref = 1.0 / math.sqrt(float(baseline_reg["resize_scale_x"]) * float(baseline_reg["resize_scale_y"]))
    uv_maps = rgb_channel_maps(uv_rgb, 8.0 * uv_scale_native_per_ref)
    high_maps = rgb_channel_maps(high_rgb, 8.0 * transform.scale)
    old_maps = rgb_channel_maps(legacy_nir_rgb, 8.0)
    pca_uv, pca_high, pca_record = make_pca_chromatic_maps(uv_rgb, high_rgb, baseline_registered_rgb, high_common_rgb)
    uv_maps["PCA_chromatic"], high_maps["PCA_chromatic"] = pca_uv, pca_high
    pca_center = np.asarray(pca_record["center"], float); pca_component = np.asarray(pca_record["first_component"], float)
    old_chroma = legacy_nir_rgb / np.maximum(legacy_nir_rgb.sum(axis=-1, keepdims=True), 1.0)
    old_maps["PCA_chromatic"] = np.tensordot(old_chroma - pca_center, pca_component, axes=([-1], [0])).astype(np.float32)
    polarity_record: dict[str, Any] = {}
    for channel_name in ["ExR", "Lab_a", "local_darkness", "red_fraction", "optical_density", "PCA_chromatic"]:
        high_common_channel = warp_high_to_common(high_maps[channel_name], transform, common_shape)
        corr = ncc(old_maps[channel_name], high_common_channel)
        sign = -1.0 if np.isfinite(corr) and corr < 0 else 1.0
        high_maps[channel_name] *= sign
        polarity_record[channel_name] = {"legacy_NIR_to_highres_common_correlation_before_sign": corr, "highres_multiplier": sign,
                                         "selection_rule": "sign of same-condition spatial correlation; independent of FWHM/gain"}
    write_json(out / "channel_polarity.json", polarity_record)
    write_json(out / "pca_chromatic_parameters.json", pca_record)
    distance = np.arange(-PROFILE_HALF_LENGTH_REF, PROFILE_HALF_LENGTH_REF + PROFILE_STEP_REF / 2, PROFILE_STEP_REF)

    if args.candidate_csv is not None:
        candidate_csv_path = args.candidate_csv.resolve()
        if not candidate_csv_path.exists():
            raise FileNotFoundError(f"Candidate CSV does not exist: {candidate_csv_path}")
        candidates = pd.read_csv(candidate_csv_path).to_dict(orient="records")
        required_candidate_fields = {
            "candidate_ID", "candidate_source", "source_line", "center_x", "center_y",
            "tangent_angle_deg", "normal_angle_deg", "line_length_ref_px", "coherence",
            "direction_15deg",
        }
        missing_fields = required_candidate_fields.difference(candidates[0].keys() if candidates else set())
        if missing_fields:
            raise ValueError(f"Candidate CSV is missing required fields: {sorted(missing_fields)}")
        metadata["candidate_geometry_csv"] = {
            "path": str(candidate_csv_path),
            "sha256": sha256(candidate_csv_path),
            "selection_boundary": "candidate geometry prefiltered without FWHM or UV/NIR gain; all candidates are re-screened on raw-image quality",
        }
        write_json(out / "input_metadata.json", metadata)
    else:
        candidates = build_roi_candidates(legacy_nir_rgb, candidate_common_rgb)
    selection_evaluation: list[dict[str, Any]] = []
    assessed_by_channel: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for selection_channel in ["ExR", "Lab_a", "local_darkness", "red_fraction", "optical_density"]:
        assessed_tuple = screen_candidates(candidates, uv_maps[selection_channel], high_maps[selection_channel], distance, transform, baseline_reg, common_shape)
        assessed_by_channel[selection_channel] = assessed_tuple
        assessed_channel = assessed_tuple[0]
        complete_valid = [r for r in assessed_channel if r["UV_profile_valid"] and r["NIR_profile_valid"]]
        passed = [r for r in assessed_channel if r["preselection_pass"]]
        median_pair_cnr = float(np.nanmedian([r["minimum_pair_CNR"] for r in complete_valid])) if complete_valid else np.nan
        selection_evaluation.append({"channel": selection_channel, "N_candidates": len(assessed_channel), "N_complete_valid": len(complete_valid),
                                     "N_pass_frozen_quality_gate": len(passed), "median_min_pair_CNR": median_pair_cnr,
                                     "selection_score_no_gain": len(passed) * 10000 + len(complete_valid) * 10 + (median_pair_cnr if np.isfinite(median_pair_cnr) else 0),
                                     "rule": "maximize quality-gate passes, then complete profiles and median pair CNR; FWHM/gain excluded"})
    selection_evaluation.sort(key=lambda r: r["selection_score_no_gain"], reverse=True)
    selected_quality_channel = selection_evaluation[0]["channel"]
    all_candidates, selected, rejected = assessed_by_channel[selected_quality_channel]
    for row in all_candidates:
        row["ROI_selection_channel"] = selected_quality_channel
    write_csv(out / "roi_selection_channel_evaluation.csv", selection_evaluation)
    for direction in ["X", "Y"]:
        if len([r for r in selected if direction_from_normal(r["normal_angle_deg"]) == direction]) < 2:
            pool = sorted([r for r in all_candidates if r["preselection_pass"] and direction_from_normal(r["normal_angle_deg"]) == direction and r not in selected], key=lambda r: r["quality_score_no_gain"], reverse=True)
            while len([r for r in selected if direction_from_normal(r["normal_angle_deg"]) == direction]) < 2 and pool:
                selected.append(pool.pop(0))
    selected_ids = {r["candidate_ID"] for r in selected}
    for i, row in enumerate(sorted(selected, key=lambda r: (direction_from_normal(r["normal_angle_deg"]), r["center_y"], r["center_x"])), 1):
        row["ROI_ID"] = f"R{i:02d}"; row["selected"] = True; row["rejection_reason"] = ""; row["ROI_selection_channel"] = selected_quality_channel
    rejected = [r for r in all_candidates if r["candidate_ID"] not in selected_ids]
    for r in rejected:
        r["selected"] = False
        if not r.get("rejection_reason"):
            r["rejection_reason"] = "not retained after blinded quality ranking"
    write_csv(out / "all_candidate_rois.csv", all_candidates)
    write_csv(out / "selected_rois.csv", selected)
    write_csv(out / "rejected_rois.csv", rejected)
    plot_directional_rois(out / "directional_rois.png", candidate_common_rgb, all_candidates, selected)

    primary_rows, primary_profiles, primary_cache = measure_rois(selected, uv_maps["ExR"], high_maps["ExR"], "high", distance, transform, baseline_reg)
    primary_df = pd.DataFrame(primary_rows)
    primary_df.to_csv(out / "roi_level_fwhm.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(primary_profiles).to_csv(out / "all_roi_profiles.csv", index=False, encoding="utf-8-sig")
    primary_summary = directional_summary(primary_df)
    write_csv(out / "directional_fwhm_summary.csv", primary_summary)
    plot_profile_examples(
        out / "profile_examples_ExR.png", selected, primary_cache, distance,
        "ExR", "Prespecified primary channel; invalid pairs remain explicitly missing",
    )

    if selected_quality_channel == "ExR":
        quality_primary_rows = primary_rows
    else:
        quality_primary_rows, _, _ = measure_rois(
            selected, uv_maps[selected_quality_channel], high_maps[selected_quality_channel],
            "high", distance, transform, baseline_reg, channel=selected_quality_channel,
        )

    robustness: list[dict[str, Any]] = []
    append_robustness(robustness, "Primary", primary_rows, PRIMARY_ORIENTATION_TOL, "primary", "ExR; half=50%; sigma=0.75 ref px; similarity")
    append_robustness(robustness, f"Secondary {selected_quality_channel}", quality_primary_rows, PRIMARY_ORIENTATION_TOL,
                      "secondary_quality_channel", "quality channel frozen without FWHM/gain; half=50%; sigma=0.75 ref px")
    for sigma_ref in [0.0, 0.25, 0.5, 0.75, 1.0]:
        rr, _, _ = measure_rois(selected, uv_maps["ExR"], high_maps["ExR"], "high", distance, transform, baseline_reg, sigma_ref=sigma_ref)
        append_robustness(robustness, f"Gaussian sigma={sigma_ref:g}", rr, PRIMARY_ORIENTATION_TOL, "smoothing")
        rr_q, _, _ = measure_rois(selected, uv_maps[selected_quality_channel], high_maps[selected_quality_channel], "high", distance, transform, baseline_reg, sigma_ref=sigma_ref, channel=selected_quality_channel)
        append_robustness(robustness, f"{selected_quality_channel}: Gaussian sigma={sigma_ref:g}", rr_q, PRIMARY_ORIENTATION_TOL, "secondary_smoothing")
    for half in [0.45, 0.50, 0.55]:
        rr, _, _ = measure_rois(selected, uv_maps["ExR"], high_maps["ExR"], "high", distance, transform, baseline_reg, half_level=half)
        append_robustness(robustness, f"Threshold={int(half*100)}%", rr, PRIMARY_ORIENTATION_TOL, "half_max_threshold")
        rr_q, _, _ = measure_rois(selected, uv_maps[selected_quality_channel], high_maps[selected_quality_channel], "high", distance, transform, baseline_reg, half_level=half, channel=selected_quality_channel)
        append_robustness(robustness, f"{selected_quality_channel}: Threshold={int(half*100)}%", rr_q, PRIMARY_ORIENTATION_TOL, "secondary_half_max_threshold")
    for tol in [10.0, 15.0, 20.0]:
        append_robustness(robustness, f"Orientation tolerance ±{int(tol)}°", primary_rows, tol, "orientation")
        append_robustness(robustness, f"{selected_quality_channel}: orientation tolerance ±{int(tol)}°", quality_primary_rows, tol, "secondary_orientation")

    channel_rows_all: list[dict[str, Any]] = []
    channel_rows_cache: dict[str, list[dict[str, Any]]] = {}
    channel_profile_cache: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    all_channel_roi_rows: list[dict[str, Any]] = []
    all_channel_profile_rows: list[dict[str, Any]] = []
    all_channel_summary_rows: list[dict[str, Any]] = []
    for channel in ["ExR", "Lab_a", "local_darkness", "red_fraction", "optical_density", "PCA_chromatic"]:
        rr, pp, cc = measure_rois(selected, uv_maps[channel], high_maps[channel], "high", distance, transform, baseline_reg, channel=channel)
        channel_rows_cache[channel] = rr
        channel_profile_cache[channel] = cc
        all_channel_roi_rows.extend(rr)
        all_channel_profile_rows.extend(pp)
        for sr in directional_summary(pd.DataFrame(rr)):
            all_channel_summary_rows.append({"channel": channel, **sr})
        gx, gy, nx, ny = gains_from_rows(rr)
        valid = [r for r in rr if r["valid"]]
        channel_rows_all.append({"channel": channel, "valid_fraction": len(valid) / max(len(rr), 1),
                                 "median_CNR": float(np.nanmedian([r["CNR"] for r in valid])) if valid else np.nan,
                                 "median_peak_to_noise": float(np.nanmedian([r["peak_to_noise"] for r in valid])) if valid else np.nan,
                                 "Gx_sensitivity": gx, "Gy_sensitivity": gy, "N_x": nx, "N_y": ny,
                                 "selection_basis": "validity/CNR/stability; gain not used; ExR retained as prespecified primary"})
        append_robustness(robustness, f"Channel={channel}", rr, PRIMARY_ORIENTATION_TOL, "channel")
    write_csv(out / "channel_sensitivity.csv", channel_rows_all)
    write_csv(out / "roi_level_fwhm_all_channels.csv", all_channel_roi_rows)
    write_csv(out / "all_roi_profiles_all_channels.csv", all_channel_profile_rows)
    write_csv(out / "directional_fwhm_summary_all_channels.csv", all_channel_summary_rows)
    plot_profile_examples(
        out / "profile_examples.png", selected, channel_profile_cache[selected_quality_channel], distance,
        selected_quality_channel,
        f"Representative complete pairs on the frozen ROI-quality channel ({selected_quality_channel}); not the prespecified primary endpoint",
    )

    for method in ["median", "bilateral", "tv", "wavelet"]:
        rr, _, _ = measure_rois(selected, uv_maps["ExR"], high_maps["ExR"], "high", distance, transform, baseline_reg, filter_method=method)
        append_robustness(robustness, f"Profile denoiser={method}", rr, PRIMARY_ORIENTATION_TOL, "denoising", "applied symmetrically after native-grid sampling and 11-profile averaging")
        rr_q, _, _ = measure_rois(selected, uv_maps[selected_quality_channel], high_maps[selected_quality_channel], "high", distance, transform, baseline_reg, filter_method=method, channel=selected_quality_channel)
        append_robustness(robustness, f"{selected_quality_channel}: profile denoiser={method}", rr_q, PRIMARY_ORIENTATION_TOL, "secondary_denoising", "applied symmetrically after native-grid sampling and 11-profile averaging")
    robustness.append({"Method": "Profile denoiser=NLM", "Category": "denoising", "Detail": "not used: 1-D ROI-mean profiles do not provide a defensible patch-neighborhood model", "orientation_tolerance_deg": PRIMARY_ORIENTATION_TOL, "Gx": np.nan, "Gy": np.nan, "N_x": 0, "N_y": 0, "Gx_ge_2p3": False, "Gy_ge_2p3": False})
    robustness.append({"Method": "Profile denoiser=BM3D", "Category": "denoising", "Detail": "not available in the isolated environment; excluded from primary by design", "orientation_tolerance_deg": PRIMARY_ORIENTATION_TOL, "Gx": np.nan, "Gy": np.nan, "N_x": 0, "N_y": 0, "Gx_ge_2p3": False, "Gy_ge_2p3": False})

    transform_variants = {
        "similarity_primary": transform,
        "scale_minus_1pct": transform_about_center(transform, common_shape, scale_factor=0.99),
        "scale_plus_1pct": transform_about_center(transform, common_shape, scale_factor=1.01),
        "rotation_minus_0p25deg": transform_about_center(transform, common_shape, rotation_delta=-0.25),
        "rotation_plus_0p25deg": transform_about_center(transform, common_shape, rotation_delta=0.25),
        "small_affine_x_plus1pct_surrogate": transform_about_center(transform, common_shape, scale_factor=1.005),
        "small_affine_y_plus1pct_surrogate": transform_about_center(transform, common_shape, scale_factor=0.995),
    }
    for name, tv in transform_variants.items():
        rr, _, _ = measure_rois(selected, uv_maps["ExR"], high_maps["ExR"], "high", distance, tv, baseline_reg)
        append_robustness(robustness, f"Registration={name}", rr, PRIMARY_ORIENTATION_TOL, "registration")
        rr_q, _, _ = measure_rois(selected, uv_maps[selected_quality_channel], high_maps[selected_quality_channel], "high", distance, tv, baseline_reg, channel=selected_quality_channel)
        append_robustness(robustness, f"{selected_quality_channel}: registration={name}", rr_q, PRIMARY_ORIENTATION_TOL, "secondary_registration")
    for roi_id in [r["ROI_ID"] for r in selected]:
        rr = [r for r in primary_rows if r["ROI_ID"] != roi_id]
        append_robustness(robustness, f"Leave {roi_id} out", rr, PRIMARY_ORIENTATION_TOL, "leave_one_ROI_out")
        rr_q = [r for r in quality_primary_rows if r["ROI_ID"] != roi_id]
        append_robustness(robustness, f"{selected_quality_channel}: leave {roi_id} out", rr_q, PRIMARY_ORIENTATION_TOL, "secondary_leave_one_ROI_out")
    write_csv(out / "robustness_analysis.csv", robustness)

    fit_details: list[dict[str, Any]] = []
    for fit_channel in list(dict.fromkeys(["ExR", selected_quality_channel])):
        for (condition, roi_id), cached in channel_profile_cache[fit_channel].items():
            roi = next(r for r in selected if r["ROI_ID"] == roi_id)
            for row in fit_profile_models(distance, cached["raw"]):
                fit_details.append({"channel": fit_channel, "condition": condition, "ROI_ID": roi_id,
                                    "Direction": direction_from_normal(roi["normal_angle_deg"]), **row,
                                    "interpretation_boundary": "apparent profile model; no independently calibrated PSF"})
    write_csv(out / "model_fit_details.csv", fit_details)

    method_rows: list[dict[str, Any]] = []
    for estimator in ["PrimaryNonparametric", "Gaussian", "Lorentzian", "GeneralizedGaussian", "Voigt", "FiniteLineGaussianPSF"]:
        if estimator == "PrimaryNonparametric":
            gx, gy, nx, ny = gains_from_rows(primary_rows); rmse = np.nan
        else:
            rows = []
            for fit in [f for f in fit_details if f["channel"] == "ExR" and f["estimator"] == estimator and f["fit_success"]]:
                roi = next(r for r in selected if r["ROI_ID"] == fit["ROI_ID"])
                rows.append({"condition": fit["condition"], "ROI_ID": fit["ROI_ID"], "normal_angle_deg": roi["normal_angle_deg"], "FWHM": fit["fitted_FWHM"]})
            gx, gy, nx, ny = gains_from_rows(rows) if rows else (np.nan, np.nan, 0, 0)
            rmse_values = [f["RMSE"] for f in fit_details if f["channel"] == "ExR" and f["estimator"] == estimator and f["fit_success"]]
            rmse = float(np.nanmedian(rmse_values)) if rmse_values else np.nan
        method_rows.append({"strategy": "B_native_grids_common_reference", "channel": "ExR", "endpoint_status": "prespecified_primary",
                            "estimator": estimator, "Gx": gx, "Gy": gy, "N_x": nx, "N_y": ny, "median_fit_RMSE": rmse})

    # The quality-channel analysis is explicitly secondary. It is added so the
    # result cannot be mistaken for a successful ExR endpoint.
    for estimator in ["PrimaryNonparametric", "Gaussian", "Lorentzian", "GeneralizedGaussian", "Voigt", "FiniteLineGaussianPSF"]:
        if estimator == "PrimaryNonparametric":
            gx_q, gy_q, nx_q, ny_q = gains_from_rows(channel_rows_cache[selected_quality_channel]); rmse_q = np.nan
        else:
            rows_q = []
            for fit in [f for f in fit_details if f["channel"] == selected_quality_channel and f["estimator"] == estimator and f["fit_success"]]:
                roi = next(r for r in selected if r["ROI_ID"] == fit["ROI_ID"])
                rows_q.append({"condition": fit["condition"], "ROI_ID": fit["ROI_ID"], "normal_angle_deg": roi["normal_angle_deg"], "FWHM": fit["fitted_FWHM"]})
            gx_q, gy_q, nx_q, ny_q = gains_from_rows(rows_q) if rows_q else (np.nan, np.nan, 0, 0)
            rmse_values_q = [f["RMSE"] for f in fit_details if f["channel"] == selected_quality_channel and f["estimator"] == estimator and f["fit_success"]]
            rmse_q = float(np.nanmedian(rmse_values_q)) if rmse_values_q else np.nan
        method_rows.append({"strategy": "B_native_grids_common_reference", "channel": selected_quality_channel,
                            "endpoint_status": "secondary_quality_channel", "estimator": estimator,
                            "Gx": gx_q, "Gy": gy_q, "N_x": nx_q, "N_y": ny_q, "median_fit_RMSE": rmse_q})

    baseline_roi = pd.read_csv(baseline_roi_path)
    baseline_roi["condition"] = baseline_roi["condition"].map({"before": "UV", "after": "NIR"})
    normal_lookup = {"R01": 180.0, "R02": 180.0, "R03": 90.0, "R04": 92.0, "R05": 152.0, "R06": 35.0}
    baseline_roi["normal_angle_deg"] = baseline_roi["ROI_ID"].map(normal_lookup)
    baseline_rows = baseline_roi[["condition", "ROI_ID", "normal_angle_deg", "FWHM"]].to_dict("records")
    gx_a, gy_a, nx_a, ny_a = gains_from_rows(baseline_rows)
    method_rows.append({"strategy": "A_legacy_common_lowres", "channel": "ExR", "endpoint_status": "historical_baseline",
                        "estimator": "PrimaryNonparametric", "Gx": gx_a, "Gy": gy_a, "N_x": nx_a, "N_y": ny_a, "median_fit_RMSE": np.nan})
    step_c = max(0.05, 1.0 / transform.scale)
    distance_c = np.arange(-PROFILE_HALF_LENGTH_REF, PROFILE_HALF_LENGTH_REF + step_c / 2, step_c)
    rows_c, _, _ = measure_rois(selected, uv_maps["ExR"], high_maps["ExR"], "high", distance_c, transform, baseline_reg)
    gx_c, gy_c, nx_c, ny_c = gains_from_rows(rows_c)
    method_rows.append({"strategy": "C_high_resolution_common_sampling", "channel": "ExR", "endpoint_status": "sampling_sensitivity",
                        "estimator": "PrimaryNonparametric", "Gx": gx_c, "Gy": gy_c, "N_x": nx_c, "N_y": ny_c, "median_fit_RMSE": np.nan,
                        "note": "UV upsampling used only for subpixel evaluation; it adds no information"})
    write_csv(out / "method_comparison.csv", method_rows)

    sampling_rows: list[dict[str, Any]] = []
    for sampling_channel in list(dict.fromkeys(["ExR", selected_quality_channel])):
        old_rows, _, _ = measure_rois(selected, uv_maps[sampling_channel], old_maps[sampling_channel], "old_nir", distance, transform, baseline_reg, channel=sampling_channel)
        high_common_map = warp_high_to_common(high_maps[sampling_channel], transform, common_shape)
        common_interp_rows, _, _ = measure_rois(selected, uv_maps[sampling_channel], high_common_map, "old_nir", distance, transform, baseline_reg, channel=sampling_channel)
        prefiltered_high = ndimage.gaussian_filter(high_maps[sampling_channel], max(0.5, 0.42 * transform.scale), mode="nearest")
        high_antialias_common = warp_high_to_common(prefiltered_high, transform, common_shape)
        downsample_rows, _, _ = measure_rois(selected, uv_maps[sampling_channel], high_antialias_common, "old_nir", distance, transform, baseline_reg, channel=sampling_channel)
        variants = {
            "legacy_976_TIFF": old_rows,
            "highres_native": channel_rows_cache[sampling_channel],
            "highres_single_interpolation_to_common": common_interp_rows,
            "highres_simulated_antialias_downsample": downsample_rows,
        }
        for name, rows in variants.items():
            summ = mean_by_direction(directional_summary(pd.DataFrame(rows)))
            for d in ["X", "Y", "All"]:
                sampling_rows.append({"channel": sampling_channel, "variant": name, "Direction": d, "N_ROI": summ[d]["N_ROI"],
                                      "UV_mean": summ[d]["UV_mean_FWHM"], "NIR_mean": summ[d]["NIR_mean_FWHM"], "Gain": summ[d]["Gain"]})
        for quality in [95, 85]:
            buf = io.BytesIO(); Image.fromarray(high_rgb.astype(np.uint8)).save(buf, format="JPEG", quality=quality, subsampling=0)
            recompressed = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("RGB"), dtype=np.float32)
            if sampling_channel == "Lab_a":
                # OpenCV avoids a rare boolean-index assignment failure in
                # skimage.color.rgb2lab on some large JPEG round-trips.  The
                # affine Lab-a scale difference is removed by ROI-level
                # normalization and therefore does not change W50 crossings.
                recompressed_map = cv2.cvtColor(
                    np.ascontiguousarray(recompressed.astype(np.uint8)), cv2.COLOR_RGB2LAB
                )[..., 1].astype(np.float32)
            else:
                recompressed_map = rgb_channel_maps(recompressed, 8.0 * transform.scale)[sampling_channel]
            recompressed_map *= polarity_record[sampling_channel]["highres_multiplier"]
            rr, _, _ = measure_rois(selected, uv_maps[sampling_channel], recompressed_map, "high", distance, transform, baseline_reg, channel=sampling_channel)
            summ = mean_by_direction(directional_summary(pd.DataFrame(rr)))
            for d in ["X", "Y", "All"]:
                sampling_rows.append({"channel": sampling_channel, "variant": f"additional_JPEG_reencode_Q{quality}", "Direction": d,
                                      "N_ROI": summ[d]["N_ROI"], "UV_mean": summ[d]["UV_mean_FWHM"],
                                      "NIR_mean": summ[d]["NIR_mean_FWHM"], "Gain": summ[d]["Gain"]})
    write_csv(out / "sampling_bias_analysis.csv", sampling_rows)

    summary_by = mean_by_direction(primary_summary)
    gx, gy = summary_by["X"]["Gain"], summary_by["Y"]["Gain"]
    quality_summary = directional_summary(pd.DataFrame(channel_rows_cache[selected_quality_channel]))
    quality_by = mean_by_direction(quality_summary)
    q_gx, q_gy = quality_by["X"]["Gain"], quality_by["Y"]["Gain"]

    # A direction needs at least three spatial ROIs before it is called a
    # credible directional estimate. This is descriptive, not an inference to
    # a population of specimens.
    credible_quality = [(d, quality_by[d]["Gain"], quality_by[d]["N_ROI"]) for d in ["X", "Y"]
                        if quality_by[d]["N_ROI"] >= 3 and np.isfinite(quality_by[d]["Gain"])]
    if credible_quality:
        best_direction, max_credible_gain, best_n = max(credible_quality, key=lambda z: z[1])
        gap = max(0.0, 2.3 - max_credible_gain)
    else:
        best_direction, max_credible_gain, best_n, gap = "none", np.nan, 0, np.nan

    robust_df = pd.DataFrame(robustness)
    parameter_rows = robust_df[robust_df["Category"].str.contains("smoothing|half_max_threshold|orientation|registration|channel|leave_one_ROI_out", regex=True)]
    gx_vals = parameter_rows.loc[(parameter_rows["N_x"] >= 3) & parameter_rows["Gx"].notna(), "Gx"].to_numpy(float)
    gy_vals = parameter_rows.loc[(parameter_rows["N_y"] >= 3) & parameter_rows["Gy"].notna(), "Gy"].to_numpy(float)
    robust_support_x = bool(len(gx_vals) >= 3 and np.percentile(gx_vals, 10) >= 2.3)
    robust_support_y = bool(len(gy_vals) >= 3 and np.percentile(gy_vals, 10) >= 2.3)
    support_any = robust_support_x or robust_support_y
    conclusion = "当前数据在至少一个方向上稳健支持 ≥2.3×。" if support_any else "当前数据不能可靠支持 2.3×。"
    baseline_fwhm = next(r for r in baseline["summary_statistics"] if r["metric"] == "FWHM")
    sampling_df = pd.DataFrame(sampling_rows)
    sampling_q = sampling_df[sampling_df.channel == selected_quality_channel]
    native_all = sampling_q[(sampling_q.variant == "highres_native") & (sampling_q.Direction == "All")].iloc[0]
    old_all = sampling_q[(sampling_q.variant == "legacy_976_TIFF") & (sampling_q.Direction == "All")].iloc[0]
    down_all = sampling_q[(sampling_q.variant == "highres_simulated_antialias_downsample") & (sampling_q.Direction == "All")].iloc[0]
    bias_pct = (float(old_all.NIR_mean) / float(native_all.NIR_mean) - 1) * 100 if np.isfinite(float(old_all.NIR_mean)) and np.isfinite(float(native_all.NIR_mean)) and float(native_all.NIR_mean) > 0 else np.nan
    downsample_widen_pct = (float(down_all.NIR_mean) / float(native_all.NIR_mean) - 1) * 100 if np.isfinite(float(down_all.NIR_mean)) and np.isfinite(float(native_all.NIR_mean)) and float(native_all.NIR_mean) > 0 else np.nan
    jpeg_q = sampling_q[(sampling_q.variant.str.startswith("additional_JPEG_reencode")) & (sampling_q.Direction == "All")]
    jpeg_gain_min = float(jpeg_q.Gain.min()) if len(jpeg_q) else np.nan
    jpeg_gain_max = float(jpeg_q.Gain.max()) if len(jpeg_q) else np.nan
    secondary_model_rows = [r for r in method_rows if r.get("channel") == selected_quality_channel and r.get("strategy") == "B_native_grids_common_reference" and r.get("estimator") != "PrimaryNonparametric" and int(r.get("N_x", 0)) >= 3 and np.isfinite(r.get("Gx", np.nan))]
    model_gain_text = "; ".join(f"{r['estimator']}={float(r['Gx']):.3f}× (n={int(r['N_x'])})" for r in secondary_model_rows)
    model_dependent_max = max((float(r["Gx"]) for r in secondary_model_rows), default=np.nan)
    x_selected = [r for r in selected if direction_from_normal(r["normal_angle_deg"]) == "X"]
    x_left = len([r for r in x_selected if r["center_x"] < common_shape[1] / 2])
    x_right = len(x_selected) - x_left
    y_selected = [r for r in selected if direction_from_normal(r["normal_angle_deg"]) == "Y"]
    y_left = len([r for r in y_selected if r["center_x"] < common_shape[1] / 2])
    y_right = len(y_selected) - y_left

    def rf(value: Any, digits: int = 3, suffix: str = "") -> str:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "不可估计"
        return f"{v:.{digits}f}{suffix}" if np.isfinite(v) else "不可估计"

    primary_status = (
        f"ExR 主分析的完整 UV/NIR 成对 ROI 数为 X={summary_by['X']['N_ROI']}、Y={summary_by['Y']['N_ROI']}；"
        f"对应 Gx={rf(gx)}、Gy={rf(gy)}。"
    )
    credible_statement = (f"最大可信描述性增益为 {rf(max_credible_gain)}×（{best_direction}，n={best_n}），距 2.3× 为 {rf(gap)}×（约 {rf(gap/2.3*100,1)}% 的目标值）。"
                          if np.isfinite(max_credible_gain) else "没有方向达到至少 3 个完整空间 ROI，无法给出可信方向性增益。")
    sampling_statement = (f"在备选 {selected_quality_channel} 通道和冻结 ROI 上，旧 TIFF 相对高分辨率 native 图的 NIR apparent FWHM 高 {rf(bias_pct,1)}%；模拟抗混叠降采样使高分辨率结果改变 {rf(downsample_widen_pct,1)}%。额外 JPEG Q95/Q85 重编码使总体增益落在 {rf(jpeg_gain_min)}–{rf(jpeg_gain_max)}×，说明颜色剖面对有损编码敏感。"
                          if np.isfinite(bias_pct) else f"在备选 {selected_quality_channel} 通道上，旧 TIFF 与高分辨率 native 图没有足够完整的同 ROI 对，sampling bias 不能稳定量化。")
    directional_sampling_statement = (
        f"X、Y 均达到至少3个完整ROI（X={quality_by['X']['N_ROI']}，Y={quality_by['Y']['N_ROI']}），"
        "可以作为当前单幅图像内的描述性方向估计；但空间ROI不是独立实验重复，不能外推为材料总体。"
        if quality_by["X"]["N_ROI"] >= 3 and quality_by["Y"]["N_ROI"] >= 3
        else "至少一个方向少于3个完整ROI，因此该方向只能作为局部观察。"
    )
    coverage_statement = (
        f"X向左右覆盖为{x_left}/{x_right}，Y向左右覆盖为{y_left}/{y_right}；候选不再集中于单侧。"
        if min(x_left, x_right, y_left, y_right) > 0
        else "至少一个方向仍缺少单侧覆盖，不能外推为整幅样品方向平均。"
    )

    report = f"""# 高分辨率 976 nm 方向性 apparent-FWHM 分析报告

## 结论

**{conclusion}** {primary_status} 冻结的 ROI 质量通道为 **{selected_quality_channel}**（按完整剖面数/CNR 选择，完全不使用 FWHM 或增益）；它是预先规定 ExR 失败后的敏感性结果，不可冒充主终点。该通道得到 X 向 {rf(q_gx)}×（n={quality_by['X']['N_ROI']}），Y 向 {rf(q_gy)}×（n={quality_by['Y']['N_ROI']}）。{credible_statement}

## 关键证据链

1. 旧基线被隔离复现：UV {baseline_fwhm['before_mean']:.4f} ± {baseline_fwhm['before_SD']:.4f} px，旧 NIR {baseline_fwhm['after_mean']:.4f} ± {baseline_fwhm['after_SD']:.4f} px，总体增益 {baseline_fwhm['before_mean']/baseline_fwhm['after_mean']:.4f}×。
2. 新附件为 4096×3000、8-bit JPEG，而非 TIFF/RAW；元数据含 Photoshop/Windows Photo Editor 派生历史。结构盲匹配分数：旧 NIR={coarse_nir['combined_score']:.3f}，旧 UV={coarse_uv['combined_score']:.3f}，判定为 **{classification}**。高分辨率图与旧 NIR 的 ExR 空间相关在符号校正前为 {polarity_record['ExR']['legacy_NIR_to_highres_common_correlation_before_sign']:.3f}；只做符号翻转（不改变宽度）后仍无完整 ExR 成对 ROI，提示颜色/导出链不适合把 ExR 当作可测终点。
3. 几何尺度在任何 FWHM 计算前冻结：{transform.scale:.5f} high-res px/reference px，旋转 {transform.rotation_deg:.4f}°，平移 ({transform.tx:.2f}, {transform.ty:.2f}) high-res px。SIFT/RANSAC 中位残差 {feature_checks.get('SIFT', {}).get('median_residual_common_px', np.nan):.3f} reference px。没有 μm/pixel 标定，故只报告 **common-reference-grid apparent FWHM**。
4. UV 在 native TIFF 网格采样，NIR 在 4096×3000 JPEG native 网格采样；11 条邻近剖面先平均，再施加对两条件相同的 σ={PRIMARY_SIGMA_REF} reference px 弱 1-D Gaussian 稳定。未逐剖面对齐峰值。
5. ROI 通过结构、局部方向相干性、两条件 CNR、背景裕量及剖面有效性筛选。共冻结 {len(selected)} 个 ROI；X={len(x_selected)}（左/右={x_left}/{x_right}），Y={len(y_selected)}（左/右={y_left}/{y_right}），Diagonal={len([r for r in selected if direction_from_normal(r['normal_angle_deg']) == 'Diagonal'])}。{coverage_statement}

## 方向性结果

### 预设主通道 ExR

|方向|完整 ROI 数|UV apparent FWHM|NIR apparent FWHM|Gain|
|---|---:|---:|---:|---:|
|X|{summary_by['X']['N_ROI']}|{rf(summary_by['X']['UV_mean_FWHM'])}|{rf(summary_by['X']['NIR_mean_FWHM'])}|{rf(gx)}|
|Y|{summary_by['Y']['N_ROI']}|{rf(summary_by['Y']['UV_mean_FWHM'])}|{rf(summary_by['Y']['NIR_mean_FWHM'])}|{rf(gy)}|

### 备选质量通道 {selected_quality_channel}

|方向|n|UV mean ± SD|NIR mean ± SD|Gain|Reduction|
|---|---:|---:|---:|---:|---:|
|X|{quality_by['X']['N_ROI']}|{rf(quality_by['X']['UV_mean_FWHM'])} ± {rf(quality_by['X']['UV_SD'])}|{rf(quality_by['X']['NIR_mean_FWHM'])} ± {rf(quality_by['X']['NIR_SD'])}|{rf(q_gx)}×|{rf(quality_by['X']['Reduction_percent'],1)}%|
|Y|{quality_by['Y']['N_ROI']}|{rf(quality_by['Y']['UV_mean_FWHM'])} ± {rf(quality_by['Y']['UV_SD'])}|{rf(quality_by['Y']['NIR_mean_FWHM'])} ± {rf(quality_by['Y']['NIR_SD'])}|{rf(q_gy)}×|{rf(quality_by['Y']['Reduction_percent'],1)}%|

{directional_sampling_statement} {coverage_statement} ROI 是同一图像对内的空间子样本，不是独立实验重复，故不计算材料总体 P 值。

参数模型在备选 {selected_quality_channel} 的 X 向给出：{model_gain_text}。这些模型相关结果数值上可超过 2.3，最大为 {rf(model_dependent_max)}×，但它们建立在备选通道、较少的完整拟合对和未独立标定的 PSF 上；非参数 crossing 为 X={rf(q_gx)}×、Y={rf(q_gy)}×，参数扰动第 10 百分位为 X={rf(np.percentile(gx_vals,10) if len(gx_vals) else np.nan)}×、Y={rf(np.percentile(gy_vals,10) if len(gy_vals) else np.nan)}×。因此把参数模型列为 **model-dependent sensitivity**，不能覆盖非参数与稳健性结果。

## 对十个问题的明确回答

1. **新高分辨率图更适合 FWHM 吗？** 就采样而言是：约 {transform.scale:.2f} high-res px/reference px，亚像素 crossing 的离散化更细；但它是有编辑历史的 JPEG，证据等级仍低于未经处理的 TIFF/RAW。
2. **旧 12.27→7.70 px 有明显 sampling bias 吗？** {sampling_statement} 这只是文件链/采样敏感性，不能证明全部差异均由低分辨率像素化造成，因为两文件未被证明来自同一未经处理的采集链。
3. **新的科学合理 FWHM？** ExR 主分析不可估计。备选 {selected_quality_channel} 的 X、Y apparent FWHM 见上表；逐 ROI 结果见 `roi_level_fwhm_all_channels.csv`。
4. **Gx？** 主分析不可估计；备选 {selected_quality_channel} 为 {rf(q_gx)}×（n={quality_by['X']['N_ROI']}）。
5. **Gy？** 主分析不可估计；备选 {selected_quality_channel} 为 {rf(q_gy)}×（n={quality_by['Y']['N_ROI']}，证据不足）。
6. **至少一方向 ≥2.3×？** 备选通道的描述性点估计为 X={rf(q_gx)}×、Y={rf(q_gy)}×；稳健支持判定为 X={robust_support_x}、Y={robust_support_y}。备选通道的单个高值不能替代 ExR 主通道或完整扰动结果。
7. **若超过 2.3，合理参数范围内是否仍成立？** 不成立。至少 3 ROI 的敏感性行中，X 第 10 百分位={rf(np.percentile(gx_vals,10) if len(gx_vals) else np.nan, suffix='×')}，Y={rf(np.percentile(gy_vals,10) if len(gy_vals) else np.nan, suffix='×')}；均未形成稳健 ≥2.3 证据。
8. **最大可信增益？** 保守、模型无关的描述值为 {rf(max_credible_gain)}×（{best_direction}，n={best_n}，备选通道）；模型相关上限为 {rf(model_dependent_max)}×，不作为确认性结论。
9. **距 2.3×？** {rf(gap)}×。
10. **怎样科学提高精度？** 获取无 JPEG/Photoshop 派生的 976-nm TIFF/RAW；同一相机/物镜/像元/曝光下成对采集 UV/NIR；保存 μm/pixel；用刀口或亚分辨珠独立标定 LSF/PSF；增加左右两侧、横纵方向均衡的 ROI 和独立图像/样品重复；冻结分析方案后用层级模型汇总。AI 超分辨率、单条件锐化、按结果调尺度均不得用于定量证据。

## 模型、稳健性与完整性边界

`method_comparison.csv` 在 ExR 与备选 {selected_quality_channel} 上比较非参数 crossing、Gaussian、Lorentzian、Voigt、generalized Gaussian 与有限线宽⊗Gaussian。最后一种只作 apparent decomposition；没有独立 PSF，不可解释为真正 intrinsic linewidth。`robustness_analysis.csv` 覆盖 σ=0–1、45/50/55%、±10/15/20°、通道、配准扰动、对称去噪和 leave-one-ROI-out；任何 n<3 的方向数值都不用于“可信 ≥2.3”判断。NLM/BM3D 未进入主分析，理由在表中明确记录。

所有定量值来自实验 TIFF/JPEG 的确定性非生成式处理。没有 AI 超分辨率、生成式修复、锐化、deblur、非线性 warp、按结果删 ROI，也没有把空间 ROI 当作独立实验重复。ROI 图以旧 NIR 作为同一参考网格的位置底图；定量仍从 UV 与高分辨率 NIR 的 native 网格采样。

## 参考方法

- Guizar-Sicairos M, Thurman ST, Fienup JR. Efficient subpixel image registration algorithms. *Optics Letters* 33, 156–158 (2008). https://doi.org/10.1364/OL.33.000156
- Lowe DG. Distinctive image features from scale-invariant keypoints. *International Journal of Computer Vision* 60, 91–110 (2004). https://doi.org/10.1023/B:VISI.0000029664.99615.94
- Ahmad R, Ding Y, Simonetti OP. Edge sharpness assessment by parametric modeling. *Concepts in Magnetic Resonance Part A* 44, 138–149 (2015). https://doi.org/10.1002/cmr.a.21339
- Hsu WF, Hsu YC, Chuang KW. Measurement of the spatial frequency response of digital still-picture cameras using a modified slanted-edge method. *Proc. SPIE* 4080 (2000). https://doi.org/10.1117/12.389433
- Descloux A, Grußmayer KS, Radenovic A. Parameter-free image resolution estimation based on decorrelation analysis. *Nature Methods* 16, 918–924 (2019). https://doi.org/10.1038/s41592-019-0515-7
- Rousseeuw PJ, Croux C. Alternatives to the Median Absolute Deviation. *JASA* 88, 1273–1283 (1993). https://doi.org/10.1080/01621459.1993.10476408
- Vaux DL, Fidler F, Cumming G. Replicates and repeats—what is the difference and is it significant? *EMBO Reports* 13, 291–296 (2012). https://doi.org/10.1038/embor.2012.36
"""
    (out / "analysis_report.md").write_text(report, encoding="utf-8")

    package_versions = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__, "opencv": cv2.__version__, "matplotlib": matplotlib.__version__}
    log_lines.extend([
        f"completed_utc={datetime.now(timezone.utc).isoformat()}", f"classification={classification}",
        f"scale_high_per_ref={transform.scale:.8f}", f"rotation_deg={transform.rotation_deg:.8f}",
        f"selected_ROIs={len(selected)}",
        f"Gx={gx if np.isfinite(gx) else 'not_estimable'}",
        f"Gy={gy if np.isfinite(gy) else 'not_estimable'}",
        f"secondary_{selected_quality_channel}_Gx={q_gx if np.isfinite(q_gx) else 'not_estimable'}",
        f"secondary_{selected_quality_channel}_Gy={q_gy if np.isfinite(q_gy) else 'not_estimable'}",
        f"robust_support_x={robust_support_x}", f"robust_support_y={robust_support_y}",
        f"package_versions={json.dumps(package_versions)}", "BM3D=unavailable_not_primary", "NLM=not_defensible_for_1D_ROI_mean_not_primary",
    ])
    (out / "analysis_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    deliverables = []
    for p in sorted(x for x in out.iterdir() if x.is_file() and x.name != "deliverables_manifest.csv"):
        deliverables.append({"file": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)})
    write_csv(out / "deliverables_manifest.csv", deliverables)
    print(json.dumps(json_ready({"Gx": gx, "Gy": gy, "support_any_robust": support_any, "selected_ROIs": len(selected), "scale": transform.scale, "classification": classification}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        out = Path(__file__).resolve().parent
        with (out / "analysis_log.txt").open("a", encoding="utf-8") as f:
            f.write("FATAL_ERROR\n" + traceback.format_exc() + "\n")
        raise
