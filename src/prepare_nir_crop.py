"""Create the high-resolution NIR field of view corresponding to the UV reference grid.

The crop is lossless (native pixels only).  `registration.json` stores the
frozen similarity mapping required to sample the cropped NIR image on the UV
reference grid; no sharpening, deconvolution, or resampling is applied here.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measurement_core import Similarity

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "NIR_highres_unsharpened_original.jpg"
OUT = ROOT / "data" / "NIR_highres_cropped_to_UV_FOV.tif"
REG = ROOT / "config" / "registration.json"
COMMON_W, COMMON_H = 285, 355
TRANSFORM = Similarity(8.4269951607774, 0.08596964176050484, 1087.3136345390114, 6.230904383546789)

def main() -> None:
    image = np.asarray(Image.open(RAW).convert("RGB"), dtype=np.uint8)
    x = np.array([0, COMMON_W - 1, 0, COMMON_W - 1], float)
    y = np.array([0, 0, COMMON_H - 1, COMMON_H - 1], float)
    hx, hy = TRANSFORM.map(x, y)
    pad = 3
    x0, x1 = max(0, int(np.floor(hx.min())) - pad), min(image.shape[1], int(np.ceil(hx.max())) + pad + 1)
    y0, y1 = max(0, int(np.floor(hy.min())) - pad), min(image.shape[0], int(np.ceil(hy.max())) + pad + 1)
    Image.fromarray(image[y0:y1, x0:x1]).save(OUT, compression="tiff_lzw")
    reg = {
        "common_grid_width_px": COMMON_W, "common_grid_height_px": COMMON_H,
        "uv_mapping": {"resize_scale_x": 0.9076433121019108, "resize_scale_y": 0.9102564102564102, "angle_deg": 1.5, "shift_y_px": 1.3, "shift_x_px": -3.25},
        "nir_crop_origin_native_px": [x0, y0],
        "nir_similarity_common_to_cropped_native": {"scale": TRANSFORM.scale, "rotation_deg": TRANSFORM.rotation_deg, "tx": TRANSFORM.tx - x0, "ty": TRANSFORM.ty - y0},
        "note": "NIR crop is a native-pixel field of view corresponding to the UV common grid; it is not resized to 285 x 355 pixels."
    }
    REG.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    print(json.dumps({"crop_shape_yx": [y1-y0, x1-x0], "crop_origin_xy": [x0,y0]}, indent=2))

if __name__ == "__main__": main()
