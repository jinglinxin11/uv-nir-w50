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

def main() -> None:
    image = np.asarray(Image.open(RAW).convert("RGB"), dtype=np.uint8)
    reg = json.loads(REG.read_text(encoding="utf-8"))
    common_w, common_h = int(reg["common_grid_width_px"]), int(reg["common_grid_height_px"])
    source = reg["nir_similarity_common_to_original_native"]
    transform = Similarity(source["scale"], source["rotation_deg"], source["tx"], source["ty"])
    x = np.array([0, common_w - 1, 0, common_w - 1], float)
    y = np.array([0, 0, common_h - 1, common_h - 1], float)
    hx, hy = transform.map(x, y)
    pad = 3
    x0, x1 = max(0, int(np.floor(hx.min())) - pad), min(image.shape[1], int(np.ceil(hx.max())) + pad + 1)
    y0, y1 = max(0, int(np.floor(hy.min())) - pad), min(image.shape[0], int(np.ceil(hy.max())) + pad + 1)
    Image.fromarray(image[y0:y1, x0:x1]).save(OUT, compression="tiff_lzw")
    reg["nir_crop_origin_native_px"] = [x0, y0]
    reg["nir_similarity_common_to_cropped_native"] = {"scale": transform.scale, "rotation_deg": transform.rotation_deg, "tx": transform.tx - x0, "ty": transform.ty - y0}
    reg["note"] = "Native source pixels are preserved during cropping; no additional resampling, sharpening, deconvolution, or super-resolution is applied."
    REG.write_text(json.dumps(reg, indent=2), encoding="utf-8")
    manifest = {"crop_shape_yx": [y1-y0, x1-x0], "crop_origin_xy": [x0,y0], "output": str(OUT.relative_to(ROOT)), "native_source_pixels_preserved": True}
    (ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "tables" / "crop_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__": main()
