# UV–NIR apparent-W50 analysis

This repository reproduces the frozen five-transect comparison of 248-nm UV and 976-nm NIR writing images. It reports an image-level, X-direction apparent-W50 comparison in common-reference-grid pixels.

## Inputs and roles

`data/UV_248T.tif` and `data/NIR_highres_cropped_to_UV_FOV.tif` are the two quantitative inputs. The latter is a native-pixel crop of the supplied unsharpened high-resolution NIR image; it covers the UV reference field of view but is not resized. `data/NIR_976-T_display.tif` is included only for registered-image localization and figure display.

## Run

```bash
python src/prepare_nir_crop.py
python src/run_analysis.py
python src/make_single_figures.py
```

All figures are exported as standalone PNG and editable SVG files. No raw image is sharpened, deconvolved, generatively edited, or overwritten. Five technical transects are nested within four parent spatial ROIs; the reported values describe one registered image pair and are not independent-sample statistics. `Gx` is the ratio of the UV and NIR mean apparent W50 values, not a direct measurement of storage capacity, pitch, or bit-error rate.
