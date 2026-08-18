# UV–NIR apparent-W50 reproducibility package

This repository reproduces the reported image-based comparison of apparent transverse writing width for 248 nm UV and 976 nm NIR conditions. It is a computational companion to the associated manuscript, not a replacement for the manuscript or its Supplementary Information.

The released analysis uses five prespecified X-direction technical transects nested in four parent spatial ROIs. For each condition and transect, 11 parallel profiles are sampled normal to the local written ridge, averaged, and measured at the half-maximum crossings. `Gx` is the ratio of condition means, `mean(W50_UV) / mean(W50_NIR)`; it is an apparent-width comparison metric, not a direct measurement of absolute storage capacity, pitch, crosstalk, or bit-error rate.

## Released inputs and provenance

| Item | Role |
| --- | --- |
| `data/UV_248T.tif` | Quantitative UV input |
| `data/NIR_highres_unsharpened_original.jpg` | Original, unsharpened NIR quantitative source |
| `data/NIR_highres_cropped_to_UV_FOV.tif` | Deterministic native-pixel NIR field-of-view crop derived from the original source |
| `data/NIR_976-T_display.tif` | Display/localization image only; not used for quantitative NIR profiles |
| `config/registration.json` | Authoritative stored UV and NIR coordinate transforms |
| `config/transects.csv` | Input-only frozen geometry for X01–X05; it has no W50, PNR, or gain outcome columns |
| `provenance/ROI_selection_records.csv` | Full frozen ROI-selection record retained for audit, including historical screening/outcome fields |
| `provenance/input_checksums.sha256` | SHA-256 hashes verified by `verify_release.py` |

`src/prepare_nir_crop.py` reads the NIR transform from `config/registration.json` and regenerates the cropped TIFF without an additional interpolation step. The crop manifest is written to `results/tables/crop_manifest.json`.

## One-command reproduction

Tested environment: Python 3.12.13 on Windows. Create a virtual environment, then install the pinned dependencies:

```bash
python -m pip install -r requirements-lock.txt
python reproduce_all.py
```

The workflow runs, in order: deterministic crop generation; input-only transect construction; the five-transect analysis; smoothing and leave-one-parent-ROI-out robustness analyses; all standalone PNG figure scripts; and release verification. It regenerates tables under `results/tables/` and figures under `results/figures/`.

## Expected primary outputs

| Validation target | Expected value |
| --- | ---: |
| Valid condition × transect measurements | 10 / 10 |
| UV mean apparent W50 | 14.5261048734 reference px |
| NIR mean apparent W50 | 6.0509975853 reference px |
| `Gx` (ratio of means) | 2.4006132325 |
| Apparent-W50 reduction | 58.343977% |

The primary profile channel is CIE Lab a*. `verify_release.py` checks the input hashes, frozen geometry (five transects/four parent ROIs), valid measurement count, numerical targets, and the robustness-output schema.

## Output map

| Output(s) | Generating script | Purpose |
| --- | --- | --- |
| `Fig_S9_profile_X01.png`–`Fig_S9_profile_X05.png` | `src/make_single_figures.py` | Individual paired ROI profiles |
| `Supporting_ROI_locations.png` | `src/make_single_figures.py` | ROI location support figure; not assigned an SI number |
| `Figure_S10_representative_W50_X04.png` | `src/make_single_figures.py` | Representative W50 extraction |
| `Figure_S11_PNR_validity.png` | `src/make_supplementary_and_maintext_figures.py` | PNR validity checks |
| `Figure_S12_smoothing_UV.png`, `Figure_S12_smoothing_NIR.png` | `src/make_supplementary_and_maintext_figures.py` | Primary smoothing sensitivity, split into condition panels |
| `Fig_Main_paired_W50.png`, `Fig_Main_Gx_proxy.png` | `src/make_single_figures.py` | Independent main-text candidate figures (not S11/S12) |
| `Fig_Main_*` RGB/Lab-a/profile PNGs | `src/make_supplementary_and_maintext_figures.py` | Standalone display and profile figures |
| `Figure4f_ROI_selection_transverse_width_workflow.png` | `src/make_figure4f_roi_workflow.py` | ROI-to-W50 method schematic; panel (c) is illustrative |

All released figure outputs are PNG files. The manuscript and word-processing documents are deliberately not bundled in this code/data release.

## Repository layout

```text
config/       transforms and input-only transect geometry
data/         released image inputs
provenance/   frozen selection record and checksums
results/      regenerated tables and standalone PNG figures
src/          preparation, analysis, robustness, and figure scripts
verify_release.py  release integrity and numerical validation
```

When citing this repository before archival release, record the exact commit hash together with the associated manuscript citation.
