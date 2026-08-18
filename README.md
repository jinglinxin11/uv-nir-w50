# UV–NIR W50 Reproducibility

Reproducible analysis of apparent profile W50 under 248-nm UV and 976-nm NIR writing conditions.

## Overview

This repository accompanies the associated manuscript and its Supplementary Information. It provides the released image inputs, fixed transect configuration, coordinate transforms, Python implementation, output tables, and standalone figures for the reported apparent-W50 comparison.

The Supplementary Information is the primary source for the scientific method and interpretation. This repository provides the computational implementation and provenance needed to reproduce the released analysis; it does not replace the Supplementary Information.

## Relationship to the Supplementary Information

| Supplementary Information component | Repository implementation |
| --- | --- |
| Image inputs and analytical roles | `data/UV_248T.tif`, `data/NIR_highres_unsharpened_original.jpg`, `data/NIR_highres_cropped_to_UV_FOV.tif`, and `data/NIR_976-T_display.tif` |
| Common-coordinate and native-grid mappings | `config/registration.json` and `src/prepare_nir_crop.py` |
| Fixed X-direction transects | `config/M4_5X_ROIs.csv` |
| Profile extraction and W50 calculation | `src/measurement_core.py` and `src/run_analysis.py` |
| Measurement tables and summary | `results/tables/measurements.csv`, `profiles.csv`, and `summary.csv` |
| Supplementary profile, W50, paired-width, and Gx figures | `src/make_single_figures.py` and `results/figures/` |
| Figure 4f ROI-to-W50 workflow schematic | `src/make_figure4f_roi_workflow.py` and `results/figures/Figure4f_ROI_selection_transverse_width_workflow.png` |
| Released Supplementary Information and algorithm description | `docs/` |

## Analysis workflow

```text
UV and high-resolution NIR inputs
        ↓
stored common-coordinate transforms and fixed X transects
        ↓
11 parallel profiles normal to each local written ridge
        ↓
CIE Lab a* profile → background normalization → Gaussian filtering
        ↓
half-maximum crossings → apparent W50
        ↓
ratio of condition means: Gx = mean(W50UV) / mean(W50NIR)
```

`Gx` is calculated from the two condition means, not from the arithmetic mean of local UV/NIR ratios.

## Repository structure

```text
.
├── config/
│   ├── M4_5X_ROIs.csv                 # five fixed technical transects in four parent ROIs
│   └── registration.json              # stored UV mapping and cropped-NIR similarity mapping
├── data/
│   ├── UV_248T.tif                    # quantitative UV input
│   ├── NIR_highres_unsharpened_original.jpg
│   ├── NIR_highres_cropped_to_UV_FOV.tif
│   └── NIR_976-T_display.tif          # display/localization image only
├── docs/                               # released Supplementary Information and algorithm description
├── results/
│   ├── figures/                       # standalone PNG outputs
│   └── tables/                        # measurement, profile, summary, and manifest files
├── src/
│   ├── prepare_nir_crop.py            # derives the native-pixel NIR field-of-view crop
│   ├── measurement_core.py             # profile-sampling and W50 functions
│   ├── run_analysis.py                 # main five-transect computation
│   ├── make_single_figures.py          # individual supplementary figure outputs
│   └── make_figure4f_roi_workflow.py  # Figure 4f schematic
├── reproduce_all.py                    # released one-command workflow
└── requirements.txt
```

## Reproducing the released analysis

```bash
git clone https://github.com/jinglinxin11/uv-nir-w50.git
cd uv-nir-w50
python -m venv .venv
```

Activate the virtual environment, then install the released dependencies:

```bash
python -m pip install -r requirements.txt
python reproduce_all.py
```

`reproduce_all.py` only calls the released crop, analysis, and plotting scripts. It regenerates the NIR field-of-view crop, analysis tables, standalone PNG figures, and the Figure 4f workflow figure.

## Expected outputs

The main outputs are written to `results/tables/` and `results/figures/`.

| Validation target | Released value |
| --- | ---: |
| Valid condition-by-transect measurements | 10 |
| UV mean apparent W50 | 14.5261048734 reference px |
| NIR mean apparent W50 | 6.0509975853 reference px |
| Gx, ratio of condition means | 2.4006132325 |
| Apparent W50 reduction | 58.343977% |

Widths are reported in common-reference-grid apparent pixels. The five technical transects are nested within four parent spatial ROIs.

## Reproducibility check

`src/run_analysis.py` verifies that ten condition-by-transect measurements are present and valid before writing the output tables. `results/tables/analysis_manifest.json` records the released numerical summary. The fixed transects and registration parameters are retained in `config/` so that the output values can be checked against the validation targets above.

## Data provenance

`UV_248T.tif` and `NIR_highres_cropped_to_UV_FOV.tif` are the quantitative inputs. The cropped NIR TIFF is generated from the supplied unsharpened high-resolution JPEG with a native-pixel crop; it is not resized. `NIR_976-T_display.tif` is used for registration-display and ROI-localization figures, not quantitative NIR profile extraction. The stored ROI coordinates and transformations are respectively in `M4_5X_ROIs.csv` and `registration.json`.

## Figures and source data

| Output | Script | Direct source data |
| --- | --- | --- |
| `Fig_S9_ROI_locations.png` and `Fig_S9_profile_X01–X05.png` | `src/make_single_figures.py` | ROI CSV, display image, `profiles.csv`, `measurements.csv` |
| `Fig_S10_representative_W50_X04.png` | `src/make_single_figures.py` | `profiles.csv`, `measurements.csv` |
| `Fig_S11_paired_W50.png` | `src/make_single_figures.py` | `measurements.csv`, `summary.csv` |
| `Fig_S12_Gx_proxy.png` | `src/make_single_figures.py` | `summary.csv` |
| `Figure4f_ROI_selection_transverse_width_workflow.png` | `src/make_figure4f_roi_workflow.py` | display image and ROI CSV; panel (c) is an explicitly illustrative measurement schematic |

## Environment

Install the required Python packages using `requirements.txt`. The released analysis has been run with Python and the package set listed in that file.

## Citation

If you use this repository, please cite the associated manuscript. Full citation information will be added after publication.

## Version corresponding to the manuscript

The manuscript-associated release will be archived upon publication. Until then, use the Git commit shown by `git rev-parse HEAD` when recording the exact code version.
