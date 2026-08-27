# Provenance records

`ROI_selection_records.csv` is the retained upstream ROI-selection record. It contains selection provenance and previously recorded outcomes and is not used as the measurement input. The released analysis reads only `config/transects.csv`, which contains frozen geometry. The complete upstream candidate-generation history is not included in this release.

The released images are sufficient to reconstruct the high-resolution NIR similarity transform and to run diagnostic checks on both frozen coordinate mappings. The historical control points used to establish the UV rotation and translation are not included, so `src/estimate_registration.py` labels the UV mapping as structurally and phase checked rather than independently re-estimated. This boundary is recorded explicitly in `results/tables/registration_reconstruction.json`.

`input_checksums.sha256` is an immutable release record and is checked before derived artifacts are regenerated. Observed hashes from each run are written separately to `results/tables/input_checksums_observed.sha256`.
