# Provenance records

The released analysis reads `config/transects.csv` as its sole transect input. It contains only fixed geometry, with five transects in four parent ROIs and no condition-specific width, PNR or gain outcomes. `src/build_release_metadata.py` verifies this schema and the immutable checksum of the actual input directly. The locations were constructed from skeleton-derived candidates and hierarchical sub-ROIs; the release does not include the complete candidate-generation history or evidence establishing selection before inspection of results. They are therefore described as fixed rather than prospectively prespecified.

The released images are sufficient to reconstruct the high-resolution NIR similarity transform and to run diagnostic checks on both frozen coordinate mappings. The historical control points used to establish the UV rotation and translation are not included, so `src/estimate_registration.py` labels the UV mapping as structurally and phase checked rather than independently re-estimated. This boundary is recorded explicitly in `results/tables/registration_reconstruction.json`.

`input_checksums.sha256` is an immutable release record and is checked before derived artifacts are regenerated. Observed hashes from each run are written separately to `results/tables/input_checksums_observed.sha256`.
