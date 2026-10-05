# Reproduction integrity fixes, 5 October 2026

The naming/structure follow-up provides `manuscript_figure_map.json` and a README
table mapping the historical S9-S12 output suffixes to the current merged
manuscript's detailed Figs. 14-17. Historical files and scientific arrays remain
unchanged; final-number delivery copies are byte-identical to their sources.

This additive maintenance revision changes verification only. The quantitative
measurement implementation, frozen geometry, registration settings and images
are unchanged. It retains the manuscript's expected numerical targets and
absolute verification tolerance of 1e-10.

`verify_release.py` now rejects non-finite widths and robustness values,
duplicate/missing condition-transect or robustness keys, inconsistent
half-maximum crossings, stale summaries and stale analysis manifests. Primary
means and gains are recomputed from the measurement table. Leave-one-parent
results are checked against the corresponding measurement subset.

Run `python -m unittest discover -s tests -v` and `python reproduce_all.py`.
No manuscript equation, figure definition or reported effect size is changed.
The earlier exact source commit remains the original study record; this fix is
a distinct maintenance revision and does not rewrite that historical commit.
