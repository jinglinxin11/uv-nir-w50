"""Run the complete released UV–NIR apparent-W50 reproduction workflow."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    ROOT / "src" / "prepare_nir_crop.py",
    # Hashes are written after the deterministic derived crop is refreshed.
    ROOT / "src" / "build_release_metadata.py",
    ROOT / "src" / "run_analysis.py",
    ROOT / "src" / "run_robustness.py",
    ROOT / "src" / "make_single_figures.py",
    ROOT / "src" / "make_supplementary_and_maintext_figures.py",
    ROOT / "src" / "make_figure4f_roi_workflow.py",
    ROOT / "verify_release.py",
]


def main() -> None:
    for script in SCRIPTS:
        print(f"\n>>> {script.relative_to(ROOT)}")
        subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)
    print("\nReproduction completed. See results/tables and results/figures.")


if __name__ == "__main__":
    main()
