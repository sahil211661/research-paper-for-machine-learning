#!/usr/bin/env python
"""Assemble notebooks/concept_drift_research.ipynb from the cell definitions nb_part*.py."""
import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nb_part1, nb_part2, nb_part3, nb_part4  # noqa: E402

nb = nbf.v4.new_notebook()
nb.cells = [*nb_part1.CELLS, *nb_part2.CELLS, *nb_part3.CELLS, *nb_part4.CELLS]
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
nbf.validate(nb)
out = HERE.parent / "notebooks" / "concept_drift_research.ipynb"
nbf.write(nb, out)
print(f"wrote {out} with {len(nb.cells)} cells ({sum(c.cell_type == 'code' for c in nb.cells)} code)")
