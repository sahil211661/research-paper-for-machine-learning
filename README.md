# When Does a Machine Learning Model Become Unreliable?
### Concept drift detection, performance degradation and model recovery — a reproducible empirical study

MSc Data Science research project. The primary deliverable is **`notebooks/concept_drift_research.ipynb`**
(stored with the outputs of the validated full run; the unexecuted source is `scripts/notebook_unexecuted_source.ipynb`, rebuilt by `scripts/build_notebook.py`).

## Objectives
1. Quantify how abrupt, gradual, recurring and covariate shifts affect an online classifier (River Hoeffding tree) — **RQ1**.
2. Compare *no adaptation*, *fixed-window rebuild* and *drift-triggered rebuild* (ADWIN / Page-Hinkley; reset or replay) — **RQ2**.
3. Measure trade-offs between accuracy, detection delay, false alarms, recovery time and computational cost — **RQ3**.

Hypotheses H1–H3 are tested with paired statistics over 20 independent seeds (Holm-corrected). The study does **not** claim a new
algorithm or novelty of the detectors/policies; results and conclusions are generated from executed experiments only.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate     # Python >= 3.10 (validated: 3.12.3)
pip install -r requirements.txt                       # exact validated versions: requirements-validated.txt
pytest                                                # 43 unit tests, no downloads needed (about 4 s)
jupyter lab notebooks/concept_drift_research.ipynb    # Kernel -> Restart & Run All
```

### Execution order / expected runtime (one CPU core, no GPU)
| Step | Command | Time |
|---|---|---|
| Unit tests | `pytest` | ~4 s |
| Smoke check of the pipeline (**never report these numbers**) | `DRIFT_PROFILE=smoke jupyter nbconvert --to notebook --execute notebooks/concept_drift_research.ipynb` or `python scripts/run_experiments.py --profile smoke --tag smoke` | ~1 min |
| **Full study** (notebook) | Run All, or `jupyter nbconvert --to notebook --execute ...` | **~15 min** (main grid ≈ 13 min: 1540 prequential runs) |
| Full study (script) | `python scripts/run_experiments.py --profile full --tag main` | ~14 min |
| Rebuild tables from saved raw results | `python scripts/export_results.py --tag main` | seconds |
| Regenerate the notebook source | `python scripts/build_notebook.py` | seconds |

## Configuration
Everything is in `configs/default.yaml` (seeds, stream length, drift positions and widths, model/detector parameters, window sizes,
evaluation windows, repetitions, output paths) and is loaded by `src/config.py`. `make_variants()` builds configuration grids.
Profiles: `full` (default) and `smoke`, selected with the environment variable `DRIFT_PROFILE`.

## Data
* **Stage A (completed):** SEA-style synthetic streams generated from seeds (no files).
* **Stage B (real-world, Elec2):** loaded through `river.datasets.Elec2` or `data/raw/electricity.csv`. **In the delivered run the documented download host was
  unreachable from the execution sandbox, so Stage B was NOT executed** (the loader raised `DatasetUnavailableError`; no substitute dataset was used).
  To run it: see `data/README.md`, put `electricity.csv` in `data/raw/`, and re-execute the notebook — Sections 8, 17 and Figure 16 then run automatically.

## Repository layout
```
notebooks/   concept_drift_research.ipynb (+ _executed copy)
src/         config, data_streams, models, detectors, evaluation, experiments, statistical_analysis, reporting, visualization
tests/       43 unit tests (toy inputs only)
configs/     default.yaml
scripts/     run_experiments.py, export_results.py, build_notebook.py (+ nb_part*.py cell sources)
results/     raw/ (run-level CSV), logs/ (event logs, error arrays, config snapshot, env), tables/ (exported tables)
figures/     fig01 … fig17 (PNG 300 DPI + PDF; fig16 only if Elec2 was available)
reports/     literature_review, research_methodology, experimental_protocol, results_summary (generated), limitations, validation_report (generated), references.bib
```

## Outputs
Tables (`results/tables/*.csv`): overall performance, detector performance, detection delay, false alarms/missed, adaptation performance, recovery time,
computational cost, results by drift type, results by seed, statistical comparisons, H1 degradation, recovery sensitivity, Friedman omnibus, model comparison.
Figures: see `figures/README.md`.

## Troubleshooting
* `ModuleNotFoundError` — activate the virtual environment and run `pip install -r requirements.txt`.
* `DatasetUnavailableError` for Elec2 — expected without internet access; see `data/README.md`. The synthetic study is unaffected.
* Notebook cannot find the project — the first code cell searches upwards for `pyproject.toml`; open the notebook from inside the repository.
* Timings differ between machines — processing-time results are hardware-dependent; accuracy/alarm results are deterministic for a given seed and River version.
* Very long run — use `DRIFT_PROFILE=smoke` for a pipeline check, or `python scripts/run_experiments.py --n-seeds 5`.

## Licence
No licence has been chosen yet — see `LICENSE` (placeholder). The project owner must select one for the original code; third-party datasets/libraries keep their own terms.
