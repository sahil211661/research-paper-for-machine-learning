# Limitations and threats to validity

See notebook Section 19 for the full discussion. Summary:
* **Real-world stage not executed in the delivered run**: the documented Elec2 source (River download host) was unreachable from the execution sandbox; the code path was tested on a toy fixture with the Elec2 schema (unit tests only). No claim about transfer to real data is made.
* Synthetic SEA-style streams only; one learner family (plus a small NB sub-experiment); two detectors with three a-priori sensitivity levels each; ADWIN and PH parameters are not equivalent.
* Single fixed window W = 2000 and replay size R = 500; fixed policy cost/benefit depends on these.
* Immediate-label assumption; no delayed or missing labels.
* Wall-clock timings from one shared CPU core; tracemalloc captures Python-level allocations only.
* 20 seeds: limited power for small effects; bootstrap CIs approximate; many comparisons (Holm applied within families).
* Detection delay depends on the matching window and on measuring from onset (gradual drift is only partly present early in the transition).
* Elec2 (if used): pre-normalised over the whole period by its publishers; no calendar timestamps.
* Nothing was deployed or validated in production.
