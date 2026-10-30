# Research methodology

**Design.** Controlled factorial simulation study + (conditional) real-world check. Unit of replication: one independent stream run (seed).
Policies and detector settings are compared on identical streams (common random numbers) and analysed with paired statistics.

**Streams.** SEA-style (3 uniform features, 2 relevant; concept A: y = 1[x1+x2 > 8]; 5 % label noise). Scenarios: stationary; covariate-shift control (x1 ~ U[3,10] after the change; labelling rule unchanged);
abrupt concept swap (label rule x2+x3 > 8; P(X), P(Y) unchanged; label disagreement with A about 0.30); abrupt threshold shift (8 -> 9.5; disagreement about 0.13; P(Y) also changes);
gradual swap with linear mixture ramp of width 1000 and 2500 (Bernoulli concept draw per example with p(t) = (t - start + 0.5)/width); recurring swap (A-B-A-B at 3 change points).

**Learner.** River `HoeffdingTreeClassifier` (grace period 100, delta 1e-5, leaf prediction `nba`); `GaussianNB` in a small sub-experiment.

**Detectors.** ADWIN (delta in {0.02, 0.002 (library default), 0.0002}; accepts only error *increases*) and Page-Hinkley (threshold in {25, 50 (default), 100}; mode `up`; delta 0.005, alpha 0.9999, min_instances 30) on the binary error indicator.

**Policies.** none; fixed (every W = 2000 observations); drift-triggered (accepted alarm). Action: reset (fresh model) or replay (fresh model trained on the last R = 500 labelled observations).

**Metrics and statistics.** See `experimental_protocol.md` and notebook Sections 10-14.

**Integrity rules.** Parameters fixed before running; no tuning on outcomes; failed runs recorded (none occurred in the final run - see `validation_report.md`); no fabricated data or citations; smoke-profile numbers never reported.
