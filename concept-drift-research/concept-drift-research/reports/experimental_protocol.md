# Experimental protocol

## Prequential loop (per observation t)
1. yhat_t = predict_one(x_t) before the model has seen y_t (None -> class 0).
2. e_t = 1[yhat_t != y_t].
3. detector.update(e_t) (drift policies).
4. replay buffer append (replay actions; past data only).
5. adaptation decision from information up to t: accepted alarm (drift) or (t+1) % W == 0 and t+1 < N (fixed). On adaptation: new model; reset -> learns (x_t, y_t); replay -> learns last R buffered observations incl. t; detector re-initialised. Otherwise learn_one(x_t, y_t).
6. log predictions, alarms, adaptation positions, timing.

## Definitions (fixed in advance; see configs/default.yaml)
* Rolling window w = 200; reference window 1000 before the change; drift window = [onset, end + 1000).
* Detection window of event i = [onset_i, min(end_i + 1000, onset_{i+1}, N)). First alarm = detection (delay = alarm - onset; also reported from end);
  later alarms in the window = duplicates; alarms elsewhere = false alarms (rate per 1000 observations outside windows); no alarm = missed (delay undefined, never imputed).
* Recovery: rolling accuracy (windows entirely after onset) >= 0.95 x pre-drift reference for 200 consecutive positions; time = end of first qualifying window - onset
  (minimum w - 1); never recovered -> right-censored at the horizon (next onset or end of stream), recorded with recovered = False. Sensitivity: 0.90, 0.98.
* Control streams have no concept change: the stationary stream uses its midpoint, the covariate-shift stream its shift point, as reference points for accuracy windows.

## Experiment layout
Monitor-only: the no-adaptation error stream is passed to six detector settings (same signal for all).
In-loop: 11 policies. 7 scenarios x 20 seeds. Memory: tracemalloc in a separate pass (3 seeds). Model sub-experiment: GaussianNB vs Hoeffding tree, 2 scenarios, 10 seeds.

## Statistics
Paired per-seed differences; bootstrap CI (5000), Cohen's d_z, Wilcoxon signed-rank, Holm correction within each family; Friedman omnibus per scenario.
Practical thresholds: 1 pp drift-window accuracy, 0.5 pp overall accuracy, 100 observations recovery time.

## Real-world protocol (conditional on data availability)
Chronological single pass; policies none, fixed_replay, adwin_replay, ph_replay; no ground-truth delay; descriptive only.
