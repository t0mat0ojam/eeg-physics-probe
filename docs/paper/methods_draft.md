# Methods (draft, written before probe-1 results)

## 1. Question and competing hypotheses
EEG foundation models fill in missing electrodes well. Two accounts predict that equally:
**H1 (smoothing)** — the model interpolates hidden channels from spatial neighbours, as spherical
splines do; **H2 (physics)** — the model has internalised the structure that volume conduction
imposes on scalp fields (fields are projections of a few compact cortical sources through the head),
and uses it to infer hidden channels. Accuracy alone cannot separate them. We design a test where
the two accounts make different, falsifiable predictions.

## 2. Matched possible and impossible scalp patterns
Recordings (64-channel EEGMMIDB layout, 256 Hz, 10 s) are generated from a 4-shell spherical head
model (MNE) with a 1227-point source grid.
- *Real:* k dipoles (stratified locations, random orientations, log-normal amplitudes), each with a
  rhythmic (8–13 Hz bursts) plus 1/f time course, projected through the lead field.
- *Patterns (control):* the same number of sources, time courses and amplitudes, but each source's
  scalp pattern is a random smooth field drawn from N(0, Σ1), where Σ1 = E[l lᵀ] is the ensemble
  covariance of single-dipole fields.
Both kinds share one global gain and 20 dB white sensor noise, so their ensemble spatial covariance
is identical by construction. Consequently any fixed linear infill method has the same expected
mean-squared error on both kinds (verified with n = 300 per kind). A method can do better on *real*
patterns only by exploiting structure beyond second-order statistics — the geometry of fields that
a head can actually produce.

## 3. A scale leak in the reference implementation, and a leak-proof protocol
ZUNA 1.1 z-scores each channel per 5 s segment with that channel's own statistics — including
channels marked for repair — and maps the model output back with those statistics; the per-segment
average reference also includes masked channels. We confirmed that the output amplitude of a
repaired channel tracks whatever the channel contains (×10 content → ×9.4 output; honest neutral
fill raises error ~4×), while predicted waveform shape is nearly unaffected (r ≈ 0.93–0.97).
We therefore (i) re-reference every recording to the average of the visible electrodes and
zero-fill hidden channels, so no statistic computed by the pipeline depends on hidden data, and
(ii) score all methods by an amplitude-free **shape error** E = mean over hidden channels and 5 s
segments of (1 − Pearson r).

## 4. Readout: a physics-use score with calibrated reference ranges
score_m = mean over pattern recordings of (E_m − E_spline) − mean over real recordings of (E_m − E_spline).
Positive scores mean method m gains more over splines on physically possible patterns than on
impossible ones. Because E is nonlinear, linear smoothers do not score exactly zero; we calibrate an
**H1 band** from a family of linear smoothers (inverse-distance weighting, best linear predictor with
the true covariance, a smoothed linear predictor) and an **H2 reference** from an oracle that knows
the head model (RAP-MUSIC source scan with the true lead field, then least squares), each with
n = 400 per kind. A dose–response prediction follows from H2: the score must shrink as the number of
simultaneous sources k grows and fields approach Gaussian mixtures.

## 5. Masks, models and statistics
Six fixed masks (3 scattered × 16 hidden, 3 contiguous patches × 12 hidden; T9/T10/Iz, which fall
outside ZUNA's position range, never hidden), one per recording, cycled identically across kinds.
ZUNA 1.1.6, 25 diffusion steps, one posterior sample per recording, run on 2 × T4 GPUs. Primary test
at k = 1 (200 + 200 recordings), dose–response at k = 2 and 4 (100 + 100 each). 95% bootstrap CIs
over recordings. Decision rules and reference values were committed before the run
(PREREGISTRATION.md).
