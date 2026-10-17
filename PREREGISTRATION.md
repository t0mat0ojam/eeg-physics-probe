# Pre-registration log — "Did the EEG foundation model learn the physics of the head?"

Each section is committed to git before the run it governs. Deviations are added below the
section, time-stamped, never edited in place.

## Probe 1 — ZUNA 1.1, synthetic recordings (DRAFT, 17 Oct 2026; sample size fixed after the bench run)

**Question.** When ZUNA fills in hidden electrodes, does it use the structure that head physics
imposes on scalp fields (H2), or does it behave like a spatial smoother (H1)?

**Recordings** (`src/physprobe/probe.py`; 64-channel EEGMMIDB cap, 256 Hz, 10 s, MNE 4-shell sphere,
1227-point source grid, average reference, one global gain per k, 20 dB white sensor noise):
- *real*: k dipoles (stratified locations, random orientations, log-normal amplitudes), each with a
  rhythmic (8–13 Hz bursts) + 1/f time course;
- *patterns* (primary control): k sources with the same time-course and amplitude process, but each
  scalp pattern is a random smooth field drawn from N(0, Σ1), Σ1 = the single-dipole ensemble
  covariance — same per-recording rank, same ensemble covariance, not a dipole field;
- *surrogate* (secondary control): spatially Gaussian with covariance k·E[a²]·Σ1, same time courses.
Verified before any ZUNA run (n = 300 per kind): splines and the best linear predictor tie across
the three kinds (`experiments/check_surrogate_match.py`).

**Infill.** Fixed masks (3 scattered × 16 hidden, 3 contiguous patches × 12 hidden; T9/T10/Iz never
hidden), one mask per recording, cycled identically across kinds. ZUNA 1.1.6, 25 diffusion steps,
posterior mean of 2 samples. References: spherical splines (H1 stand-in), minimum-norm with the
sphere lead field, best linear predictor (true covariance).

**Primary readout** (k = 1, patch masks — where smoothing and physics differ most):
physics-use score = [NMSE_ZUNA(patterns) − NMSE_ZUNA(real)] − [NMSE_bestlin(patterns) − NMSE_bestlin(real)],
pooled NMSE, 95% bootstrap CI over recordings.
- **H2 supported** if the score's CI lies above 0 **and** the score at k = 1 exceeds the score at k = 16
  (dose–response: the advantage must fade as fields become a blur of many sources).
- **H1 supported** if the CI lies within ±0.02 (equivalence margin) or below 0.
- Otherwise inconclusive.

**Secondary.** Same score for scattered masks and for the surrogate control; ZUNA vs best linear
and vs splines on real recordings; single-sample vs 2-sample mean.

**Known limitation stated in advance.** Synthetic recordings are cleaner than real EEG; a ZUNA that
learned physics from real data could still be hurt by this shift. That affects all three kinds
equally, which is why the primary readout is a difference between kinds.
