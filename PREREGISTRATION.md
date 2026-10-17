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


### Deviation 1 (17 Oct 2026, before any probe data): ZUNA leaks the hidden channels' own scale

Reading ZUNA 1.1.6's code (`eeg_data.py`, v4 path) showed that each channel is z-scored per 5 s
segment with **its own** mean/std — including channels marked bad — and the model output for masked
cells is mapped back with those same statistics; the per-segment average reference also includes
the masked channels. Confirmed empirically (`jobs/leak`, 6 recordings x 8 hidden channels, Kaggle T4):

| hidden channels filled with | output std / true std | correlation | NMSE |
|---|---|---|---|
| true signal | 1.00 | 0.965 | 0.069 |
| true x 10 | 9.39 | 0.921 | 73.2 |
| true x 0.1 | 0.16 | 0.955 | 0.73 |
| white noise (median visible std) | 0.88 | 0.953 | 0.27 |
| zeros | 0.09 | 0.928 | 0.85 |

So ZUNA predicts the normalised waveform *shape*; the amplitude comes from whatever the hidden
channel contains. Consequences for this study (decided before any probe data):
1. **Leak-proof input.** Each recording is re-referenced to the average of the visible electrodes and
   hidden channels are zero-filled, so neither the per-channel statistics nor the average-reference
   offset carry hidden information.
2. **Shape score replaces NMSE.** E = mean over hidden channels and 5 s segments of (1 − Pearson r),
   for every method alike (amplitude-free; ZUNA's native output).
3. Under a nonlinear score, linear smoothers no longer tie exactly on real vs patterns recordings, so
   the H1 band is the range of physics-use scores of a family of linear smoothers (splines, IDW,
   best linear, smoothed linear), estimated offline with large n (`experiments/null_band.py`);
   the H2 reference is an oracle RAP-MUSIC model with the true lead field.

## Probe 1 — FINAL settings and decision rules (17 Oct 2026; supersedes the draft above; committed before the run)

**Protocol.** Leak-proof input (visible-average reference, hidden channels zero-filled), shape error
E = mean over hidden channels and the two 5 s segments of (1 − r). One mask per recording, cycled
identically across kinds (masks in `jobs/probe1/config.json`; T9/T10/Iz never hidden).
ZUNA 1.1.6, 25 steps, one posterior sample, 2 × T4 (Kaggle). Seed 2027.

**Sample.** k = 1: 200 real + 200 patterns recordings (primary); k = 2 and k = 4: 100 + 100 each.
10 s recordings, 256 Hz, 64 channels, 20 dB sensor noise.

**Score.** score_m = mean_patterns(E_m − E_spline) − mean_real(E_m − E_spline), all masks pooled,
95% bootstrap CI over recordings (5000 resamples).

**Reference values fixed in advance** (`results/null_band.json`, n = 400 per kind, same protocol):

| k | H1 band: linear smoothers (IDW, best linear, smoothed linear; splines = 0) | H2 reference: oracle physics (RAP-MUSIC) |
|---|---|---|
| 1 | −0.029 … +0.027 | +0.265 |
| 2 | −0.017 … +0.037 | +0.151 |
| 4 | −0.014 … +0.023 | +0.091 |

**Decision (k = 1, primary).**
- **H2 — uses head physics beyond any smoother:** lower CI bound of ZUNA's score > +0.027 (top of the
  H1 band) **and** ZUNA's score at k = 1 > its score at k = 4.
- **H1 — behaves like a smoother:** upper CI bound < +0.060 (i.e. ZUNA exceeds the best smoother by less
  than ~1/8 of the oracle's physics use).
- Otherwise **inconclusive**. In every case we also report ZUNA's score as a fraction of the oracle's.

**Validity check (pre-specified).** If ZUNA's mean shape error on real k = 1 recordings exceeds that of
IDW (offline: 0.222), the leak-proof input is judged to have broken ZUNA and the run is not interpreted.

**Secondary.** Scores per mask type (scattered / patch) and for k = 2; dose–response across k = 1, 2, 4;
ZUNA vs splines and vs best linear on real recordings.

### Probe 1 — OUTCOME (recorded 17 Oct 2026, 16:40 JST)

Run: Kaggle 2 × T4, 800 recordings, 258 min, all completed (`results/kaggle/probe1/38998916959`).

**Primary score (k = 1, all masks):** ZUNA +0.138 [+0.097, +0.178]. Lower bound > +0.027 (top of the H1
band) and score(k = 1) +0.138 > score(k = 4) +0.104 — the H2 criterion is met **by the letter**.

**Validity check: FAILED.** ZUNA's mean shape error on real k = 1 recordings is 0.234, above IDW's
0.222 (offline; 0.202 in this run). By the rule fixed in advance, the run is **not interpreted as
evidence for H2**.

**What the numbers show (descriptive, not a confirmatory claim):** ZUNA's positive score comes from
breaking down on impossible patterns (error 0.444 vs splines 0.224), not from beating smoothers on
possible ones (0.234 vs splines 0.152; ZUNA better than splines on only 27% of real k = 1 recordings).
The physics oracle shows the opposite profile (0.085 on possible patterns). ZUNA's score also barely
fades with k (0.138 → 0.133 → 0.104) while the oracle's does (0.265 → 0.151 → 0.091).
Exploratory reading: ZUNA's errors are strongly sensitive to physical realism (a violation-of-
expectation signature), but on these synthetic recordings it does not use that knowledge to fill in
electrodes better than splines. Candidate explanations to test next, each pre-registered separately:
(a) one diffusion sample instead of a posterior mean; (b) synthetic recordings too clean / unlike
training data; (c) the leak-proof input itself.
