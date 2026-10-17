# Did the EEG foundation model learn the physics of the head?

Competing hypotheses for how EEG foundation models (ZUNA 1.1 first) fill in hidden electrodes:
- **H1 – smoothing:** interpolate from neighbours (like spherical splines).
- **H2 – physics:** have learned how brain sources spread through the head (volume conduction),
  i.e. the non-Gaussian structure of physically realisable scalp fields.

**Test (surrogate logic).** Real fields (k dipolar sources through a head model, later real EEG) vs
Gaussian surrogates with *identical covariance*. Every linear method (splines, minimum-norm, the best
possible linear predictor) must score the same on both, so
`physics-use score = error(surrogate) - error(real)` is ~0 for any smoother and > 0 only for a model
that uses source structure. Dose-response: the score should shrink as the number of sources k grows.

- `experiments/pilot_discriminability.py` – CPU pilot with stand-ins (no foundation model): linear methods
  score 0 (CIs include 0); an oracle dipole fit scores +0.37 at k=1, falling to ~0 at k=16. `results/fig_pilot.png`.
- `docs/novelty_check.md` – prior-work check (17 Oct 2026): no paper runs this test.
