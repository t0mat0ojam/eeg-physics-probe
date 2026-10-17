# Finding: ZUNA 1.1 takes a hidden electrode's loudness from the electrode itself (17 Oct 2026)

**What.** ZUNA 1.1.6 z-scores every channel per 5 s segment using that channel's own mean and std,
*including channels marked bad / to be repaired*, and maps the model output for masked cells back with
those statistics. The per-segment average reference also includes the masked channels.

**Test** (`jobs/leak`, Kaggle T4, 6 synthetic recordings x 8 hidden electrodes, same recordings, hidden
electrodes filled 5 ways):

| hidden electrodes filled with | output loudness / true | shape correlation | error (NMSE) |
|---|---|---|---|
| true signal | 1.00 | 0.965 | 0.069 |
| true x 10 | 9.39 | 0.921 | 73.2 |
| true x 0.1 | 0.16 | 0.955 | 0.73 |
| neutral noise | 0.88 | 0.953 | 0.27 |
| zeros | 0.09 | 0.928 | 0.85 |

**Meaning.** ZUNA predicts the normalised waveform shape well (r ~ 0.95 regardless of fill), but the
amplitude of a repaired channel is whatever the broken channel contained. With honest neutral fill the
error is ~4x higher than with the true signal left in place. For real repair (flat or noisy channels)
the output amplitude will be wrong. Whether the published benchmarks are affected is not known from here.

**Consequence for the physics probe.** Leak-proof input (visible-average reference, hidden channels
zero-filled) and an amplitude-free shape score for all methods.
