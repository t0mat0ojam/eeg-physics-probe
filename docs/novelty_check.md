# Novelty check: "Did the EEG foundation model learn the physics of the head?" (searched 2026-10-17)

**Bottom line: I found no paper that already runs this test.** No paper compares how well a foundation model or deep interpolator infills lead-field-realizable scalp fields versus matched non-realizable ones. Prior work handles the physics in only three ways:
- **Built into the model:** DeeperBrain, NeuroDyn-EEG.
- **Used as an analogy:** BrainOmni.
- **Used as a post-hoc tool:** EEG-PRISM.

None of them tests whether a model learned the physics.

## 1. Prior work

| Paper | Year/venue | What it does | How it differs from the planned paper | URL |
|---|---|---|---|---|
| ZUNA (Warner et al., Zyphra) | arXiv 2602.18478 (on the ICLR 2026 site) | 380M diffusion autoencoder with 4D RoPE over xyz + time. Beats spherical splines at channel infill and upsampling. | Says splines are a strong baseline because of volume conduction. Has no lead-field, synthetic-data or interpretability analysis; lists mechanistic interpretability as future work. | https://arxiv.org/abs/2602.18478 |
| ZUNA1.1 (Warner et al.) | arXiv 2607.27308, Jul 2026 | Infills arbitrary channels and time spans, inputs up to 30 s. Beats MNE splines. | No physics analysis. | https://arxiv.org/abs/2607.27308 |
| DeeperBrain (Wang et al.) | arXiv 2601.06134, 2026 | Builds volume conduction into the model with a learnable exp(-d/τ) kernel on 3D coordinates. Fig. 4 shows a smooth "receptive field" at Cz, against an impulse-like one for the baseline. | The physics is built in, and the evidence is one qualitative plot. No synthetic forward-model data and no infill evaluation. The lead field appears only as motivation. | https://arxiv.org/abs/2601.06134 |
| BrainOmni (Xiao et al.) | NeurIPS 2025 | Joint EEG/MEG model. A sensor encoder takes position, orientation and sensor type; latent "sources" come from learned cross-attention, framed as the inverse/forward solution. | The lead field is only conceptual (App. B); no head model is computed. App. J shows attention topomaps, qualitatively only. No physical-consistency test. | https://arxiv.org/abs/2505.18185 |
| NeuroDyn-EEG (Cui et al.) | arXiv 2609.36773, 2026 | Pretrained on 10^6 synthetic samples from a Jansen-Rit neural mass model projected through an MNE lead field (90 regions to 19 channels). | Physics comes from the synthetic training data by construction. Does not test learned physics in generic foundation models. | https://arxiv.org/abs/2609.36773 |
| EEG-PRISM (Shama et al.) | arXiv 2608.13676, 2026 | Maps foundation-model attributions into source space with a forward/inverse model. Uses SEREEGA simulations and 5 foundation models. | Uses the head model to interpret predictions, not to test the physics of reconstruction. | https://arxiv.org/abs/2608.13676 |
| Kwon et al., "Super-resolution for EEG... feasibility" | Sensors 2019 | CNN upsampler trained and tested on BEM-simulated dipoles; reports source-localization error. | The closest synthetic head-model evaluation of a deep upsampler. No impossible-field control, no foundation model, no spline baseline. | https://www.mdpi.com/1424-8220/19/23/5317 |
| Svantesson et al., "Virtual EEG-electrodes" | bioRxiv 2020 | CNN versus MNE splines on real EEG; the CNN is mostly better. | Signal-similarity metrics only. | https://www.biorxiv.org/content/10.1101/2020.04.20.049916 |
| RESIT (Dong et al.) | Brain Topogr. 2021 | Interpolates from equivalent sources through a 3-sphere lead field; beats splines. | A classical physics baseline. | https://link.springer.com/article/10.1007/s10548-021-00844-2 |
| Mellot et al. | arXiv 2403.15415, 2024 | Physics-based field interpolation to a common montage, compared with splines. | A physics baseline; no deep models. | https://arxiv.org/abs/2403.15415 |
| EMAG; coordinate-queryable neural field; Neural Brain Fields | arXiv 2605.29731, 2606.23707, 2601.00012 | New deep EEG upsamplers: Gaussian "sources" rendered to the scalp, or implicit neural fields. | None tests physical consistency. Possible extra models to test. | https://arxiv.org/abs/2605.29731 , https://arxiv.org/abs/2606.23707 , https://arxiv.org/abs/2601.00012 |
| Foundation-model interpretability: sparse autoencoders (Lehn-Schiøler et al.); clinical-feature probing; robustness/attribution (Širca et al.); EEG-FM-Audit | arXiv 2605.13930, 2605.11410, 2605.17562, 2605.26910 | Sparse autoencoders, probing for clinical features, perturbation and attribution, noise injected into regions of interest. | None probes volume conduction. Širca et al. note that their noise ignores volume conduction. | https://arxiv.org/abs/2605.13930 , https://arxiv.org/abs/2605.11410 , https://arxiv.org/abs/2605.17562 , https://arxiv.org/abs/2605.26910 |
| **Analogue:** Garrido et al. (Meta) | arXiv 2502.11831, 2025 | Violation of expectation: compares V-JEPA's surprise on matched possible and impossible videos. | The method template. | https://arxiv.org/abs/2502.11831 |
| **Analogue:** Vafa et al. | ICML 2025 | Probes inductive bias with synthetic data from a known world model. Models predict orbits well but do not learn Newtonian mechanics. | Template for "accurate reconstruction ≠ learned physics". | https://arxiv.org/abs/2507.06952 |
| **Analogue:** Hakim & Masanam | AIES 3(3), 2024 | Tests Pangu-Weather with idealized perturbations (tropical heating, geostrophic adjustment). | Template: synthetic inputs where the physics gives a known answer. | https://arxiv.org/abs/2309.10867 |

## 2. Open EEG foundation models

| Model | Venue | Public weights | Infills arbitrary hidden channels out of the box? | Channel-layout flexibility |
|---|---|---|---|---|
| ZUNA / ZUNA1.1 | arXiv 2602.18478 / 2607.27308 | Yes: HF Zyphra/ZUNA and Zyphra/ZUNA1.1, Apache-2.0, `pip install zuna` | **Yes.** Built for this, including channels at any xyz. | Any layout (xyz RoPE) |
| BrainOmni | NeurIPS 2025 | Yes: HF OpenTSLab/BrainOmni (tiny, base, tokenizer) | **Probably.** The tokenizer's reconstructor is trained with 25% of channels dropped and decodes every sensor from its position. Confirm the decoder is in the checkpoint. | Any layout; EEG and MEG |
| REVE | NeurIPS 2025 | Yes: HF brain-bzh/reve-base (collection brain-bzh/reve) | Maybe. It is a masked autoencoder trained with 3 cm spatial block masking and whole-channel dropout. Unclear whether the decoder is released; infill was never evaluated. | Any layout (4D Fourier positional encoding) |
| LUNA | NeurIPS 2025 | Yes: HF thorir/LUNA (7M to 311M), CC BY-ND 4.0 | Partial. Masked-patch decoder with queries indexed by channel. Unclear whether the decoder is released. | 3D coordinates plus learned queries; weak on dense unseen layouts |
| CBraMod | ICLR 2025 | Yes: HF weighting666/CBraMod, MIT | Partial. Has a raw-patch reconstruction head, but it was trained with random patch masking and takes no coordinates. | Convolutional positional encoding; no geometry |
| EEGPT | NeurIPS 2024 | Yes: figshare (58-channel large checkpoint) | Partial. Has a reconstructor for masked raw patches, trained with 80% of channel patches masked; works only within its 58 named channels. | Fixed 58-channel vocabulary |
| LaBraM | ICLR 2024 | Yes (GitHub) | No: predicts quantized (VQ) codes, not raw signal | Named-channel vocabulary |
| NeuroLM | ICLR 2025 | Yes: HF Weibang/NeuroLM | No: quantized tokens fed to an LLM | Named channels |
| BIOT | NeurIPS 2023 | Yes (GitHub; 16- and 18-channel checkpoints) | No: embeddings only | Fixed montages |
| BENDR | arXiv 2101.12037 (2021) | Yes (GitHub release) | No: contrastive pretraining | Fixed |
| DeeperBrain | arXiv 2026 | No ("will be released") | Pretrained on masked channel-second patches; infill not evaluated | 3D coordinates plus decay kernel |
| NeuroDyn-EEG | arXiv 2026 | Code only | No: outputs neural mass model parameters | Fixed 19 channels |

**Suggested test set:** ZUNA1.1 as the main model, then BrainOmni, then REVE if its decoder is available. CBraMod and EEGPT can serve as contrasts with fixed or no geometry.

## 3. Verdict

**What the paper can claim as novel:**
1. The first falsifiable test of spatial smoothing versus an implicit forward model in EEG foundation models.
2. The first use of non-realizable control fields matched in smoothness and dynamics.
3. Two bracketing baselines: spherical splines (pure smoothing) and lead-field minimum-norm reconstruction (pure physics).
4. A comparison across several open models, linked to how well each transfers to unusual montages.

**Closest prior work:**
- DeeperBrain §IV-G1 is the only paper that claims a foundation model's spatial field reflects volume conduction. Its kernel is hand-built, and the evidence is one qualitative plot.
- Kwon et al. 2019 is the closest in method: a deep upsampler evaluated on BEM-simulated dipoles.

**Main novelty risks:**
- **Being scooped.** Zyphra lists mechanistic interpretability as future work, and at least six papers interpreting or auditing EEG foundation models appeared between May and Sep 2026.
- **A confound reviewers will raise.** Every training field is realizable, so lower error on realizable fields may only show that the model learned the data manifold, not that it runs an internal forward model.
  - Fields with low spatial frequency on a sphere are largely inside the lead field's range. "Matched smoothness but outside the range" fields may therefore be hard to build, or unusual in other ways.
  - Frame H2 as "learned the manifold of realizable fields" and include controls for distribution shift.
