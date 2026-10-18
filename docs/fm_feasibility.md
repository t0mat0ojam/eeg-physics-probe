# Can other EEG foundation models fill in hidden electrodes under our protocol?

Date: 2026-10-18. Scope: BrainOmni, REVE, LUNA, CBraMod. For each one: is there a raw-signal decoder, how are inputs normalised, is there a leak, and how would we run it?

**Method.** I made shallow, read-only clones of the four public GitHub repos into `/home/claude/fm_repos/{brainomni,reve,luna_biofoundation,cbramod}`. I did not run or install any repo code. I read the Hugging Face model cards and file lists with WebFetch. Direct HTTP calls to huggingface.co from this sandbox were blocked (proxy 403), so I could not download weights or read safetensors headers. Anything that depends on the weight files is marked **unverified**. The line numbers below refer to the cloned HEAD on 2026-10-18.

---

## 1. BrainOmni (NeurIPS 2025)

- **Code:** https://github.com/OpenTSLab/BrainOmni (MIT).
- **Weights:** HF `OpenTSLab/BrainOmni`, about 220 MB in total, in these folders:
  - `braintokenizer/BrainTokenizer.pt` (24.4 MB) and `braintokenizer/model_cfg.json`
  - `base/BrainOmni.pt` (147 MB) and `base/model_cfg.json`
  - `tiny/`
  - The card's badge also links to `sigureling/BrainOmni`.

### Decoder / reconstruction
The model has two stages. The Stage-1 **BrainTokenizer** is a channel-infill autoencoder by construction:

- **Training masks whole channels.** `braintokenizer/model.py` L38 sets `self.mask_ratio = 0.25  # hard coded`. In L141–152, a random 25% of channels is **dropped** from the encoder input: `self.encoder(self.add_noise(x[:, n_mask_channel:]), sensor_embedding[:, n_mask_channel:])`. The decoder then reconstructs **all** channels: `x_rec = self.decoder(feature, sensor_embedding)`. The loss covers every channel (L157–165).
- **The decoder is queried by sensor position.** In `model_utils/module.py`, `BrainTokenizerDecoder` (L175–205) uses `ForwardSolution` (L34–59). The query is the sensor embedding of the *target* channel: an MLP on 6-D position plus a sensor-type embedding (L11–31). The keys and values are 16 latent "neuro" tokens. A per-channel SEANet decoder then outputs a 512-sample waveform.
- **Arbitrary positions can be queried**, including places with no physical electrode. That makes it a good fit for our physics probe.
- **Stage 2 does not reconstruct signals.** The BrainOmni LM (`brainomni/model.py`) masks latent neuro tokens, not sensor channels. It predicts RVQ codebook indices (L163–169, cross-entropy). Channel infill only needs Stage 1.
- **Is the decoder in the released checkpoint?** Most likely yes. `demo.ipynb` cell 3 calls `tokenizer.visualize()`, which runs the decoder. Its saved output plot shows raw and reconstructed traces that closely overlap (I extracted and inspected the PNG). However, the demo loads with `load_state_dict(..., strict=False)`, which would hide missing keys. **On first load, assert that there are no missing keys.**
- Whether `BrainOmni.pt` also contains the tokenizer decoder is **unverified**. It is not needed: use `BrainTokenizer.pt`.

### Input normalisation (`factory/utils.py`; preprocessing is applied per saved segment)
```
L174  eeg_mean = np.mean(eeg_data, axis=0, keepdims=True) # reset virtual reference
L175  eeg_data = eeg_data - eeg_mean
L176  eeg_std = np.std(eeg_data) + 1.0e-5 # scale as a group,perserve the magnitude relationship between eeg channels
L177  data[eeg_mask, :] = eeg_data / (eeg_std)
```
- **Common average reference over all EEG channels**, then **one global scalar std** across all channels and samples of the segment. There is no per-channel z-scoring; the code comment at L170 says "didn't do per channel z-score".
- **This contradicts the paper.** The arXiv HTML says each channel is "normalised to zero mean and unit variance at the sample level". Which variant trained the released weights is **unverified**; the code is what ships.
- **Segment length:** tokenizer pretraining used 10 s segments (`script/pretrain_preprocess.sh`: "we train tokenizer using time=10 stride=10"), so the statistics are per 10 s segment.
- **Reconstruction target:** each channel is z-scored per 2 s window (`braintokenizer/model.py` L118–125: `x = x - x.mean(dim=-1,...)`; `x = x / (x.std(dim=-1,...)+1e-6)`). The output is therefore in **per-channel, per-window z-units**, with no amplitude and no offset.
- **Other preprocessing:**
  - 256 Hz, 0.1–96 Hz band-pass, notch at 50 and 60 Hz (`constant.py` L4–6; `factory/utils.py` L150–151, L143).
  - Auto-detected bad channels are spherical-spline interpolated (L295). So part of the pretraining data contains spline-interpolated channels, which is a geometric prior the model may have absorbed.
- **Positions:**
  - EEG `pos = [x, y, z, 0, 0, 0]` from MNE `info['chs'][i]['loc'][:3]`, in metres (L81).
  - Centred and scaled over all EEG channels present (L155–166: `eeg_scale = sqrt(3*mean(sum(pos**2)))`).

### Leak analysis
- **Pretraining reference leak (likely; same family as ZUNA's).** The average reference (L174) and the global std (L176) were computed over **all** channels, *before* the tokenizer dropped 25% of them. After an average reference, the channels sum to zero at every sample. So the sum of the masked channels equals minus the sum of the visible ones. With one hidden channel, it can be recovered exactly.
  - The model may have learned to exploit this.
  - **Our protocol removes it:** reference to the visible average and take the std from visible channels only.
  - **Direct test:** compare infill accuracy with an all-64 average reference against a visible-only average reference. A large gap means the model uses the leak.
- **Output-scale leak.** Getting volts back for a hidden channel would need that channel's own per-window mean and std, which is the target itself. Two leak-free options:
  - score with scale-free metrics (Pearson r per window); or
  - set the scale from a leak-free estimate, such as the std of a spherical-spline interpolation from visible channels, or a per-electrode gain fitted on held-out subjects.
- **Position normalisation** uses all 64 positions. That is fine: positions are known in advance and are not signal.

### Requirements
| Item | Requirement |
|---|---|
| Sampling rate | **256 Hz exactly** (matches our protocol) |
| Window | 512 samples (2 s); a 10 s recording is exactly 5 windows |
| Max channels | No hard limit (attention over channels; also trained on MEG) |
| Positions | 3-D coordinates plus sensor type (EEG = 0). Channel names are not used. |
| Referencing | Average reference during pretraining |
| Pretraining data | 11 OpenNeuro-style EEG sources (paper Appendix D). PhysioNet-MI appears only as a downstream task. Its preprocessing file is `downstream/dataset_maker/make_PhysioNet-MI.py`. |

- `share/custom_montages/physio_info.pkl` holds montage info for EEGMMIDB. It is a pickle, so **do not load it** (loading a pickle can run code). Use the MNE `standard_1005` montage instead.

### Recipe (Kaggle T4)
```python
# git clone --depth 1 https://github.com/OpenTSLab/BrainOmni ; pip install einops einx vector-quantize-pytorch mne
import sys, types, json, numpy as np, torch, torch.distributed as tdist, mne
ds = types.ModuleType("deepspeed"); ds.comm = tdist                  # model_utils/vq.py imports deepspeed.comm
sys.modules.update({"deepspeed": ds, "deepspeed.comm": tdist}); sys.path.insert(0, "BrainOmni")
from huggingface_hub import snapshot_download; from braintokenizer.model import BrainTokenizer
ck = snapshot_download("OpenTSLab/BrainOmni", allow_patterns=["braintokenizer/*"]) + "/braintokenizer"
tok = BrainTokenizer(**json.load(open(f"{ck}/model_cfg.json")))
miss, unexp = tok.load_state_dict(torch.load(f"{ck}/BrainTokenizer.pt", map_location="cpu", weights_only=True), strict=False)
assert not miss, miss                                                # demo uses strict=False -> check yourself
tok = tok.float().cuda().eval()                                     # eval(): VQ updates EMA codebooks in train mode; T4 has no bf16
P = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
pos = np.stack([np.r_[P[n], 0, 0, 0] for n in names64]).astype("float32")      # metres, 6-D
pos[:, :3] -= pos[:, :3].mean(0); pos[:, :3] /= np.sqrt(3 * np.mean((pos[:, :3] ** 2).sum(1)))  # = normalize_pos, all 64
vis = torch.tensor(~hidden_np).cuda()                               # x: (64, 2560) volts @256 Hz, 0.1-96 Hz
xv = x[~hidden_np] - x[~hidden_np].mean(0, keepdims=True); xv /= xv.std() + 1e-5   # visible-only CAR + global std
with torch.no_grad():
    se = tok.sensor_embed(torch.tensor(pos)[None].cuda(), torch.zeros(1, 64, dtype=torch.long).cuda())
    f = tok.encoder(tok.unfold(torch.tensor(xv, dtype=torch.float32)[None].cuda()), se[:, vis])  # DROP hidden (as trained), don't zero-fill
    f, _, _ = tok.quantizer(f)
    rec = tok.decoder(f, se)                                        # (1, 64, 5, 512) at all 64 positions
rec_hidden = rec[0, ~vis].reshape(int((~vis).sum()), -1)            # per-channel / per-2 s z-units -> score scale-free
```
**Blockers:** none hard.
- `deepspeed.comm` needs either the stub above or `pip install deepspeed`.
- `torch.nn.utils.weight_norm` prints a deprecation warning but works in torch 2.x.
- Do not feed zero-filled hidden channels. Training removed masked channels, so drop them instead; this changes nothing about what the model sees of the hidden channels.

---

## 2. REVE (NeurIPS 2025)

- **Code:** https://github.com/elouayas/reve_eeg (linked from https://brain-bzh.github.io/reve/).
- **Weights:** HF `brain-bzh/reve-base` (69.2M params; `model.safetensors` 277 MB), `brain-bzh/reve-large` (0.4B) and `brain-bzh/reve-positions` (position bank).
- **License:** "REVE Responsible Use License v1.0". I saw no gating notice on the page.

### Decoder / reconstruction: **encoder only is released**
- **The MAE decoder exists only in the training code.** `src/models/mae.py` has a 4-layer transformer decoder (`src/configs/decoder/base.yaml`: dim 512, depth 4), a `mask_token` and `to_pixels = nn.Linear(decoder_dim, patch_size)` (L27–35). It predicts raw patches of masked tokens with an L1 loss (L120–125).
- **The HF class has no decoder.** `Reve(PreTrainedModel)` in `hf/reve-base/modeling_reve.py` (L299–362; the same file is on HF) holds only the patch embedding, the 4D positional encoding, a 22-layer transformer, `final_layer = nn.Identity()` and `cls_query_token`. There is no `mask_token`, decoder or `to_pixels`.
- **Pretraining saved only the encoder:** `src/train.py` L99 `accelerator.save(reve_encoder.state_dict(), "encoder.pth")`.
- **The file size matches an encoder-only model:** 69.2M × 4 bytes ≈ 277 MB.
- **The paper confirms it:** "the decoder is discarded, and only the encoder is used".
- **No other release found.** The HF org lists only reve-base, reve-large and reve-positions. GitHub issues were not reachable from here (robots/API blocked). The braindecode REVE page was also blocked (the URL permission request was not answered in time). Any other decoder release is therefore **unverified**, but I found none.

### Input normalisation
- **Pretraining** (`src/utils/data_loading.py` L708–717):
  - `eeg -= stats[0]; eeg /= stats[1] + eps`
  - **Per-channel z-score** with per-session statistics, stored as `stats` of shape `(n_sessions, 2, n_chans)` (L657–662). The paper says "Z-score normalization with statistics computed across the recording sessions".
  - Then `eeg.float().clip(-self.clip, self.clip)` with `clip: 15` (`src/configs/preprocessing/default.yaml` L2), so values are clipped at ±15 SD.
- **Downstream PhysioNet-MI:**
  - average reference, 0.3 Hz high-pass, 60 Hz notch, resample to 200 Hz, µV (`preprocessing/preprocessing_physio.py` L138–152)
  - divide by a fixed `scale_factor: 100.0` (`src/configs/task/physio.yaml` L15; `src/downstream_tasks/dataloaders.py` L56)
  - The `clip: 100` key in the task YAMLs is not referenced by any downstream Python code I could grep.
- **Leak:**
  - Inputs: z-scoring is per channel, so it does not mix channels. MAE drops masked tokens from the encoder (`mae.py` L87), so hidden data never reaches the encoder. No input leak if hidden channels are removed.
  - Outputs: the MAE target is the masked channel's *own* z-scored signal. Mapping back to volts would need that channel's own session std, which is the same scale-leak issue as above.
  - Pretraining did include whole-channel masking (see the next item), which matches our protocol.
- **Masking:** spatio-temporal blocks (55% of tokens; spatial radius 0.03 m; 3-patch temporal blocks). In addition, "dropout" channels (10% of channels within a 0.04 m cluster) are masked over the **whole** time series (`data_loading.py` L528–579; config L4–12).

### Requirements
| Item | Requirement |
|---|---|
| Sampling rate | 200 Hz |
| Patches | 200 samples with 20 overlap (step 180); 10 s = 2000 samples = 11 patches (same as the 2000-sample pretraining window) |
| Max channels | Any number |
| Positions | xyz in metres via `brain-bzh/reve-positions`. All 64 EEGMMIDB names are in the bank (I checked `hf/reve-positions/positions.json`, 543 names). |
| Referencing | Not stated for pretraining; average reference for downstream PhysioNet |

- **Pitfall:** `RevePositionBank.forward` silently **drops unknown names** (`position_bank.py` L50–53), which can misalign channels.

### Recipe: embeddings only; raw infill is BLOCKED
```python
from transformers import AutoModel
bank = AutoModel.from_pretrained("brain-bzh/reve-positions", trust_remote_code=True)
reve = AutoModel.from_pretrained("brain-bzh/reve-base", trust_remote_code=True).cuda().eval()
# x: (64, 2000) @200 Hz; per-channel z-score over the recording + clip(-15, 15) (pretraining convention)
xv = torch.tensor(z[~hidden], dtype=torch.float32)[None].cuda()               # drop hidden (MAE-style)
pv = bank([n for n, h in zip(names64, hidden) if not h])[None].cuda()
assert pv.shape[1] == xv.shape[1]                                             # bank silently drops unknown names
with torch.no_grad(): tokens = reve(xv, pv)                                   # (1, Cvis, 11, 512) - latent only
# Raw-signal infill needs mask_token + 4-layer decoder + to_pixels (src/models/mae.py), not released.
```
**Blocker:** no released decoder. There are two ways forward:
- **(a)** Ask the authors for the full MAE checkpoint.
- **(b)** Re-implement the decoder from `mae.py`/`backbone.py` and train it on frozen REVE with whole-channel masking on separate subjects. The physics would then partly come from *our* decoder, so this is a different experiment.

---

## 3. LUNA (NeurIPS 2025)

- **Code:** https://github.com/pulp-bio/BioFoundation, file `models/LUNA.py`.
- **Weights:** HF `thorir/LUNA` resolves to `PulpBio/LUNA`: `LUNA_base.safetensors` (26.7 MB), `LUNA_large` (163 MB), `LUNA_huge` (1.22 GB).
- **License:** weights are **CC BY-ND 4.0**.

### Decoder
- **Built only in pretraining mode.** With `num_classes == 0`, the model builds `decoder_head = PatchReconstructionHeadWithQueries` (a 1-layer TransformerDecoder plus an MLP to 40 samples) and `channel_emb` (L231–233).
- **Queries combine position and name.** In L304–307, `decoder_queries = channel_locations_emb + channel_emb(channel_names)` (a NeRF positional embedding of xyz plus a **learned per-name embedding**). The output is `(B, C, T)` for all channels.
- **Probably in the released file, but unverified.** The fine-tune loader explicitly filters `'decoder_head'` and `"channel_emb"` keys out of the released safetensors (`tasks/finetune_task_LUNA.py` L163), which strongly suggests they are present. I could not read the file header (HF blocked), and the 26.7 MB size cannot separate the two cases (the decoder is about 0.1M params).
- **Check first:** list the keys with `safetensors.safe_open`.

### Input normalisation
- **Per-channel, per-window z-score**, applied to the full window before the model (`tasks/pretrain_task_LUNA.py` L38–40 `mean = tensor.mean(dim=2…)`, `std = tensor.std(dim=2…)`, `(tensor - mean)/(std + eps)`; applied at L118).
- **Data:** µV, 256 Hz, 0.1–75 Hz, 60 Hz notch, 5 s slices (`make_datasets/make_tueg_bipolar.py` L202–203, L326–332; `process_raw_eeg.py` L165, L180–183).
- **Positions:** MNE `standard_1005` (`models/modules/channel_embeddings.py` L106–123). They are **min–max normalised per sample over the channels present** (`LUNA.py` L274–276).
- **Leak:**
  - The input statistics are per channel. For a hidden channel, the zero-filled input stays 0, and its patch tokens are fully replaced by `mask_token` (L273). So there is no cross-channel input leak.
  - The output is in the hidden channel's own z-units, which raises the same scale-leak issue for volts.
  - During pretraining, masked 1×40 patches were included in their channel's statistics, a mild within-channel leak. That does not matter for whole-channel hiding.

### Other caveats
- **Whole-channel masking was never part of training.** Masking picks 50% of [1 channel × 40 samples] rectangles at random (`pretrain_task_LUNA.py` L64–98), so a whole channel is essentially never masked.
- **Pretraining data:** TUEG bipolar (20 or 22 derivations, with positions at electrode midpoints) and Siena (29 unipolar channels). Source: `config/data_module/multiloader_data_module.yaml`, and the HF card says "TUEG + Siena".
- **The name vocabulary is SEED-62 ∪ TUEG-bipolar ∪ Siena-29** (L27–92).
  - Six EEGMMIDB names are **not in it**: AF7, AFz, AF8, T9, T10, Iz. They raise a `KeyError` in `get_channel_indices` (checked against the torcheeg 1.1.3 `SEED_CHANNEL_LIST`).
  - SEED was not in pretraining, so names that appear *only* in SEED (for example FC3, C1, CP4, POz, T7/T8/P7/P8 under those names) presumably have **untrained** embeddings. This is inferred from the config and is **unverified**.
  - Only about 23 EEGMMIDB names match Siena (Fp1/2, F3/4/7/8, Fz, FC1/2/5/6, C3/4, Cz, CP1/2/5/6, P3/4, Pz, O1/2). Four more work through the aliases T3/T4/T5/T6.
- **Random jitter at inference.** Whenever a mask is passed, `channel_locations + randn*0.02` is applied *even in eval* (L277–278). Fix the seed or patch this out.

### Requirements
| Item | Requirement |
|---|---|
| Sampling rate | 256 Hz |
| Length | Multiple of 40 samples (10 s = 64 patches; pretraining used 5 s, so 10 s relies on RoPE extrapolation — **unverified**) |
| Max channels | Any number |
| Positions | xyz plus channel-name indices |
| Referencing | Bipolar for TUEG; Siena reference **unverified** |

### Recipe
```python
# git clone --depth 1 https://github.com/pulp-bio/BioFoundation ; pip install timm einops mne torcheeg safetensors
import torch; from safetensors.torch import load_file; from huggingface_hub import hf_hub_download
from models.LUNA import LUNA; from models.modules.channel_embeddings import get_channel_indices, get_channel_locations
m = LUNA(patch_size=40, num_queries=4, embed_dim=64, depth=8, num_heads=2, num_classes=0)   # num_classes=0 -> decoder
sd = {k.removeprefix("model."): v for k, v in load_file(hf_hub_download("PulpBio/LUNA", "LUNA_base.safetensors")).items()}
miss, unexp = m.load_state_dict(sd, strict=False)
assert not [k for k in miss if "decoder_head" in k or "channel_emb" in k], "decoder not released"   # BLOCKER check
m = m.cuda().eval()
names = [n.upper() for n in names58]                  # drop AF7/AFz/AF8/T9/T10/Iz (not in vocab); alias T7->T3, T8->T4, P7->T5, P8->T6
x = torch.tensor(eeg_uV, dtype=torch.float32)[None]   # (1, 58, 2560) @256 Hz, 0.1-75 Hz, visible-avg ref, hidden rows = 0
x = (x - x.mean(2, keepdim=True)) / (x.std(2, keepdim=True) + 1e-8)
mask = torch.zeros_like(x, dtype=torch.bool); mask[:, hidden_idx, :] = True
locs = torch.tensor(get_channel_locations(names), dtype=torch.float32)[None]
idx = torch.tensor(get_channel_indices(names))[None]
torch.manual_seed(0)                                  # positional jitter is applied in eval too
with torch.no_grad(): rec, _, _ = m(x.cuda(), mask.cuda(), locs.cuda(), idx.cuda())          # (1, 58, 2560) z-units
```
**Possible blockers:**
- The decoder keys may be missing (check first).
- Only names with trained embeddings can sensibly be hidden.
- Whole-channel infill is out of distribution.

---

## 4. CBraMod (ICLR 2025)

- **Code:** https://github.com/wjq-learning/CBraMod.
- **Weights:** HF `weighting666/CBraMod`, a single file `pretrained_weights.pth` (19.8 MB; Apache-2.0). It is a pickle `.pth`, so load it with `weights_only=True`.

### Decoder
- **The reconstruction head is part of the model:** `proj_out = nn.Sequential(nn.Linear(d_model, out_dim))`, 200 → 200 samples per patch (`models/cbramod.py` L18–24).
- **It is almost certainly in the checkpoint.** `quick_example.py` loads the checkpoint with the default `strict=True` into a full `CBraMod()` that includes `proj_out`, and only then replaces it with `Identity`. A strict load would fail without those keys. I inferred this from the code and did not inspect the file.
- **Masked patches are set to zeros:** `mask_encoding = zeros`, not learnable (L43), applied as `mask_x[mask == 1] = self.mask_encoding` (L77). This is the same as our zero-filling.

### Input normalisation
- **Fixed scaling only:** µV divided by 100 (`pretrain_trainer.py` L64 `x = x.to(self.device)/100`; `datasets/physio_dataset.py` L33).
- **Pretraining samples were rejected if `max|x| ≥ 100 µV`** (`preprocessing/preprocessing_tueg_for_pretraining.py` L84). No clipping, no z-score.
- **TUEG preprocessing:** 19 fixed channels in a fixed order, original AR or LE reference, 200 Hz, 0.3–75 Hz, 60 Hz notch (L11–27, L65–67), 30 s samples.
- **Leak:** none from normalisation (nothing depends on the data). The output maps straight back to µV with ×100.

### Caveats (scientific rather than technical)
- **No electrode positions or names are used.** Spatial information comes only from a depthwise Conv2d "positional encoding" with kernel (19, 7) over the (channel index, patch) grid (L39–42).
  - Spatial neighbourhood therefore means *index order*, learned on the 19-channel TUEG order.
  - The model cannot know where a hidden electrode is, beyond its row index.
- **Whole-channel masking was never seen.** Masking is Bernoulli(0.5) per (channel, patch) (`utils/util.py` L11–14), so the chance of a whole 30-patch channel being masked is about 1e-9.
- **Montage mismatch:** trained on 19 channels; EEGMMIDB has 64.

### Requirements
| Item | Requirement |
|---|---|
| Sampling rate | 200 Hz |
| Patch | 200 samples (input shape `(B, C, n_patches, 200)`); 10 s = 10 patches |
| Max channels | Any number |
| Positions | None (channel order only) |
| Referencing | TUEG as recorded; average reference in their PhysioNet pipeline |

### Recipe
```python
# git clone --depth 1 https://github.com/wjq-learning/CBraMod ; pip install einops
import torch; from huggingface_hub import hf_hub_download; from models.cbramod import CBraMod
m = CBraMod()
m.load_state_dict(torch.load(hf_hub_download("weighting666/CBraMod", "pretrained_weights.pth"),
                             map_location="cpu", weights_only=True))       # strict -> proj_out must be present
m = m.cuda().eval()                                                         # eval(): spectral_proj has Dropout(0.1)
x = torch.tensor(eeg_uV, dtype=torch.float32) / 100                         # (64, 2000) @200 Hz, 0.3-75 Hz, visible-avg ref
x[hidden_idx] = 0
xp = x.reshape(1, 64, 10, 200).cuda()
mask = torch.zeros(1, 64, 10, dtype=torch.long, device="cuda"); mask[:, hidden_idx, :] = 1
with torch.no_grad(): rec_uV = m(xp, mask=mask).reshape(64, 2000) * 100     # hidden rows = infill in µV
```
**Blockers:** none technical. The scientific limits are the absence of geometry and the out-of-distribution channel count and masking pattern.

---

## Summary table

| Model | Raw-signal infill possible? | Decoder released? | Normalisation | Leak risk | Effort to run |
|---|---|---|---|---|---|
| **BrainOmni** (BrainTokenizer) | **Yes.** Trained exactly for it: 25% of channels dropped, decoder queried by 3-D position, arbitrary positions possible | **Yes, very likely** (`braintokenizer/BrainTokenizer.pt`; demo output shows working reconstruction; confirm no missing keys) | Average reference over **all** channels plus one global std per 10 s segment; target is per-channel, per-2 s z-score | **High in its native pipeline:** average reference and std computed before channel dropping, so the masked channels' sum is encoded in the visible ones. Volts need the hidden channel's own stats. Avoidable under our protocol, and testable. | **Low–medium:** about half a day; small model, 256 Hz native, stub `deepspeed.comm` |
| **REVE** | No, not with released weights | **No, encoder only** (69.2M ≈ 277 MB; `train.py` saves only the encoder) | Per-channel, per-session z-score, clip ±15 (pretraining); µV/100 (downstream) | Low for inputs; output-scale leak if de-normalised with the hidden channel's stats | Encoder: trivial. Infill: **blocked** (request MAE weights, or train our own decoder) |
| **LUNA** | Probably | **Probably** (fine-tune code strips `decoder_head`/`channel_emb` from the released file; **unverified**) | Per-channel, per-5 s-window z-score; positions min–max normalised per sample | Low for inputs; output-scale leak as above | **Medium:** verify keys; 6 names out of vocabulary; trained name embeddings only for about 23 EEGMMIDB names; never trained on whole-channel masking; eval-time jitter |
| **CBraMod** | Yes technically, but with no geometry input | **Yes** (`proj_out` is in the strict-loaded `.pth`) | Fixed µV/100; samples ≥100 µV dropped in pretraining | **None** from normalisation (output is in µV) | **Low** to run, but weak scientifically: channel order only, 19-channel TUEG training, whole-channel masking never seen |

## Recommendation

1. **BrainOmni BrainTokenizer: test next.** It is the closest analogue to our ZUNA setup.
   - It is a released, position-queried decoder trained specifically to reconstruct dropped channels.
   - It runs natively at 256 Hz, with 2 s windows that tile our 10 s recordings exactly.
   - It has its own ZUNA-like normalisation leak, the all-channel average reference, which we can measure directly by comparing an all-64 reference with a visible-only one.
   - Its "forward solution" decoder can be queried at virtual positions, which suits the physics probe.
   - Main caveat: the outputs are per-window z-scores, so use scale-free metrics or a leak-free amplitude calibration.
2. **Second model: LUNA if the decoder keys are present, otherwise CBraMod.**
   - The key check is a 5-minute `safe_open` on Kaggle.
   - LUNA adds a second position-aware decoder, but only for electrodes whose names have trained embeddings.
   - CBraMod is the cheapest to run and is useful as a **no-geometry control**: it cannot know electrode positions. Its 64-channel and whole-channel-masking inputs are out of distribution, though.
   - **REVE:** pursue only if the authors share the full MAE (decoder) checkpoint.

## Not verified / open items
- LUNA safetensors key list (HF downloads were blocked from this sandbox).
- Whether `OpenTSLab/BrainOmni` `BrainTokenizer.pt` loads with no missing keys (inferred from the demo output).
- Contents of `BrainOmni.pt`.
- Which normalisation (paper's per-channel or code's global) trained the released BrainOmni weights.
- Whether REVE pretraining excluded EEGMMIDB. The paper says downstream recordings were removed, and "Physionet" is listed as a pretraining source.
- Whether the SEED-only LUNA name embeddings are untrained.
- The braindecode REVE page (blocked).
- GitHub issues for all four repos (API and robots blocked).
