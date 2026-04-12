# Evidence-Matched Implementation Plan

Concrete architecture for a virality predictor whose every signal ties back to a specific finding in Gao 2025 (SVA) or Dores 2025 (likes review). Each feature below cites the paper result it operationalizes.

---

## Core design principle

**Every scalar the model emits must correspond to a published neural finding.** No fuzzy "engagement categories." Channels are defined as anatomical regions whose activation pattern was reported in the papers, with the polarity the papers establish.

---

## 1. Channel definitions (replaces `ENGAGEMENT_CATEGORIES`)

Each channel = (Destrieux pattern list, polarity, evidence citation).

### 1.1 Cortical reward / valuation — polarity +1

| Channel | Destrieux patterns | Evidence |
|---|---|---|
| `vmpfc_valuation` | `G_front_med`, `G_subcallosal`, `G_rectus`, `S_suborbital`, `S_orbital_med-olfact` | Davey 2010, Gunther 2010, Wikman 2022 (vmPFC activated by positive feedback); Gao 2025 (OFC valuation) |
| `ofc_reward` | `G_orbital`, `S_orbital_lateral`, `S_orbital-H_Shaped` | **Gao 2025 — ↑GMV in OFC scales directly with SVA severity (r=0.353)** |
| `pcc_self_reward` | `G_cingul-Post-dorsal`, `G_cingul-Post-ventral`, `S_pericallosal`, `G_precuneus` | Davey 2010, Wikman 2022 (PCC/precuneus for positive feedback); Sherman 2016 (self-photos with many likes → precuneus + PCC); Gao 2025 (↑ReHo PCC in SVA) |
| `posterior_insula` | `S_circular_insula_inf`, `G_Ins_lg_and_S_cent_ins` | Wikman 2022 (positive feedback → left posterior insula) |

### 1.2 Cortical cognitive control — polarity −1 (suppression = immersion)

| Channel | Destrieux patterns | Evidence |
|---|---|---|
| `dlpfc_control` | `G_front_middle`, `G_front_sup`, `S_front_sup`, `S_front_middle` | **Su 2023 (cited in Gao): "prefrontal suppression in short-video viewing"**; Gao 2025 (DLPFC ReHo mediates envy→SVA) |
| `ifg_inhibition` | `G_front_inf-Opercular`, `G_front_inf-Triangul`, `S_front_inf` | Su 2023 (IFG + MFG deactivation signature of immersion) |
| `vlpfc_negative` | `G_front_inf-Orbital`, `S_orbital_lateral` (lateral portion) | Wikman 2022 (vlPFC activates for *negative* feedback) — inverse signal |
| `anterior_insula` | `G_insular_short`, `S_circular_insula_ant`, `S_circular_insula_sup` | Wikman 2022 (anterior insula for negative feedback / conflict) |

### 1.3 Sensory / multimodal — polarity +1

| Channel | Destrieux patterns | Evidence |
|---|---|---|
| `visual_cortex` | `G_occipital_sup`, `G_occipital_middle`, `G_oc-temp_med-Lingual`, `S_calcarine`, `G_cuneus`, `Pole_occipital` | Sherman 2016 (lateral occipital for self-photos with likes); standard visual processing |
| `fusiform_face` | `G_oc-temp_lat-fusifor` | Face processing — dedicated channel (removed from visual double-count) |
| `auditory_cortex` | `G_temp_sup-G_T_transv`, `S_temporal_transverse`, `G_temp_sup-Plan_tempo`, `G_temp_sup-Lateral` | Standard auditory processing |

### 1.4 Social cognition — polarity +1

| Channel | Destrieux patterns | Evidence |
|---|---|---|
| `temporal_pole` | `Pole_temporal`, `G_temp_sup-Plan_polar` | **Gao 2025 — TP ReHo pattern mediates envy→SVA**; Olson 2007 (TP = social cognition / ToM) |
| `stg_social` | `G_temporal_middle`, `S_temporal_sup` (posterior portion) | Wikman 2022 (STG activates for peer feedback) |

### 1.5 Affective — polarity +1

| Channel | Destrieux patterns | Evidence |
|---|---|---|
| `acc_mcc` | `G_and_S_cingul-Ant`, `G_and_S_cingul-Mid-Ant`, `G_and_S_cingul-Mid-Post` | Davey 2010 (mid-cingulate for positive feedback); Somerville 2006 (dorsal/ventral ACC for feedback valence) |
| `parahippocampal` | `G_oc-temp_med-Parahip` | Gunther 2010 (parahippocampal for expectation violation) |

### 1.6 Default mode drift — polarity −1

| Channel | Destrieux patterns | Evidence |
|---|---|---|
| `dmn_drift` | `G_pariet_inf-Angular`, `G_parietal_sup` (medial portion), `S_parieto_occipital` | DMN when decoupled from reward — indicates mind-wandering |

**Dedup rule:** each vertex gets assigned to one channel only. Priority order:
`ofc_reward` > `vmpfc_valuation` > `pcc_self_reward` > `fusiform_face` > `temporal_pole` > `anterior_insula` > `posterior_insula` > `acc_mcc` > `dlpfc_control` > `ifg_inhibition` > `visual_cortex` > `auditory_cortex` > `stg_social` > `parahippocampal` > `vlpfc_negative` > `dmn_drift`.

---

## 2. Per-channel feature vector

For each channel's mean time series `x[t]` (length = clip seconds):

```python
features = {
    "mean":          np.mean(x),
    "peak":          np.max(x),
    "peak_time_frac": np.argmax(x) / len(x),       # where the peak lands (0=start, 1=end)
    "hook_3s":       np.mean(x[:3]),                # first-3-second mean
    "peak_end_3s":   np.mean(x[-3:]),               # last 3s (peak-end rule)
    "variability":   np.std(x),
    "dynamism":      np.mean(np.abs(np.diff(x))),   # rate of change
    "reho":          regional_homogeneity(x),       # Gao's key SVA biomarker
}
```

`regional_homogeneity` is Kendall's W over the channel's constituent vertex time-series within the clip — directly matches Gao 2025's ReHo measure.

---

## 3. Cross-channel features (the immersion signature)

These operationalize Su 2023's "prefrontal suppression" and Su 2021's DAN-FPN-Core pathways.

```python
def sync(x, y): return np.corrcoef(x, y)[0, 1]

cross = {
    # Su 2023: immersion = valuation high + DLPFC low
    "immersion_anticorr": -sync(vmpfc_valuation_ts, dlpfc_control_ts),

    # "Reward locked to stimulus" — viral clips chain reward onto sensory events
    "reward_visual_sync":  sync(vmpfc_valuation_ts, visual_cortex_ts),
    "reward_audio_sync":   sync(ofc_reward_ts, auditory_cortex_ts),

    # Su 2021: DAN-FPN-Core pathway. DMN coupled to reward = self-referential engagement
    "reward_pcc_sync":     sync(vmpfc_valuation_ts, pcc_self_reward_ts),

    # Social content driver (Gao: TP mediates envy→SVA)
    "tp_reward_sync":      sync(temporal_pole_ts, vmpfc_valuation_ts),

    # Anti-engagement: DMN decoupled from reward = drifting mind
    "dmn_drift_penalty":   -sync(dmn_drift_ts, vmpfc_valuation_ts),
}
```

---

## 4. Fixed-reference normalization (fixes current per-clip min-max bug)

On startup, load a reference corpus of ~100 mixed-quality Reels. For each feature, compute population mean + std across the corpus. At inference:

```python
z = (feature_value - ref_mean[feature]) / ref_std[feature]
```

This makes cross-clip scores comparable — a boring flat clip and a dynamic viral clip no longer get stretched to look the same.

Reference corpus bootstraps from any 100 real Reels run through TRIBE v2 once. Store `ref_stats.json` in the repo.

---

## 5. Composite scoring

### 5.1 Direct virality score (hand-weighted, evidence-grounded)

Weights assigned by evidence strength in the papers (highest-signal channels weighted most):

```python
IMMERSION_SIGNATURE = (
    # Cortical reward (Gao: OFC↑GMV is the strongest structural SVA marker)
    + 0.22 * z("ofc_reward",        "mean")
    + 0.18 * z("vmpfc_valuation",   "mean")
    + 0.10 * z("pcc_self_reward",   "mean")
    + 0.06 * z("posterior_insula",  "mean")

    # Control suppression (Su 2023 — THE immersion signature)
    - 0.15 * z("dlpfc_control",     "mean")
    - 0.08 * z("ifg_inhibition",    "mean")

    # Sensory drive
    + 0.08 * z("visual_cortex",     "mean")
    + 0.06 * z("auditory_cortex",   "mean")
    + 0.06 * z("fusiform_face",     "mean")

    # Social & affect
    + 0.08 * z("temporal_pole",     "mean")
    + 0.06 * z("acc_mcc",           "mean")

    # Anti-signals
    - 0.06 * z("anterior_insula",   "mean")     # conflict / negative feedback
    - 0.04 * z("vlpfc_negative",    "mean")
    - 0.06 * z("dmn_drift",         "mean")
)

DYNAMICS_SIGNATURE = (
    + 0.15 * cross["immersion_anticorr"]    # the core Su 2023 finding
    + 0.10 * cross["reward_visual_sync"]
    + 0.08 * cross["reward_audio_sync"]
    + 0.08 * cross["tp_reward_sync"]
    + 0.05 * cross["reward_pcc_sync"]
    + 0.06 * cross["dmn_drift_penalty"]
)

HOOK_SIGNATURE = (
    # First 3 seconds matter disproportionately on Reels
    + 0.20 * z("ofc_reward",        "hook_3s")
    + 0.15 * z("vmpfc_valuation",   "hook_3s")
    - 0.15 * z("dlpfc_control",     "hook_3s")
)

PEAK_END_SIGNATURE = (
    # Last 3 seconds drive the like decision (Dores: receive-a-like network)
    + 0.15 * z("vmpfc_valuation",   "peak_end_3s")
    + 0.15 * z("pcc_self_reward",   "peak_end_3s")
    + 0.10 * z("ofc_reward",        "peak_end_3s")
)

REHO_SIGNATURE = (
    # Gao 2025: ReHo in these regions is the SVA biomarker
    + 0.08 * z("dlpfc_control",     "reho")      # Gao: DLPFC ReHo mediates envy→SVA
    + 0.08 * z("temporal_pole",     "reho")      # Gao: TP ReHo mediates envy→SVA
    + 0.06 * z("pcc_self_reward",   "reho")
)

virality_raw = (
      0.35 * IMMERSION_SIGNATURE
    + 0.25 * DYNAMICS_SIGNATURE
    + 0.15 * HOOK_SIGNATURE
    + 0.15 * PEAK_END_SIGNATURE
    + 0.10 * REHO_SIGNATURE
)

# Map to 0-100 via sigmoid calibrated on reference corpus
virality_score = 100 * sigmoid(virality_raw)
```

### 5.2 Separate "like-ability" sub-score (strict Dores match)

Dores isolates a distinct "receiving a like" network (Sherman, Hernandez 2018): striatum + thalamus + VTA + mPFC + motor + occipital + cerebellum. Of these, only mPFC (via `vmpfc_valuation`) and occipital are cortical. We report this as a separate scalar so the UI can distinguish "engaging to watch" from "likely to get liked."

```python
like_probability_raw = (
      0.40 * z("vmpfc_valuation", "peak_end_3s")
    + 0.25 * z("pcc_self_reward", "peak_end_3s")
    + 0.15 * z("ofc_reward",      "peak_end_3s")
    + 0.10 * z("fusiform_face",   "mean")        # faces drive likes — Sherman 2016
    + 0.10 * cross["reward_pcc_sync"]             # self-relevance × reward
)
```

---

## 6. Subcortical proxy (crosses the fsaverage5 ceiling)

Strongest virality signals per both papers live in NACC, VTA, amygdala, thalamus, cerebellum — all absent from TRIBE output. Two-stage approach:

### 6.1 Stage-A (immediate, no training): cortical-proxy linear combination

Published frontostriatal/limbic structural connectivity gives us heuristic weights:

```python
# NACC proxy: weighted sum of cortical regions with strongest known NACC coupling
nacc_proxy = 0.5 * ts("vmpfc_valuation") + 0.3 * ts("ofc_reward") + 0.2 * ts("acc_mcc")

# Amygdala proxy: vmPFC + anterior temporal
amygdala_proxy = 0.6 * ts("vmpfc_valuation") + 0.4 * ts("temporal_pole")

# VTA/midbrain proxy: strongest coupling with OFC + ACC
vta_proxy = 0.5 * ts("ofc_reward") + 0.5 * ts("acc_mcc")
```

Add these as extra channels with weight ~0.05 each in the composite. Flagged as `proxy=True` so they can be swapped out later.

### 6.2 Stage-B (with training data): learned proxy

When a whole-brain movie-watching dataset is accessible (HCP 7T Movie, Neuromod Friends, Naturalistic Neuroimaging Database), train:

```
cortical_20k_vertex_timeseries → NACC/VTA/amygdala/thalamus ROI timeseries
```

Small 1D-CNN or ridge regression per subcortical ROI. Freeze, deploy alongside TRIBE. Replaces Stage-A heuristic proxies with learned ones. This is the single highest-ceiling improvement because it directly recovers the signal TRIBE architecturally cannot emit.

---

## 7. Supervised calibration (biggest accuracy lever)

Once 1–5k labeled Reels are available (shares / saves / completion rate):

```python
X = concat_per_clip([
    channel_features,       # ~16 channels × 8 features = 128
    cross_features,         # 6 cross-channel scalars
    subcortical_proxy_feats # ~8 scalars
])
y = log(1 + shares)   # or (shares + 3*saves) / impressions

model = GradientBoostingRegressor(max_depth=3, n_estimators=300)
model.fit(X_train, y_train)
```

The hand-weighted composite in §5 becomes the fallback / interpretability view; the boosted model becomes the headline score. The per-channel breakdown is still shown to users — the boosted model just replaces the final aggregation.

**Label strategy:** Dores establishes that NACC activation scales with SM-use intensity (Meshi 2013). Shares/saves are the closest public proxy for that driven behavior — they are higher-signal labels than likes, which are nearly free.

---

## 8. Envy-aware / audience-aware weighting (Gao mediation finding)

Gao 2025 shows envy → SVA is mediated specifically by TP and DLPFC features. For Reels targeting envy-driven audiences (aspirational, luxury, comparison-heavy content), the prediction improves if we upweight TP and DLPFC-suppression features.

Minimal version: a clip-level `audience_profile ∈ {general, aspirational, comedy, informational, ...}` tag (user-supplied or auto-classified from captions). Per-profile multiplier vector on the channel weights. Can be learned jointly with §7.

---

## 9. Drop detection (replaces current first-difference detector)

CUSUM on the per-second virality score, plus a matched-filter template for the characteristic failure mode Gao describes:

```
template = [+dmn_drift, +dlpfc_control, -vmpfc_valuation, -ofc_reward]
```

A window is flagged when the dot product of the per-second feature delta with this template exceeds a threshold. This catches the "cognitive brake re-engages, reward collapses" moment — which the current smoothed-diff detector misses on slow drifts.

---

## 10. Module layout

```
backend/
  brain_regions.py          # rewritten — channel defs + vertex dedup + ReHo
  features.py               # NEW — per-channel feature extraction
  cross_features.py         # NEW — synchrony, anti-correlation, DMN drift
  subcortical_proxy.py      # NEW — Stage-A linear proxies; Stage-B hook
  normalization.py          # NEW — fixed-reference z-scoring
  composite.py              # NEW — §5 weighted composite
  supervised_head.py        # NEW — sklearn/xgboost wrapper for §7
  drop_detection.py         # NEW — CUSUM + matched filter
  ref_stats.json            # NEW — reference corpus feature means/stds
  tribe_analyzer.py         # unchanged — still drives TRIBE inference
  main.py                   # updated response schema
```

---

## 11. Implementation order

**Week 1 — mapping + normalization (unlocks everything else):**
1. Rewrite `brain_regions.py` with the 16-channel schema from §1, vertex dedup.
2. Implement §2 feature extraction + §4 fixed-reference z-scoring.
3. Bootstrap `ref_stats.json` by running 100 Reels through the pipeline once.
4. Wire up §5 hand-weighted composite.
5. Smoke-test on existing test clips; confirm ranking changes sensibly.

**Week 2 — dynamics + subcortical proxy:**
6. Implement §3 cross-channel features.
7. Stage-A subcortical proxies (§6.1).
8. Replace drop detector with §9.
9. Update API response + frontend to show per-channel and per-signature breakdown.

**Weeks 3–4+ — calibration:**
10. Scrape labeled Reel dataset (shares, saves).
11. Train §7 supervised head.
12. A/B vs hand-weighted composite; ship whichever wins.

**Later — Stage-B subcortical:**
13. Acquire HCP/Neuromod movie-watching data.
14. Train learned subcortical proxies, swap out Stage-A.

---

## 12. Evidence traceability checklist

For any PR touching scoring, each new feature must carry an inline comment citing the paper result. Example:

```python
# Gao 2025 Fig 2: DLPFC ReHo mediates the envy→SVA link (indirect effect = 0.387).
# Higher resting ReHo = weaker disengagement = more immersive for envy-driven clips.
"dlpfc_reho": regional_homogeneity(dlpfc_control_ts),
```

If a feature can't be traced to a paper finding, it doesn't ship.
