# Evidence-Matched Brain Region Mapping — Design

**Date:** 2026-04-12
**Scope:** Tier A of `docs/implementation-plan.md` — rewrite `backend/brain_regions.py` to ground every virality signal in a finding from Gao 2025 or Dores 2025. Fix the per-clip min-max normalization bug. Add hook and peak-end features per channel.
**Non-goals:** ReHo, cross-channel synchrony, subcortical proxies, supervised calibration, reference corpus, audience-aware reweighting, new drop detector.

---

## 1. Motivation

Two concrete bugs in the current `brain_regions.py`:

1. **Polarity error.** `reward_motivation` includes DLPFC/superior-frontal patterns (`G_front_middle`, `G_front_sup`) with `polarity: +1`. Per Su 2023 (cited in Gao 2025), these regions **deactivate** during immersive short-video viewing — their suppression is the signature of engagement. We currently reward their activation, inverting the sign on the single most discriminative cortical signal.
2. **Per-clip min-max stretch.** `compute_virality_score` (lines 211–216) rescales every channel's temporal range to `[0, 1]` per clip. A flat boring clip and a dynamic viral clip can therefore produce identical category means — absolute differences are erased.

Plus structural issues: no cognitive-control channel, vmPFC miscoded as DMN, fusiform / angular / temporal pole vertices double-counted across categories with opposite polarities, reward under-weighted relative to visual attention.

## 2. Architecture

Single-file rewrite of `backend/brain_regions.py`, matching response-schema changes in `backend/main.py`, matching display changes in `frontend/index.html`.

```
backend/brain_regions.py   rewrite  — channel schema, vertex dedup, features, composite
backend/main.py            update   — response schema reflects new channel keys + signatures
frontend/index.html        update   — render new channel names with polarity-aware styling
```

No new modules. No new dependencies.

## 3. Channel schema

16 channels. Each channel has a `patterns` list (Destrieux substrings), a `polarity` (+1 or −1), a `weight`, and a `citation` field that ties the channel to a paper finding.

### 3.1 Reward / valuation (polarity +1)

| Channel | Patterns | Citation |
|---|---|---|
| `ofc_reward` | `G_orbital`, `S_orbital_lateral`, `S_orbital-H_Shaped` | Gao 2025 — ↑GMV in OFC scales with SVA severity (r=0.353, p<0.001) |
| `vmpfc_valuation` | `G_front_med`, `G_subcallosal`, `G_rectus`, `S_suborbital`, `S_orbital_med-olfact` | Davey 2010, Gunther 2010, Wikman 2022 — vmPFC activated by positive feedback |
| `pcc_self_reward` | `G_cingul-Post-dorsal`, `G_cingul-Post-ventral`, `S_pericallosal`, `G_precuneus` | Sherman 2016 (self-photos with many likes); Gao 2025 (↑ReHo PCC in SVA) |
| `posterior_insula` | `S_circular_insula_inf`, `G_Ins_lg_and_S_cent_ins` | Wikman 2022 — left posterior insula for positive feedback |

### 3.2 Cognitive control (polarity −1 — suppression = immersion)

| Channel | Patterns | Citation |
|---|---|---|
| `dlpfc_control` | `G_front_middle`, `G_front_sup`, `S_front_sup`, `S_front_middle` | Su 2023 — "prefrontal suppression in short-video viewing"; Gao 2025 (DLPFC ReHo mediates envy→SVA) |
| `ifg_inhibition` | `G_front_inf-Opercular`, `G_front_inf-Triangul`, `S_front_inf` | Su 2023 — IFG/MFG deactivation signature of immersion |
| `vlpfc_negative` | `G_front_inf-Orbital` | Wikman 2022 — vlPFC activates for negative feedback |
| `anterior_insula` | `G_insular_short`, `S_circular_insula_ant`, `S_circular_insula_sup` | Wikman 2022 — anterior insula for negative feedback / conflict |

### 3.3 Sensory (polarity +1)

| Channel | Patterns | Citation |
|---|---|---|
| `visual_cortex` | `G_occipital_sup`, `G_occipital_middle`, `G_oc-temp_med-Lingual`, `S_calcarine`, `G_cuneus`, `Pole_occipital` | Sherman 2016 — lateral occipital activation for self-photos with likes |
| `fusiform_face` | `G_oc-temp_lat-fusifor` | Face processing — standard |
| `auditory_cortex` | `G_temp_sup-G_T_transv`, `S_temporal_transverse`, `G_temp_sup-Plan_tempo`, `G_temp_sup-Lateral` | Standard auditory processing |

### 3.4 Social cognition (polarity +1)

| Channel | Patterns | Citation |
|---|---|---|
| `temporal_pole` | `Pole_temporal`, `G_temp_sup-Plan_polar` | Gao 2025 — TP ReHo pattern mediates envy→SVA; Olson 2007 (TP = social cognition / ToM) |
| `stg_social` | `G_temporal_middle`, `S_temporal_sup` | Wikman 2022 — STG activates for peer feedback |

### 3.5 Affective (polarity +1)

| Channel | Patterns | Citation |
|---|---|---|
| `acc_mcc` | `G_and_S_cingul-Ant`, `G_and_S_cingul-Mid-Ant`, `G_and_S_cingul-Mid-Post` | Davey 2010 (mid-cingulate for positive feedback); Somerville 2006 (ACC for feedback valence) |
| `parahippocampal` | `G_oc-temp_med-Parahip` | Gunther 2010 — parahippocampal for expectation violation |

### 3.6 Default mode drift (polarity −1)

| Channel | Patterns | Citation |
|---|---|---|
| `dmn_drift` | `G_pariet_inf-Angular`, `S_parieto_occipital` | Mind-wandering signal when decoupled from reward |

### 3.7 Vertex deduplication

Each fsaverage5 vertex gets assigned to at most one channel. Priority order (winner takes the vertex):

```
ofc_reward > vmpfc_valuation > pcc_self_reward > fusiform_face
  > temporal_pole > anterior_insula > posterior_insula > acc_mcc
  > dlpfc_control > ifg_inhibition > visual_cortex > auditory_cortex
  > stg_social > parahippocampal > vlpfc_negative > dmn_drift
```

Resolves current double-counts: `G_oc-temp_lat-fusifor` (was in visual + social), `G_pariet_inf-Angular` (narrative + DMN), `Pole_temporal` (social + DMN), `G_front_med` (previously only DMN, now primarily vmPFC).

## 4. Per-channel features

For each channel's mean time series `x[t]` (length = clip seconds, one value per second from TRIBE output aggregated across the channel's vertex mask):

```python
features = {
    "mean":        float(np.mean(x)),
    "hook_3s":     float(np.mean(x[:3])) if len(x) >= 3 else float(np.mean(x)),
    "peak_end_3s": float(np.mean(x[-3:])) if len(x) >= 3 else float(np.mean(x)),
}
```

Clips shorter than 3 seconds fall back to `mean` for hook and peak-end.

## 5. Composite score

Three signatures, each a weighted sum over channels with the channel's polarity baked in.

### 5.1 Immersion signature (uses all 16 channels)

```
IMMERSION =
    + 0.22 * ofc_reward.mean
    + 0.18 * vmpfc_valuation.mean
    + 0.10 * pcc_self_reward.mean
    + 0.06 * posterior_insula.mean
    - 0.15 * dlpfc_control.mean
    - 0.08 * ifg_inhibition.mean
    - 0.04 * vlpfc_negative.mean
    - 0.06 * anterior_insula.mean
    + 0.08 * visual_cortex.mean
    + 0.06 * auditory_cortex.mean
    + 0.06 * fusiform_face.mean
    + 0.08 * temporal_pole.mean
    + 0.04 * stg_social.mean
    + 0.06 * acc_mcc.mean
    + 0.03 * parahippocampal.mean
    - 0.06 * dmn_drift.mean
```

Weights are sign-aware: the negative constants on `dlpfc_control` etc. encode the polarity directly in the formula, no separate polarity flip needed in code.

### 5.2 Hook signature (top reward + control channels only)

```
HOOK =
    + 0.20 * ofc_reward.hook_3s
    + 0.15 * vmpfc_valuation.hook_3s
    - 0.15 * dlpfc_control.hook_3s
    + 0.10 * visual_cortex.hook_3s
    + 0.08 * auditory_cortex.hook_3s
```

### 5.3 Peak-end signature (drives like decision)

```
PEAK_END =
    + 0.20 * vmpfc_valuation.peak_end_3s
    + 0.18 * pcc_self_reward.peak_end_3s
    + 0.15 * ofc_reward.peak_end_3s
    + 0.10 * fusiform_face.peak_end_3s
```

### 5.4 Aggregation

```
virality_raw   = 0.60 * IMMERSION + 0.20 * HOOK + 0.20 * PEAK_END
virality_score = 100 * sigmoid(virality_raw)
```

**No per-clip min-max rescaling.** Raw TRIBE BOLD-prediction means feed directly into the weighted sum. TRIBE v2 has fixed model weights so its output scale is consistent across clips.

### 5.5 Temporal scores for the UI

Per-second overall engagement, used by the drop detector and plotted in the UI:

```
virality_temporal[t] = 100 * sigmoid(IMMERSION_at_time_t)
```

Where `IMMERSION_at_time_t` uses the same weighted sum as §5.1 but on the per-second channel activations instead of means. Hook and peak-end are clip-level scalars and don't contribute to per-second scoring.

## 6. API response schema

Breaking changes to `/api/analyze`:

```jsonc
{
  "success": true,
  "virality_score": 72.3,
  "signatures": {
    "immersion": 0.84,
    "hook": 0.45,
    "peak_end": 0.62
  },
  "temporal_scores": [65.1, 68.4, ...],
  "channels": {
    "ofc_reward": {
      "display_name": "OFC Reward",
      "description": "...",
      "polarity": 1,
      "weight": 0.22,
      "mean": 0.71,
      "hook_3s": 0.65,
      "peak_end_3s": 0.78,
      "temporal_activation": [0.62, 0.70, ...],
      "n_vertices": 412,
      "citation": "Gao 2025 — ↑GMV in OFC scales with SVA severity"
    },
    "dlpfc_control": { ..., "polarity": -1, ... },
    ...
  },
  "engagement_drops": [...],   // unchanged
  "summary": "..."             // unchanged structure, new channel names in body
}
```

Old `category_scores` key is **removed**. Tool is internal, no back-compat shim.

## 7. Frontend changes

`frontend/index.html` renders `channels` instead of `category_scores`. Polarity-aware display:

- `polarity: +1` channels — existing positive styling (higher = better).
- `polarity: −1` channels — show with "lower is better" indicator; render the contribution to the score as `−weight × value` so the sign is visible.

New top-level row shows the three signature sub-scores (immersion / hook / peak-end) so the user can see *why* the virality score came out where it did.

No new dependencies. No styling refactor — just label and data-binding updates.

## 8. Module structure inside `brain_regions.py`

```python
CHANNELS: dict[str, ChannelSpec]     # 16 entries, each with patterns/polarity/weight/citation/display_name/description

DEDUP_PRIORITY: list[str]            # priority order from §3.7

IMMERSION_WEIGHTS: dict[str, float]  # §5.1 — keys are channel names, values signed
HOOK_WEIGHTS: dict[str, float]       # §5.2
PEAK_END_WEIGHTS: dict[str, float]   # §5.3

SIGNATURE_WEIGHTS = {"immersion": 0.60, "hook": 0.20, "peak_end": 0.20}

class BrainRegionMapper:
    def initialize(): ...
    def _build_dedup_masks(): ...    # new — priority-ordered vertex assignment
    def compute_channel_activations(predictions): ...   # renamed; returns per-channel time series + mean/hook/peak_end

def compute_virality_score(channel_activations) -> dict:
    # No per-clip min-max. Apply IMMERSION/HOOK/PEAK_END weights to raw means/hook/peak-end.
    # Return overall score, signatures, per-channel features.

def analyze_engagement_drops(temporal_scores, channel_activations, ...) -> list[dict]:
    # Unchanged algorithm (first-difference on smoothed). Updated to read per-channel activations
    # instead of region_activations. Recommendation strings updated for new channel names where
    # the mapping has changed.

def generate_summary(virality_data, drops, video_duration) -> str:
    # Rewritten to show signature sub-scores and per-channel breakdown with citations.
```

The public surface consumed by `main.py` keeps the same function names (`BrainRegionMapper`, `compute_virality_score`, `analyze_engagement_drops`, `generate_summary`), just with updated signatures and return dicts.

## 9. Testing

**One regression snapshot test.**

- Run one representative sample video (checked into `backend/tests/fixtures/sample.mp4` or pulled from an existing dev asset) through the full pipeline.
- Capture the `/api/analyze` response JSON; commit as `backend/tests/fixtures/sample_expected.json`.
- Test: run pipeline again, diff against snapshot. Non-zero diff = intentional change requires updating the snapshot in the same PR.

No unit tests. Tool is internal; the snapshot catches regressions on the only thing that ultimately matters (the response the frontend consumes).

## 10. Out of scope

Explicitly **not** built in this work:

- ReHo / Kendall's W as a feature
- Cross-channel synchrony (immersion-anticorrelation, reward↔sensory sync)
- Subcortical proxies (NAcc / VTA / amygdala from cortical combinations)
- Supervised head (gradient boosting against shares/saves)
- Labeled Reel dataset collection
- Reference corpus / z-scoring / percentile ranking
- Audience-aware reweighting (Gao envy-mediation finding)
- CUSUM or matched-filter drop detector
- Variability / dynamism / autocorrelation per-channel features

Each of these has a tier in `docs/implementation-plan.md`. This spec covers Tier A only.

## 11. Risks

- **Weights are hand-tuned, not fit.** The composite might under- or over-weight a channel on some clip genres. Mitigation: the regression snapshot catches drastic changes; actual calibration is Tier C.
- **Destrieux labels may not match on all nilearn versions.** Current code already depends on `nilearn.datasets.fetch_atlas_surf_destrieux`; the new patterns use the same label vocabulary, so same risk profile as today.
- **Vertex dedup priority order is a judgment call.** E.g. fusiform vertices going to `fusiform_face` over `visual_cortex` is defensible but other orderings are possible. Documented in §3.7; reversible.
