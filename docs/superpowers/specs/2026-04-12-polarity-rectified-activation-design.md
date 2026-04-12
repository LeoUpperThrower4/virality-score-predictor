# Polarity-Rectified Activation — Design

**Date:** 2026-04-12
**Scope:** Fix the "every video scores ~51" bug by changing the per-channel feature from raw `mean` (which averages signed z-scored TRIBE outputs to zero) to polarity-rectified activation.
**Non-goals:** Cross-clip percentile ranking (needs reference corpus — out of scope), global signal regression (not needed — TRIBE outputs are already z-scored per vertex), supervised calibration.

---

## 1. Root cause

TRIBE v2 emits z-scored BOLD: `neuralset.extractors.neuro.FmriCleaner` applies `standardize="zscore_sample"` during training, so predictions target z-scored signals. Output scale is mean ≈ 0, std ≈ 1 per vertex across time.

Current code: `channel.mean = np.mean(predictions[:, mask])` — averages a zero-centered distribution over ~56 seconds × N vertices. Result collapses to ≈ 0 for every clip. Weighted sum ≈ 0, sigmoid(0) = 0.5, score = 51.

## 2. Fix

Change the channel feature. Given per-second channel signal `x[t] = mean(predictions[t, mask])`:

```python
engagement = mean(max(0, polarity * x))
```

- Polarity +1: counts **above-baseline** activation (positive z deviations).
- Polarity −1: counts **below-baseline** activation (suppression).
- Result always in `[0, ~2]`. Interpretable as "how much did this channel do what we wanted it to do."

Polarity is now baked into the feature. Weights become all-positive (drop the negative signs in `IMMERSION_WEIGHTS` / `HOOK_WEIGHTS` / `PEAK_END_WEIGHTS`; magnitudes unchanged).

## 3. Affected features

Every feature currently computed as `np.mean(x)` or `np.mean(x[:3])` / `np.mean(x[-3:])` over `x` becomes:

```python
def rectified(x, polarity):
    return float(np.mean(np.maximum(0, polarity * np.asarray(x))))
```

Applied to:
- `mean` → stays in the output as `mean_raw` for debugging; scoring uses `engagement`.
- `hook_3s` → computed on `polarity * x[:3]` rectified.
- `peak_end_3s` → computed on `polarity * x[-3:]` rectified.
- Per-timestep temporal score: `max(0, polarity_k * x_k[t])` then weighted sum.

## 4. Composite weights

```python
IMMERSION_WEIGHTS = {  # all positive now; polarity already in feature
    "ofc_reward":       0.22,
    "vmpfc_valuation":  0.18,
    "pcc_self_reward":  0.10,
    "posterior_insula": 0.06,
    "dlpfc_control":    0.15,   # was -0.15
    "ifg_inhibition":   0.08,   # was -0.08
    "vlpfc_negative":   0.04,   # was -0.04
    "anterior_insula":  0.06,   # was -0.06
    "visual_cortex":    0.08,
    "auditory_cortex":  0.06,
    "fusiform_face":    0.06,
    "temporal_pole":    0.08,
    "stg_social":       0.04,
    "acc_mcc":          0.06,
    "parahippocampal":  0.03,
    "dmn_drift":        0.06,   # was -0.06
}

HOOK_WEIGHTS = {
    "ofc_reward":       0.20,
    "vmpfc_valuation":  0.15,
    "dlpfc_control":    0.15,   # was -0.15
    "visual_cortex":    0.10,
    "auditory_cortex":  0.08,
}

PEAK_END_WEIGHTS = {
    "vmpfc_valuation":  0.20,
    "pcc_self_reward":  0.18,
    "ofc_reward":       0.15,
    "fusiform_face":    0.10,
}
```

## 5. Sigmoid offset

With all-positive weights and features in `[0, ~2]`, `immersion_raw` sits in `[0, ~3]`. Shift the sigmoid so a "typical" clip scores near 50 rather than 70:

```python
OVERALL_OFFSET = 0.7      # subtract before final sigmoid
TEMPORAL_OFFSET = 0.3     # per-second; smaller because single-TR values are peakier

virality_raw = (
    SIGNATURE_WEIGHTS["immersion"] * immersion_raw
    + SIGNATURE_WEIGHTS["hook"]     * hook_raw
    + SIGNATURE_WEIGHTS["peak_end"] * peak_end_raw
)
overall = 100.0 * _sigmoid(virality_raw - OVERALL_OFFSET)
```

Offsets are heuristics; once a reference corpus exists they'll be replaced by proper calibration.

## 6. Channel output (for the UI)

Per channel:
```python
{
    "mean_raw": float,            # for debugging
    "engagement": float,          # polarity-rectified, new feature used by composite
    "hook_3s": float,             # rectified
    "peak_end_3s": float,         # rectified
    "temporal_activation": list,  # raw per-second x[t], unchanged (UI plots it)
    "polarity": int,
    "immersion_weight": float,    # unsigned
    "immersion_contribution": float,  # weight * engagement, always ≥ 0
    ...display_name, description, citation, n_vertices
}
```

## 7. Frontend

No structural change needed. `immersion_contribution` is now always ≥ 0 so the existing bar rendering already works. The "(lower is better)" label on polarity −1 channels stays — it now means "we measure the suppression of this channel; higher bar = more suppressed = better." Consider updating the label copy in a follow-up; out of scope here.

Bar scaling constant (`* 300` in `index.html`) should be re-tuned: with engagement typically 0.3–1.5 and weights 0.04–0.22, `immersion_contribution` typically 0.01–0.3. Scale factor `* 150` maps the expected range to 2–45 on a 100-wide bar. Adjust to `* 150`.

## 8. Risk

- **Offset values (0.7 and 0.3) are unvalidated.** They'll need re-tuning against a few real clips. The existing snapshot test scaffold is the right place to lock calibration later.
- **Rectification is lossy.** A channel that spikes positively then negatively gets only the positive half counted. For polarity +1 that's correct (we want *activation*, not dynamic range). For polarity −1 we only count suppression, not dysregulation. Both are defensible interpretations.
- **"Engagement" name collides with the project's general "engagement" nomenclature.** Accept the overlap; context makes it clear.

## 9. Implementation

Single-task, one commit:

1. In `backend/brain_regions.py`:
   - Replace `compute_channel_activations` with the polarity-rectified version.
   - Flip all negative weights in `IMMERSION_WEIGHTS`, `HOOK_WEIGHTS`, `PEAK_END_WEIGHTS` to positive.
   - In `compute_virality_score`: subtract `OVERALL_OFFSET` before the final sigmoid; read `engagement` / `hook_3s` / `peak_end_3s` from the new feature dict; per-timestep uses rectified values with `TEMPORAL_OFFSET`.
   - Surface `engagement` plus keep `mean_raw` for debug. `immersion_contribution = weight * engagement`.

2. In `frontend/index.html`:
   - Change bar scaling from `* 300` to `* 150`.

3. Commit and push.
