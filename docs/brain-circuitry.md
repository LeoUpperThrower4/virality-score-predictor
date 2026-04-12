# Brain Circuitry Behind Short-Video Virality

Synthesis of the neuroscience relevant to predicting Instagram Reels virality with TRIBE v2, grounded in two primary sources:

- **Gao et al. 2025** (*NeuroImage* 307, 121029): structural + functional + transcriptomic substrates of short-video addiction (SVA) in n=111.
- **Dores et al. 2025** (*Healthcare* 13, 89): PRISMA systematic review of fMRI/EEG studies on "like" feedback (12 studies, 537 participants).

Plus the TRIBE v2 repo (facebookresearch/tribev2) for what the model actually emits.

---

## 1. The circuits that fire during an engaging Reel

Two systems compete: a **reward/immersion** system that pulls the viewer in, and a **cognitive-control** system that would normally pull them out. Highly engaging (viral-prone) content wins the first and suppresses the second. Social feedback ("likes") then reinforces the loop.

### 1.1 Reward & motivation (pulls viewer in)

Gao et al. find SVA-positive structural (↑GMV) and functional (↑ReHo) changes in valuation cortex. Dores et al. identify the canonical social-reward network from 12 studies.

| Region | Evidence | Cortical? |
|---|---|---|
| Ventral Tegmental Area (VTA) | Activated by receiving many "likes" (Sherman, Hernandez et al.); activated by personalized TikTok clips (Su et al. 2021, cited in Gao) | ❌ subcortical |
| Nucleus Accumbens (NACC) | Activated by positive feedback; **NACC activation correlates with SM-use intensity** (Meshi 2013); activated by many "likes" (Sherman 2016) | ❌ subcortical |
| Ventral striatum / caudate / putamen | Activated by both social and monetary reward (Izuma 2008; Meshi 2013) | ❌ subcortical |
| Amygdala | Activated by positive feedback, especially from high-status raters (Davey 2010) | ❌ subcortical |
| Thalamus | Activated by positive "likes" and by personalized video (Sherman; Izuma); **↑ReHo in high-SVA individuals** (Gao) | ❌ subcortical |
| Orbitofrontal Cortex (OFC) | **↑GMV in OFC scales with SVA severity** (Gao 2025); activated by opposite-sex positive feedback (Davey) | ✅ cortical |
| Ventromedial PFC (vmPFC) | Midline reward/self-reference hub; activated by positive feedback (Davey; Wikman; Gunther) | ✅ cortical |
| Posterior Cingulate / precuneus (PCC) | Activated by positive feedback; **↑ReHo in SVA** reflects self-referential rumination (Gao; Wikman) | ✅ cortical |

**Takeaway:** most of the highest-signal reward structures are subcortical and invisible to TRIBE v2. The cortical proxies we *do* have (OFC, vmPFC, PCC) are well-established and correlate directly with SVA severity in Gao.

### 1.2 Cognitive control (the "brake")

This is the single most subtle and most commonly miscoded signal in virality prediction.

- **During active viewing of engaging short video**, DLPFC / IFG / MFG show **reduced task-evoked activation** (Su et al. 2023, cited in Gao — "prefrontal suppression in short-video viewing"). This deactivation *is* the signature of immersion.
- **At rest, in high-SVA individuals**, these same regions show **elevated ReHo** (over-synchronization), which Gao interprets as impaired ability to disengage — a trait-level biomarker of addiction vulnerability.

The two observations are not contradictory: task-evoked response goes DOWN while the individual watches; trait-level resting synchrony goes UP in chronic users. **For virality prediction from a single clip's TRIBE output, the relevant signal is task-evoked DLPFC suppression — i.e. lower DLPFC activation during the clip = more viral.**

Dores adds that **negative feedback** specifically recruits vlPFC and left mPFC (Wikman 2022) — a related inhibitory-control signature we can use as an anti-engagement marker.

### 1.3 Sensory, social & multimodal integration

| Region | Evidence |
|---|---|
| Visual cortex / fusiform | Activated for liked photos (Sherman); faces drive fusiform specifically |
| Auditory cortex (STG, Heschl's) | Music / voice / SFX drive engagement |
| **Cerebellum** | **↑GMV and ↑ReHo in SVA** (Gao); activated by "likes" (Sherman, Hernandez); processes rapid multimodal/social cues (Van Overwalle). **Not on fsaverage5.** |
| **Temporal Pole (TP)** | **↑ReHo in SVA; TP ReHo pattern mediates envy→SVA** (Gao). Social cognition, theory of mind, empathy. |
| Superior temporal gyrus | Feedback-sensitive (Wikman) |
| Insula (anterior/posterior) | Reward anticipation; activated by positive feedback (Wikman); negative feedback (vlPFC+insula) |

### 1.4 Default Mode Network (DMN) — mind wandering

Precuneus, medial PFC, angular gyrus, PCC. High DMN during viewing = drifting viewer. **But PCC is dual-coded:** in SVA the PCC is over-active as part of self-referential reward processing (Gao; Sherman self-photo studies), not just mind-wandering. Interpretation depends on whether PCC moves with or against the reward network.

### 1.5 EEG correlates (for completeness)

Dores reports: **P300 amplitude ↑** for socially relevant stimuli (attention allocation); **beta-wave dominance** when receiving many or few likes, alpha for moderate. TRIBE v2 predicts fMRI, not EEG, so these are out of scope — but relevant if we later add EEG-style temporal features.

---

## 2. What TRIBE v2 can and cannot see

- TRIBE v2 predicts fMRI on **fsaverage5 (~20k cortical vertices)**. Cortex only.
- **Subcortical reward structures (VTA, NAcc, ventral striatum, caudate, putamen, amygdala, thalamus) and the cerebellum are not in the output.** Based on Gao and Dores this is the single largest limitation: the strongest viral-engagement signals — NACC scaling with SM-use intensity, VTA/amygdala/thalamus activating for personalized video, cerebellum carrying multimodal/social integration — all live in regions the model does not emit.
- Predictions are shifted -5s for hemodynamic lag; time alignment with the stimulus is correct.
- Zero-shot canonical subject; it predicts the *average* brain, not a demographic-specific one.

**Implication:** we have to infer reward activity from cortical proxies (OFC, vmPFC, PCC, insula, midline self-reference regions) and from cortical-wide dynamics (DLPFC/IFG/MFG suppression during viewing).

---

## 3. Audit of the current code against the evidence

Reading `backend/brain_regions.py`:

### 3.1 Polarity errors and miscoded regions

1. **`reward_motivation` is actually a mix of DLPFC and OFC.** Lines 79–85 include `G_front_middle`, `G_front_sup`, `G_and_S_subcentral` — these are dorsolateral / superior frontal cortex, i.e. **cognitive control**, whose task-evoked *suppression* predicts engagement (Su 2023, Gao). They are currently assigned `polarity: +1`, so the pipeline rewards clips that *fail* to suppress the prefrontal brake. This inverts the sign on the most discriminative cortical signal.
2. **`narrative_language` includes `G_front_inf-Opercular` and `G_front_inf-Triangul` (IFG).** IFG is part of the inhibitory-control system whose suppression signals immersion. As +1 weighted it has the same inversion problem as item 1, partially offset because IFG is also genuinely language-related. This category double-counts.
3. **No vmPFC channel.** `G_front_med` only appears inside `default_mode_network` (line 92) with polarity −1. vmPFC is the single most consistently reward-positive midline region in Dores (Davey; Gunther; Sherman) — in the current code, vmPFC activation *lowers* the virality score.
4. **OFC mapping is incomplete.** `G_orbital`, `S_orbital_lateral`, `S_orbital-H_Shaped`, `G_rectus`, `G_subcallosal` are present — good — but there is no explicit `G_front_orbital` / medial OFC pattern, and they are buried in a category dominated by DLPFC patterns, so their signal is diluted.
5. **Temporal Pole (`Pole_temporal`) appears in both `social_processing` and `default_mode_network`** with opposite polarities. Same vertices get added then subtracted. Per Gao, TP is reward-positive in SVA and should be in the social/reward channel only.
6. **Angular gyrus (`G_pariet_inf-Angular`) appears in both `narrative_language` (+1) and `default_mode_network` (−1).** Same double-counting problem.
7. **Fusiform (`G_oc-temp_lat-fusifor`) appears in both `visual_attention` and `social_processing`.** Double-counts face-driven signal.

### 3.2 Missing channels

8. **No cognitive-control-suppression channel.** The whole signature of an engaging Reel — DLPFC/IFG/MFG deactivation — is not computed. Adding it (polarity −1) would be the single most impactful mapping change.
9. **No insula channel.** Dores: anterior insula for negative feedback / reward anticipation, posterior insula for positive feedback. Not mapped.
10. **No explicit midline self-reference / vmPFC channel.** vmPFC + medial PFC + subcallosal form the clearest cortical reward-valuation signal; currently fragmented.

### 3.3 Weighting and dynamics

11. **Weights are inverted relative to evidence.** Reward (0.14) < visual (0.18) (line 30 vs 86). Per Gao/Dores, reward-related cortex (OFC, vmPFC, PCC-reward component) is the *causal* virality driver; visual attention is necessary but not sufficient.
12. **Only mean per category.** `compute_region_activations` (line 141) reduces each region to mean + peak + min + std over time. No first-3-second hook, no peak-end, no rate-of-change, no cross-category synchrony. Gao's finding that DLPFC ReHo mediates envy→SVA means *synchronization* metrics (not just mean level) carry information the current pipeline discards.
13. **Normalization (`compute_virality_score` lines 211–216) per-category min-max within the clip destroys cross-clip comparability.** Every clip gets stretched to [0,1], so a flat boring clip and a dynamic viral clip can produce identical category means. This is the second most impactful issue after the polarity errors.
14. **No calibration against real outcomes.** Hand-weighted linear combo; never fit to shares / saves / completion rate.

### 3.4 Drop detection

15. **First-difference-on-smoothed-score misses slow DMN drift**, which Gao identifies as the dominant failure mode in mid-length clips. CUSUM or a matched filter against "DMN rising ∧ reward falling" would catch it.

---

## 4. Prioritized improvement plan

Ordered by expected lift per effort, now tied to evidence.

### Tier 1 — mapping fixes (hours)

**T1.1 Add `cognitive_control` channel, polarity −1.** Patterns: `G_front_middle`, `G_front_sup`, `G_front_inf-Opercular`, `G_front_inf-Triangul`, `S_front_sup`, `S_front_middle`, `S_front_inf`. Weight ~0.15. **Remove those patterns from `reward_motivation` and `narrative_language`.** Justification: Su et al. 2023 (prefrontal suppression during short-video viewing); Gao 2025 DLPFC ReHo as SVA mediator.

**T1.2 Create dedicated `valuation_vmpfc` channel, polarity +1.** Patterns: `G_front_med`, `G_subcallosal`, `G_rectus`, `S_suborbital`, `G_orbital`, `S_orbital_lateral`, `S_orbital-H_Shaped`, `S_orbital_med-olfact`. Weight ~0.18. Justification: Davey 2010, Gunther 2010, Sherman 2016, Gao 2025 (OFC ↑GMV in SVA).

**T1.3 Remove `G_front_med` from `default_mode_network`.** Keep DMN to precuneus + PCC-dorsal + angular (but see T1.4).

**T1.4 Deduplicate vertices.** Priority order: `valuation_vmpfc` > `reward_motivation` > `social_processing` > `visual_attention` > `auditory_engagement` > `narrative_language` > `default_mode_network` > `cognitive_control`. Build a vertex→category assignment so no vertex votes twice.

**T1.5 Add `insula` channel.** Patterns for anterior insula (`G_insular_short`, `S_circular_insula_ant`) and posterior insula (`G_Ins_lg_and_S_cent_ins`, `S_circular_insula_inf`, `S_circular_insula_sup`). Polarity +1 for posterior (reward anticipation — Wikman positive feedback), weight 0.06.

**T1.6 Rebalance weights:** valuation_vmpfc 0.18, reward_motivation 0.14, cognitive_control -0.15, visual 0.14, emotional 0.12, social 0.12, auditory 0.10, narrative 0.08, insula 0.06, DMN -0.08. Normalize so positives sum to 1.

### Tier 2 — dynamics and normalization (days)

**T2.1 Replace per-clip min-max with a fixed reference scale.** Fit per-category means/stds on a held-out reference corpus of ~100 mixed-quality Reels; z-score future clips against that. This alone should be a large accuracy win — currently two totally different clips can score identically.

**T2.2 Per-category feature vector instead of mean.** Extract: mean, peak, peak-time, first-3s mean (hook), last-3s mean (peak-end), |d/dt|, autocorrelation. Feed all into the composite.

**T2.3 Cross-category synchrony.** Sliding-window correlation between `valuation_vmpfc` and visual, and between `valuation_vmpfc` and auditory. Anti-correlation between valuation and `cognitive_control` as a dedicated feature (this is the neural signature of immersion per Su/Gao).

**T2.4 Better drop detection.** CUSUM on `overall_temporal`, plus a matched filter against the template "DMN + cognitive_control rising while valuation + reward falling." Replaces the current first-difference detector.

### Tier 3 — calibration against real outcomes (weeks; biggest accuracy win)

**T3.1 Label a dataset.** 1–5k Reels with public metrics. Shares and saves are the highest-signal labels (Dores: NACC tracks SM-use intensity, which correlates with share behavior).

**T3.2 Fit a supervised head.** Freeze TRIBE v2, aggregate T2 features per clip, train gradient-boosted regressor against labels. Replace hand-weighted composite with model prediction. Keep category breakdown for interpretability.

### Tier 4 — closing the subcortical gap

**T4.1 Subcortical proxy model.** Train a small regressor mapping the 20k cortical predictions → estimated NAcc / VTA / amygdala / thalamus activation using a whole-brain movie-watching dataset (HCP 7T movie, Neuromod Friends, or the Gao cohort if shareable). Even noisy subcortical estimates would directly surface the NACC/VTA signal TRIBE cannot emit — this is the ceiling the current approach can't cross without it.

**T4.2 Stimulus-side features.** Shot-change rate, loudness dynamic range, face coverage, on-screen text density, first-frame saliency, caption sentiment. Cheap, complementary to brain response.

---

## 5. Concrete next step

**Do Tier 1 and T2.1 this week.** Tier 1 fixes a sign error on the most discriminative cortical signal (cognitive-control suppression). T2.1 fixes a normalization bug that currently erases cross-clip differences. Together they are ~1 day of work and should visibly change the ranking of clips in the existing backend. Tier 3 is where the real accuracy lives, but it is gated on a labeled dataset.
