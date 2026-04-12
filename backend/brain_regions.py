"""
Brain region mapping for fsaverage5 cortical mesh.

Maps TRIBE v2 vertex predictions to engagement-relevant brain regions
using the Destrieux atlas parcellation. Each region contributes to a
composite virality/engagement score.
"""

import numpy as np

try:
    from nilearn import datasets, surface
    NILEARN_AVAILABLE = True
except ImportError:
    NILEARN_AVAILABLE = False


def _sigmoid(x: float) -> float:
    """Numerically stable logistic sigmoid."""
    if x >= 0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    z = np.exp(x)
    return float(z / (1.0 + z))


# ── Evidence-grounded channel schema ─────────────────────────────────────────
# Each channel maps Destrieux atlas label substrings to a weight and polarity.
# Polarity: +1 = engagement driver, -1 = disengagement / suppression signal.
# The `citation` field ties every channel back to a paper finding — see
# docs/superpowers/specs/2026-04-12-evidence-matched-brain-region-mapping-design.md

CHANNELS = {
    # ── Reward / valuation (polarity +1) ────────────────────────────────────
    "ofc_reward": {
        "patterns": ["G_orbital", "S_orbital_lateral", "S_orbital-H_Shaped"],
        "polarity": 1,
        "display_name": "OFC Reward",
        "description": "Orbitofrontal cortex — reward valuation and immediate gratification.",
        "citation": "Gao 2025 — ↑GMV in OFC scales with SVA severity (r=0.353).",
    },
    "vmpfc_valuation": {
        "patterns": [
            "G_front_med", "G_subcallosal", "G_rectus",
            "S_suborbital", "S_orbital_med-olfact",
        ],
        "polarity": 1,
        "display_name": "vmPFC Valuation",
        "description": "Ventromedial PFC — subjective value and self-relevance.",
        "citation": "Davey 2010, Gunther 2010, Wikman 2022 — vmPFC activated by positive feedback.",
    },
    "pcc_self_reward": {
        "patterns": [
            "G_cingul-Post-dorsal", "G_cingul-Post-ventral",
            "S_pericallosal", "G_precuneus",
        ],
        "polarity": 1,
        "display_name": "PCC Self-Reward",
        "description": "Posterior cingulate / precuneus — self-referential reward processing.",
        "citation": "Sherman 2016 (self-photos w/ many likes); Gao 2025 (↑ReHo PCC in SVA).",
    },
    "posterior_insula": {
        "patterns": ["S_circular_insula_inf", "G_Ins_lg_and_S_cent_ins"],
        "polarity": 1,
        "display_name": "Posterior Insula",
        "description": "Posterior insula — reward anticipation from positive feedback.",
        "citation": "Wikman 2022 — left posterior insula for positive feedback.",
    },

    # ── Cognitive control (polarity -1: suppression = immersion) ────────────
    "dlpfc_control": {
        "patterns": ["G_front_middle", "G_front_sup", "S_front_sup", "S_front_middle"],
        "polarity": -1,
        "display_name": "DLPFC Control",
        "description": "Dorsolateral PFC — cognitive control. Lower activation = more immersed.",
        "citation": "Su 2023 — prefrontal suppression during short-video viewing; Gao 2025 (DLPFC ReHo mediates envy→SVA).",
    },
    "ifg_inhibition": {
        "patterns": ["G_front_inf-Opercular", "G_front_inf-Triangul", "S_front_inf"],
        "polarity": -1,
        "display_name": "IFG Inhibition",
        "description": "Inferior frontal gyrus — response inhibition. Lower = more immersed.",
        "citation": "Su 2023 — IFG/MFG deactivation signature of short-video immersion.",
    },
    "vlpfc_negative": {
        "patterns": ["G_front_inf-Orbital"],
        "polarity": -1,
        "display_name": "vlPFC (Negative)",
        "description": "Ventrolateral PFC — activates for negative feedback / conflict.",
        "citation": "Wikman 2022 — vlPFC activates for negative feedback.",
    },
    "anterior_insula": {
        "patterns": ["G_insular_short", "S_circular_insula_ant", "S_circular_insula_sup"],
        "polarity": -1,
        "display_name": "Anterior Insula",
        "description": "Anterior insula — negative feedback and conflict signal.",
        "citation": "Wikman 2022 — anterior insula for negative feedback.",
    },

    # ── Sensory (polarity +1) ───────────────────────────────────────────────
    "visual_cortex": {
        "patterns": [
            "G_occipital_sup", "G_occipital_middle", "G_oc-temp_med-Lingual",
            "S_calcarine", "G_cuneus", "Pole_occipital",
        ],
        "polarity": 1,
        "display_name": "Visual Cortex",
        "description": "Occipital cortex — visual processing and scene parsing.",
        "citation": "Sherman 2016 — lateral occipital for self-photos with likes.",
    },
    "fusiform_face": {
        "patterns": ["G_oc-temp_lat-fusifor"],
        "polarity": 1,
        "display_name": "Fusiform (Faces)",
        "description": "Fusiform face area — face detection and recognition.",
        "citation": "Kanwisher 1997 (FFA); faces are primary engagement drivers.",
    },
    "auditory_cortex": {
        "patterns": [
            "G_temp_sup-G_T_transv", "S_temporal_transverse",
            "G_temp_sup-Plan_tempo", "G_temp_sup-Lateral",
        ],
        "polarity": 1,
        "display_name": "Auditory Cortex",
        "description": "Auditory cortex — music, voice, and sound effect processing.",
        "citation": "Standard auditory processing (STG / Heschl's).",
    },

    # ── Social cognition (polarity +1) ──────────────────────────────────────
    "temporal_pole": {
        "patterns": ["Pole_temporal", "G_temp_sup-Plan_polar"],
        "polarity": 1,
        "display_name": "Temporal Pole",
        "description": "Temporal pole — social cognition, theory of mind, empathy.",
        "citation": "Gao 2025 — TP ReHo pattern mediates envy→SVA; Olson 2007 (TP social cognition).",
    },
    "stg_social": {
        "patterns": ["G_temporal_middle", "S_temporal_sup"],
        "polarity": 1,
        "display_name": "STG (Social)",
        "description": "Superior temporal gyrus — peer feedback and social stimuli.",
        "citation": "Wikman 2022 — STG activates for peer feedback.",
    },

    # ── Affective (polarity +1) ─────────────────────────────────────────────
    "acc_mcc": {
        "patterns": [
            "G_and_S_cingul-Ant", "G_and_S_cingul-Mid-Ant",
            "G_and_S_cingul-Mid-Post",
        ],
        "polarity": 1,
        "display_name": "ACC / MCC",
        "description": "Anterior and mid cingulate — feedback valence and emotional salience.",
        "citation": "Davey 2010 (mid-cingulate for positive feedback); Somerville 2006 (ACC valence).",
    },
    "parahippocampal": {
        "patterns": ["G_oc-temp_med-Parahip"],
        "polarity": 1,
        "display_name": "Parahippocampal",
        "description": "Parahippocampal gyrus — expectation violation / memory encoding.",
        "citation": "Gunther 2010 — parahippocampal for expectation violation.",
    },

    # ── Default mode drift (polarity -1) ────────────────────────────────────
    "dmn_drift": {
        "patterns": ["G_pariet_inf-Angular", "S_parieto_occipital"],
        "polarity": -1,
        "display_name": "DMN Drift",
        "description": "Default mode network — high activation = mind-wandering.",
        "citation": "Standard DMN; inverted when decoupled from reward regions.",
    },
}

# Priority order for vertex deduplication. Earlier channels claim vertices
# first; later channels only get vertices not already claimed.
DEDUP_PRIORITY = [
    "ofc_reward", "vmpfc_valuation", "pcc_self_reward", "fusiform_face",
    "temporal_pole", "anterior_insula", "posterior_insula", "acc_mcc",
    "dlpfc_control", "ifg_inhibition", "visual_cortex", "auditory_cortex",
    "stg_social", "parahippocampal", "vlpfc_negative", "dmn_drift",
]

# Signed weights applied to each channel's feature in the composite signatures.
# Negative weights on cognitive-control channels encode polarity directly.

IMMERSION_WEIGHTS = {
    "ofc_reward":       +0.22,
    "vmpfc_valuation":  +0.18,
    "pcc_self_reward":  +0.10,
    "posterior_insula": +0.06,
    "dlpfc_control":    -0.15,
    "ifg_inhibition":   -0.08,
    "vlpfc_negative":   -0.04,
    "anterior_insula":  -0.06,
    "visual_cortex":    +0.08,
    "auditory_cortex":  +0.06,
    "fusiform_face":    +0.06,
    "temporal_pole":    +0.08,
    "stg_social":       +0.04,
    "acc_mcc":          +0.06,
    "parahippocampal":  +0.03,
    "dmn_drift":        -0.06,
}

HOOK_WEIGHTS = {
    "ofc_reward":       +0.20,
    "vmpfc_valuation":  +0.15,
    "dlpfc_control":    -0.15,
    "visual_cortex":    +0.10,
    "auditory_cortex":  +0.08,
}

PEAK_END_WEIGHTS = {
    "vmpfc_valuation":  +0.20,
    "pcc_self_reward":  +0.18,
    "ofc_reward":       +0.15,
    "fusiform_face":    +0.10,
}

SIGNATURE_WEIGHTS = {"immersion": 0.60, "hook": 0.20, "peak_end": 0.20}


class BrainRegionMapper:
    """Maps fsaverage5 vertex predictions to evidence-grounded channels.

    Vertices are assigned to exactly one channel via DEDUP_PRIORITY so that
    overlapping patterns (e.g. fusiform appearing in both visual and social
    regions in the atlas) don't double-count in the composite score.
    """

    def __init__(self):
        self.channel_masks: dict[str, np.ndarray] = {}
        self.all_labels: np.ndarray | None = None
        self._initialized = False

    def initialize(self):
        if self._initialized:
            return

        if not NILEARN_AVAILABLE:
            raise RuntimeError(
                "nilearn is required for brain region mapping. "
                "Install with: pip install nilearn"
            )

        atlas = datasets.fetch_atlas_surf_destrieux()
        labels_lh = atlas["map_left"]
        labels_rh = atlas["map_right"]
        label_names = [
            label.decode() if isinstance(label, bytes) else label
            for label in atlas["labels"]
        ]
        self.all_labels = np.concatenate([labels_lh, labels_rh])

        # Build a raw pattern-match mask per channel (before dedup).
        raw_masks: dict[str, np.ndarray] = {}
        for channel_key, channel_info in CHANNELS.items():
            mask = np.zeros(len(self.all_labels), dtype=bool)
            for pattern in channel_info["patterns"]:
                for idx, name in enumerate(label_names):
                    if pattern.lower() in name.lower():
                        mask |= (self.all_labels == idx)
            raw_masks[channel_key] = mask

        # Apply priority order: each vertex goes to the first channel in
        # DEDUP_PRIORITY whose raw mask includes it.
        claimed = np.zeros(len(self.all_labels), dtype=bool)
        for channel_key in DEDUP_PRIORITY:
            mask = raw_masks[channel_key] & ~claimed
            self.channel_masks[channel_key] = mask
            claimed |= mask

        self._initialized = True

    def compute_channel_activations(self, predictions: np.ndarray) -> dict:
        """Compute per-channel temporal activation and summary features.

        Args:
            predictions: Shape (n_timesteps, n_vertices) from TRIBE v2.

        Returns:
            Dict keyed by channel name. Each value has:
              - temporal_activation: list[float] per-second channel mean
              - mean, hook_3s, peak_end_3s: scalar features
              - n_vertices, polarity, display_name, description, citation
        """
        self.initialize()

        n_timesteps, n_vertices_pred = predictions.shape
        n_vertices_atlas = len(self.all_labels)

        # Handle vertex count mismatch by trimming both sides to the min.
        if n_vertices_pred != n_vertices_atlas:
            min_v = min(n_vertices_pred, n_vertices_atlas)
            trimmed_masks = {k: m[:min_v] for k, m in self.channel_masks.items()}
            predictions = predictions[:, :min_v]
        else:
            trimmed_masks = self.channel_masks

        results = {}
        for channel_key, channel_info in CHANNELS.items():
            mask = trimmed_masks[channel_key]
            if mask.sum() == 0:
                temporal = np.zeros(n_timesteps)
            else:
                temporal = predictions[:, mask].mean(axis=1)

            if n_timesteps >= 3:
                hook = float(np.mean(temporal[:3]))
                peak_end = float(np.mean(temporal[-3:]))
            else:
                hook = peak_end = float(np.mean(temporal))

            results[channel_key] = {
                "temporal_activation": temporal.tolist(),
                "mean": float(np.mean(temporal)),
                "hook_3s": hook,
                "peak_end_3s": peak_end,
                "n_vertices": int(mask.sum()),
                "polarity": channel_info["polarity"],
                "display_name": channel_info["display_name"],
                "description": channel_info["description"],
                "citation": channel_info["citation"],
            }

        return results


def compute_virality_score(channel_activations: dict) -> dict:
    """Compute composite virality from channel activations.

    Three signatures — immersion (full), hook (first 3s), peak-end (last 3s) —
    are each a signed weighted sum of channel features. Raw TRIBE outputs feed
    directly into the sums; no per-clip min-max rescaling.

    Returns:
        Dict with overall_score (0-100), signatures, temporal_scores,
        and channels (feature dump for the UI).
    """
    def weighted_sum(weights: dict, feature: str) -> float:
        total = 0.0
        for channel_key, w in weights.items():
            if channel_key in channel_activations:
                total += w * channel_activations[channel_key][feature]
        return total

    immersion_raw = weighted_sum(IMMERSION_WEIGHTS, "mean")
    hook_raw      = weighted_sum(HOOK_WEIGHTS,      "hook_3s")
    peak_end_raw  = weighted_sum(PEAK_END_WEIGHTS,  "peak_end_3s")

    # Per-second immersion for the temporal score / drop detector.
    any_channel = next(iter(channel_activations.values()))
    n_timesteps = len(any_channel["temporal_activation"])
    temporal_raw = np.zeros(n_timesteps)
    for channel_key, w in IMMERSION_WEIGHTS.items():
        if channel_key in channel_activations:
            ts = np.array(channel_activations[channel_key]["temporal_activation"])
            temporal_raw += w * ts

    temporal_scores = [100.0 * _sigmoid(float(x)) for x in temporal_raw]

    virality_raw = (
        SIGNATURE_WEIGHTS["immersion"] * immersion_raw
        + SIGNATURE_WEIGHTS["hook"]     * hook_raw
        + SIGNATURE_WEIGHTS["peak_end"] * peak_end_raw
    )
    overall_score = 100.0 * _sigmoid(virality_raw)

    # Pass through per-channel data for the UI, annotated with each channel's
    # signed contribution to the immersion signature (makes the composite
    # interpretable to the viewer).
    channels_out = {}
    for channel_key, data in channel_activations.items():
        immersion_weight = IMMERSION_WEIGHTS.get(channel_key, 0.0)
        channels_out[channel_key] = {
            **data,
            "immersion_weight": immersion_weight,
            "immersion_contribution": immersion_weight * data["mean"],
        }

    return {
        "overall_score": round(overall_score, 1),
        "signatures": {
            "immersion": round(float(_sigmoid(immersion_raw)), 3),
            "hook":      round(float(_sigmoid(hook_raw)),      3),
            "peak_end":  round(float(_sigmoid(peak_end_raw)),  3),
        },
        "temporal_scores": temporal_scores,
        "channels": channels_out,
    }


def analyze_engagement_drops(
    temporal_scores: list[float],
    channel_activations: dict,
    window_size: int = 3,
    drop_threshold: float = 10.0,
) -> list[dict]:
    """Detect seconds where engagement drops and diagnose the likely cause.

    A drop's "cause" is the channel whose signed change best explains the
    drop — e.g. a spike in dlpfc_control (polarity -1) is as much a cause
    as a dip in ofc_reward (polarity +1).
    """
    scores = np.array(temporal_scores)
    if len(scores) < window_size + 1:
        return []

    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(scores, kernel, mode="valid")

    drops = []
    for i in range(1, len(smoothed)):
        delta = smoothed[i] - smoothed[i - 1]
        if delta >= -drop_threshold:
            continue

        second = i + window_size // 2
        severity = abs(delta)

        # Score each channel's contribution to the drop: a positive-polarity
        # channel *dropping* is bad; a negative-polarity channel *rising* is
        # also bad. Use -polarity*cat_delta so "more negative = worse."
        channel_scores = {}
        for key, data in channel_activations.items():
            temporal = np.array(data["temporal_activation"])
            if 0 < second < len(temporal):
                cat_delta = float(temporal[second] - temporal[second - 1])
                badness = -data["polarity"] * cat_delta
                channel_scores[key] = {
                    "delta": cat_delta,
                    "badness": badness,
                    "display_name": data["display_name"],
                }

        sorted_causes = sorted(
            channel_scores.items(),
            key=lambda x: x[1]["badness"],
            reverse=True,
        )
        primary_cause_key = sorted_causes[0][0] if sorted_causes else "unknown"
        primary_cause_name = (
            sorted_causes[0][1]["display_name"] if sorted_causes else "Unknown"
        )
        recommendation = _get_recommendation(primary_cause_key, second, severity)

        drops.append({
            "second": second,
            "severity": round(severity, 1),
            "score_before": round(float(smoothed[i - 1]), 1),
            "score_after": round(float(smoothed[i]), 1),
            "primary_cause": primary_cause_key,
            "primary_cause_name": primary_cause_name,
            "channel_deltas": {
                k: {"delta": round(v["delta"], 4), "name": v["display_name"]}
                for k, v in sorted_causes[:3]
            },
            "recommendation": recommendation,
        })

    drops.sort(key=lambda x: x["severity"], reverse=True)
    return drops[:10]


def _get_recommendation(cause_key: str, second: int, severity: float) -> str:
    recommendations = {
        "ofc_reward": (
            f"At second {second}, orbitofrontal reward signaling dropped "
            f"(severity: {severity:.1f}). The viewer stopped finding the content "
            "immediately gratifying. Inject a payoff, punchline, or visual reward."
        ),
        "vmpfc_valuation": (
            f"At second {second}, vmPFC valuation weakened (severity: {severity:.1f}). "
            "The content stopped feeling personally relevant. Add a relatable moment, "
            "direct address, or self-referential callback."
        ),
        "pcc_self_reward": (
            f"At second {second}, self-referential reward processing dropped "
            f"(severity: {severity:.1f}). The viewer disengaged from personal relevance. "
            "Bring the focus back to the viewer's experience or identity."
        ),
        "posterior_insula": (
            f"At second {second}, reward-anticipation signaling dropped "
            f"(severity: {severity:.1f}). Tease what's coming next — a countdown, "
            "a reveal, or a question."
        ),
        "dlpfc_control": (
            f"At second {second}, the cognitive-control brake re-engaged "
            f"(severity: {severity:.1f}) — the viewer regained self-control and may "
            "scroll away. Break the pattern: unexpected cut, loud SFX, or surprise."
        ),
        "ifg_inhibition": (
            f"At second {second}, inhibitory control rose (severity: {severity:.1f}). "
            "Content became skippable. Add novelty or a jarring beat change."
        ),
        "vlpfc_negative": (
            f"At second {second}, negative-feedback signaling spiked "
            f"(severity: {severity:.1f}). The content felt off-putting. Review for "
            "tonal mismatch, awkward pacing, or conflict without resolution."
        ),
        "anterior_insula": (
            f"At second {second}, conflict/aversion signaling rose "
            f"(severity: {severity:.1f}). Something felt wrong to the viewer — check "
            "for jarring transitions or confusing cuts."
        ),
        "visual_cortex": (
            f"At second {second}, visual engagement dropped (severity: {severity:.1f}). "
            "Add a scene change, text overlay, zoom, or high-contrast visual."
        ),
        "fusiform_face": (
            f"At second {second}, face-processing activation dropped "
            f"(severity: {severity:.1f}). Bring a face back on screen — faces are "
            "primary engagement drivers."
        ),
        "auditory_cortex": (
            f"At second {second}, auditory engagement dropped (severity: {severity:.1f}). "
            "Add a sound effect, music swell, voice change, or impactful silence."
        ),
        "temporal_pole": (
            f"At second {second}, social-cognition engagement dropped "
            f"(severity: {severity:.1f}). Add a social cue — a reaction, eye contact, "
            "or interpersonal moment."
        ),
        "stg_social": (
            f"At second {second}, social-feedback processing dropped "
            f"(severity: {severity:.1f}). Reintroduce a peer/crowd element or "
            "social reaction."
        ),
        "acc_mcc": (
            f"At second {second}, emotional salience dropped (severity: {severity:.1f}). "
            "Inject a surprise, a twist, or a change in emotional register."
        ),
        "parahippocampal": (
            f"At second {second}, expectation/memory engagement dropped "
            f"(severity: {severity:.1f}). Subvert an expectation or call back a "
            "detail from earlier."
        ),
        "dmn_drift": (
            f"At second {second}, mind-wandering signals spiked "
            f"(severity: {severity:.1f}). Content became too predictable. Break the "
            "pattern with pacing change or direct address."
        ),
    }
    return recommendations.get(
        cause_key,
        f"At second {second}, overall engagement dropped (severity: {severity:.1f}). "
        "Review the content at this timestamp and consider adding dynamic elements.",
    )


def generate_summary(
    virality_data: dict,
    drops: list[dict],
    video_duration: float,
) -> str:
    """Human-readable summary showing signature sub-scores and channel breakdown."""
    score = virality_data["overall_score"]
    sigs = virality_data["signatures"]
    channels = virality_data["channels"]

    if score >= 80:
        tier = "Excellent — High viral potential"
    elif score >= 60:
        tier = "Good — Moderate viral potential"
    elif score >= 40:
        tier = "Average — Limited viral potential without optimization"
    elif score >= 20:
        tier = "Below Average — Significant improvements needed"
    else:
        tier = "Low — Major rework recommended"

    lines = [
        f"## Virality Score: {score}/100 — {tier}",
        "",
        f"Video duration analyzed: {video_duration:.0f} seconds",
        "",
        "### Signature Breakdown",
        "",
        f"- **Immersion** (overall): {sigs['immersion']:.2f}",
        f"- **Hook** (first 3s):    {sigs['hook']:.2f}",
        f"- **Peak-End** (last 3s): {sigs['peak_end']:.2f}",
        "",
        "### Top Engagement Drivers",
        "",
    ]

    # Sort channels by absolute immersion contribution (positive or negative).
    sorted_channels = sorted(
        channels.items(),
        key=lambda kv: abs(kv[1]["immersion_contribution"]),
        reverse=True,
    )
    for key, ch in sorted_channels[:8]:
        sign = "+" if ch["immersion_contribution"] >= 0 else "−"
        polarity_note = " (lower is better)" if ch["polarity"] == -1 else ""
        lines.append(
            f"- **{ch['display_name']}**{polarity_note}: "
            f"contribution {sign}{abs(ch['immersion_contribution']):.3f} "
            f"— {ch['description']}"
        )

    if drops:
        lines.extend(["", "### Key Engagement Drops", ""])
        for i, drop in enumerate(drops[:5], 1):
            lines.append(f"**Drop {i} — Second {drop['second']}** (severity: {drop['severity']})")
            lines.append(f"Score went from {drop['score_before']} → {drop['score_after']}")
            lines.append(f"Primary cause: {drop['primary_cause_name']}")
            lines.append(f"→ {drop['recommendation']}")
            lines.append("")
    else:
        lines.extend([
            "",
            "### Engagement",
            "",
            "No significant engagement drops detected — the content maintains consistent viewer attention.",
        ])

    return "\n".join(lines)
