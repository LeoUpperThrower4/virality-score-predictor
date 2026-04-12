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

# ── Engagement category definitions ──────────────────────────────────────────
# Each category maps Destrieux atlas label substrings to a weight and polarity.
# Polarity: +1 = engagement driver, -1 = disengagement signal.

ENGAGEMENT_CATEGORIES = {
    "visual_attention": {
        "label_patterns": [
            "G_occipital", "S_calcarine", "G_cuneus", "G_lingual",
            "Pole_occipital", "S_oc_sup_and_transversal",
            "S_occipital_ant", "G_oc-temp_lat-fusifor",
        ],
        "weight": 0.18,
        "polarity": 1,
        "display_name": "Visual Attention",
        "description": "Visual cortex activation — how much the video captures visual attention.",
    },
    "auditory_engagement": {
        "label_patterns": [
            "G_temp_sup-G_T_transv", "S_temporal_transverse",
            "G_temp_sup-Plan_tempo", "G_temp_sup-Lateral",
            "S_temporal_sup",
        ],
        "weight": 0.14,
        "polarity": 1,
        "display_name": "Auditory Engagement",
        "description": "Auditory cortex activation — how engaging the audio/music/speech is.",
    },
    "emotional_response": {
        "label_patterns": [
            "G_cingul-Post-dorsal", "G_cingul-Post-ventral",
            "G_and_S_cingul-Ant", "G_and_S_cingul-Mid-Ant",
            "G_and_S_cingul-Mid-Post", "G_oc-temp_med-Parahip",
            "S_cingul-Marginalis",
        ],
        "weight": 0.18,
        "polarity": 1,
        "display_name": "Emotional Response",
        "description": "Cingulate and parahippocampal activation — emotional processing and memory encoding.",
    },
    "social_processing": {
        "label_patterns": [
            "G_oc-temp_lat-fusifor", "G_temp_sup-Plan_polar",
            "Pole_temporal", "G_temporal_middle",
            "S_temporal_inf",
        ],
        "weight": 0.14,
        "polarity": 1,
        "display_name": "Social Processing",
        "description": "Fusiform and temporal pole activation — face recognition and social cognition.",
    },
    "narrative_language": {
        "label_patterns": [
            "G_front_inf-Opercular", "G_front_inf-Triangul",
            "G_temp_sup-Lateral", "S_front_inf",
            "G_pariet_inf-Angular", "G_pariet_inf-Supramar",
        ],
        "weight": 0.14,
        "polarity": 1,
        "display_name": "Narrative & Language",
        "description": "Broca's area, Wernicke's area, and angular gyrus — narrative comprehension and language.",
    },
    "reward_motivation": {
        "label_patterns": [
            "G_front_middle", "G_front_sup",
            "G_and_S_subcentral", "G_rectus",
            "G_subcallosal", "S_orbital_lateral",
            "S_orbital-H_Shaped", "G_orbital",
        ],
        "weight": 0.14,
        "polarity": 1,
        "display_name": "Reward & Motivation",
        "description": "Prefrontal and orbitofrontal activation — reward anticipation and motivation.",
    },
    "default_mode_network": {
        "label_patterns": [
            "G_precuneus", "G_front_med",
            "G_pariet_inf-Angular", "Pole_temporal",
            "G_cingul-Post-dorsal",
        ],
        "weight": 0.08,
        "polarity": -1,
        "display_name": "Mind Wandering (DMN)",
        "description": "Default mode network — high activation indicates the viewer's mind is drifting.",
    },
}


class BrainRegionMapper:
    """Maps fsaverage5 vertex predictions to engagement-relevant brain regions."""

    def __init__(self):
        self.region_masks = {}
        self.labels = None
        self._initialized = False

    def initialize(self):
        """Load the Destrieux atlas and build vertex masks per engagement category."""
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
        label_names = [label.decode() if isinstance(label, bytes) else label for label in atlas["labels"]]

        all_labels = np.concatenate([labels_lh, labels_rh])

        for category_key, category_info in ENGAGEMENT_CATEGORIES.items():
            mask = np.zeros(len(all_labels), dtype=bool)
            for pattern in category_info["label_patterns"]:
                for idx, name in enumerate(label_names):
                    if pattern.lower() in name.lower():
                        mask |= (all_labels == idx)
            self.region_masks[category_key] = mask

        self.labels = all_labels
        self._initialized = True

    def compute_region_activations(self, predictions: np.ndarray) -> dict:
        """
        Compute mean activation per engagement category over time.

        Args:
            predictions: Shape (n_timesteps, n_vertices) from TRIBE v2.

        Returns:
            Dictionary with per-category temporal activation arrays and summary stats.
        """
        self.initialize()

        n_timesteps = predictions.shape[0]
        n_vertices_pred = predictions.shape[1]
        n_vertices_atlas = len(self.labels)

        # Handle vertex count mismatch by trimming or zero-padding
        if n_vertices_pred != n_vertices_atlas:
            min_v = min(n_vertices_pred, n_vertices_atlas)
            trimmed_masks = {}
            for key, mask in self.region_masks.items():
                trimmed_masks[key] = mask[:min_v]
            predictions = predictions[:, :min_v]
        else:
            trimmed_masks = self.region_masks

        results = {}
        for category_key, category_info in ENGAGEMENT_CATEGORIES.items():
            mask = trimmed_masks[category_key]
            if mask.sum() == 0:
                temporal_activation = np.zeros(n_timesteps)
            else:
                temporal_activation = predictions[:, mask].mean(axis=1)

            results[category_key] = {
                "temporal_activation": temporal_activation.tolist(),
                "mean_activation": float(np.mean(temporal_activation)),
                "peak_activation": float(np.max(temporal_activation)),
                "min_activation": float(np.min(temporal_activation)),
                "std_activation": float(np.std(temporal_activation)),
                "n_vertices": int(mask.sum()),
                "weight": category_info["weight"],
                "polarity": category_info["polarity"],
                "display_name": category_info["display_name"],
                "description": category_info["description"],
            }

        return results


def compute_virality_score(region_activations: dict) -> dict:
    """
    Compute a composite virality score from brain region activations.

    The score is a weighted combination of engagement-positive regions
    minus engagement-negative regions (default mode network), normalized
    to a 0–100 scale.

    Returns:
        Dictionary with overall score, category breakdown, and temporal scores.
    """
    weighted_temporal = None
    total_weight = 0
    category_scores = {}

    for key, data in region_activations.items():
        temporal = np.array(data["temporal_activation"])
        polarity = data["polarity"]
        weight = data["weight"]

        # Normalize each category's activation to 0-1 range
        t_min, t_max = temporal.min(), temporal.max()
        if t_max - t_min > 1e-8:
            normalized = (temporal - t_min) / (t_max - t_min)
        else:
            normalized = np.full_like(temporal, 0.5)

        # For negative polarity (DMN), invert so "score" means engagement:
        # high DMN activation = low engagement score.
        if polarity == -1:
            engagement = 1.0 - normalized
        else:
            engagement = normalized

        category_scores[key] = {
            "score": float(np.mean(engagement) * 100),
            "temporal_scores": (engagement * 100).tolist(),
            "display_name": data["display_name"],
            "description": data["description"],
            "weight": weight,
        }

        # All categories already point in the engagement direction after the
        # polarity flip above, so contributions are additive with positive weights.
        contribution = engagement * weight
        if weighted_temporal is None:
            weighted_temporal = contribution
        else:
            weighted_temporal += contribution
        total_weight += weight

    # Normalize by total weight to stay within [0, 1], then scale to 0-100.
    if total_weight > 0:
        weighted_temporal /= total_weight

    overall_temporal = np.clip(weighted_temporal * 100, 0, 100).tolist()
    overall_score = float(np.mean(overall_temporal))

    return {
        "overall_score": round(overall_score, 1),
        "temporal_scores": overall_temporal,
        "category_scores": category_scores,
    }


def analyze_engagement_drops(
    temporal_scores: list[float],
    region_activations: dict,
    window_size: int = 3,
    drop_threshold: float = 10.0,
) -> list[dict]:
    """
    Detect moments where engagement significantly drops and diagnose the cause.

    Args:
        temporal_scores: Per-second overall virality scores.
        region_activations: Per-category activation data.
        window_size: Rolling window for smoothing.
        drop_threshold: Minimum score decrease to flag as a drop.

    Returns:
        List of drop events with timing, severity, cause, and recommendations.
    """
    scores = np.array(temporal_scores)
    if len(scores) < window_size + 1:
        return []

    # Smooth scores with rolling average
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(scores, kernel, mode="valid")

    drops = []
    for i in range(1, len(smoothed)):
        delta = smoothed[i] - smoothed[i - 1]
        if delta < -drop_threshold:
            second = i + window_size // 2
            severity = abs(delta)

            # Diagnose which categories dropped most
            category_deltas = {}
            for key, data in region_activations.items():
                temporal = np.array(data["temporal_activation"])
                if second < len(temporal) and second > 0:
                    cat_delta = temporal[second] - temporal[second - 1]
                    category_deltas[key] = {
                        "delta": float(cat_delta),
                        "display_name": data["display_name"],
                    }

            # Sort by most negative delta to find the primary cause
            sorted_causes = sorted(category_deltas.items(), key=lambda x: x[1]["delta"])
            primary_cause = sorted_causes[0] if sorted_causes else None

            cause_key = primary_cause[0] if primary_cause else "unknown"
            recommendation = _get_recommendation(cause_key, second, severity)

            drops.append({
                "second": second,
                "severity": round(severity, 1),
                "score_before": round(float(smoothed[i - 1]), 1),
                "score_after": round(float(smoothed[i]), 1),
                "primary_cause": cause_key,
                "primary_cause_name": primary_cause[1]["display_name"] if primary_cause else "Unknown",
                "category_deltas": {
                    k: {"delta": round(v["delta"], 4), "name": v["display_name"]}
                    for k, v in sorted_causes[:3]
                },
                "recommendation": recommendation,
            })

    # Sort by severity and return top drops
    drops.sort(key=lambda x: x["severity"], reverse=True)
    return drops[:10]


def _get_recommendation(cause_key: str, second: int, severity: float) -> str:
    """Generate an actionable recommendation based on the engagement drop cause."""
    recommendations = {
        "visual_attention": (
            f"At second {second}, visual attention dropped significantly (severity: {severity:.1f}). "
            "Consider adding more visually dynamic elements here — a scene change, "
            "text overlay, zoom effect, or bright color contrast to recapture the viewer's eye."
        ),
        "auditory_engagement": (
            f"At second {second}, auditory engagement dropped (severity: {severity:.1f}). "
            "The audio may feel flat or repetitive at this point. Try adding a sound effect, "
            "music transition, voice tone change, or a brief silence followed by impact."
        ),
        "emotional_response": (
            f"At second {second}, emotional response weakened (severity: {severity:.1f}). "
            "The content may feel emotionally neutral here. Inject a surprise element, "
            "relatable moment, humor, or tension to re-engage the viewer emotionally."
        ),
        "social_processing": (
            f"At second {second}, social processing dropped (severity: {severity:.1f}). "
            "There may be fewer faces or human elements visible. Show a face, "
            "a reaction, or a person-to-person interaction to boost social engagement."
        ),
        "narrative_language": (
            f"At second {second}, narrative engagement dropped (severity: {severity:.1f}). "
            "The story or message may have stalled. Add a hook, rhetorical question, "
            "plot twist, or new piece of information to pull the viewer back in."
        ),
        "reward_motivation": (
            f"At second {second}, reward/motivation signaling dropped (severity: {severity:.1f}). "
            "The viewer may not feel anticipation. Tease an upcoming payoff, "
            "use a countdown, or create curiosity about what happens next."
        ),
        "default_mode_network": (
            f"At second {second}, mind-wandering signals spiked (severity: {severity:.1f}). "
            "The content became too predictable or slow. Break the pattern with an unexpected "
            "cut, a direct address to the viewer, or a rapid change in pacing."
        ),
    }
    return recommendations.get(
        cause_key,
        f"At second {second}, overall engagement dropped (severity: {severity:.1f}). "
        "Review the content at this timestamp and consider adding more dynamic elements.",
    )


def generate_summary(
    virality_data: dict,
    drops: list[dict],
    video_duration: float,
) -> str:
    """Generate a human-readable summary of the virality analysis."""
    score = virality_data["overall_score"]
    categories = virality_data["category_scores"]

    # Tier classification
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
        "### Category Breakdown",
        "",
    ]

    sorted_cats = sorted(categories.items(), key=lambda x: x[1]["score"], reverse=True)
    for key, cat in sorted_cats:
        lines.append(f"- **{cat['display_name']}**: {cat['score']:.1f}/100 — {cat['description']}")

    if drops:
        lines.append("")
        lines.append("### Key Engagement Drops")
        lines.append("")
        for i, drop in enumerate(drops[:5], 1):
            lines.append(f"**Drop {i} — Second {drop['second']}** (severity: {drop['severity']})")
            lines.append(f"Score went from {drop['score_before']} → {drop['score_after']}")
            lines.append(f"Primary cause: {drop['primary_cause_name']}")
            lines.append(f"→ {drop['recommendation']}")
            lines.append("")
    else:
        lines.append("")
        lines.append("### Engagement")
        lines.append("")
        lines.append("No significant engagement drops detected — the content maintains consistent viewer attention.")

    return "\n".join(lines)
