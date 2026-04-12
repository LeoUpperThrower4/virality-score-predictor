"""
TRIBE v2 integration for video analysis.

Wraps the TribeModel to process uploaded videos and extract
brain-response predictions used for virality scoring.
"""

import logging
import os
import time
from pathlib import Path

import numpy as np

from brain_regions import (
    BrainRegionMapper,
    analyze_engagement_drops,
    compute_virality_score,
    generate_summary,
)

logger = logging.getLogger(__name__)

# Required for large model downloads from HuggingFace
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "300")


class TribeAnalyzer:
    """Orchestrates video analysis using TRIBE v2 and brain region mapping."""

    def __init__(self, cache_folder: str = "./cache"):
        self.cache_folder = Path(cache_folder)
        self.cache_folder.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.region_mapper = BrainRegionMapper()
        self._model_loaded = False

    def load_model(self):
        """Load the TRIBE v2 model from HuggingFace. Call once at startup."""
        if self._model_loaded:
            return

        logger.info("Loading TRIBE v2 model from HuggingFace...")
        start = time.time()

        from tribev2.demo_utils import TribeModel

        self.model = TribeModel.from_pretrained(
            "facebook/tribev2",
            cache_folder=str(self.cache_folder),
        )
        self._model_loaded = True
        logger.info(f"TRIBE v2 model loaded in {time.time() - start:.1f}s")

        logger.info("Initializing brain region mapper...")
        self.region_mapper.initialize()
        logger.info("Brain region mapper ready.")

    def analyze_video(self, video_path: str) -> dict:
        """
        Run the full analysis pipeline on a video file.

        Args:
            video_path: Path to the video file (mp4, mov, etc.).

        Returns:
            Complete analysis results with virality score, temporal data,
            engagement drops, and recommendations.
        """
        if not self._model_loaded:
            self.load_model()

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        logger.info(f"Analyzing video: {video_path.name}")

        # Step 1: Extract events (audio, text, video features)
        logger.info("Extracting multimodal events...")
        start = time.time()
        events_df = self.model.get_events_dataframe(video_path=str(video_path))
        logger.info(f"Events extracted in {time.time() - start:.1f}s — {len(events_df)} events")

        # Step 2: Predict brain responses
        logger.info("Running TRIBE v2 inference...")
        start = time.time()
        predictions, segments = self.model.predict(events=events_df)
        logger.info(
            f"Inference complete in {time.time() - start:.1f}s — "
            f"predictions shape: {predictions.shape}"
        )

        # Step 3: Map predictions to brain regions
        logger.info("Computing brain region activations...")
        region_activations = self.region_mapper.compute_region_activations(predictions)

        # Step 4: Compute virality score
        virality_data = compute_virality_score(region_activations)

        # Step 5: Detect engagement drops
        drops = analyze_engagement_drops(
            virality_data["temporal_scores"],
            region_activations,
        )

        # Step 6: Estimate video duration. TRIBE v2 predictions are at 1 TR (~1s)
        # and are already offset by -5s to compensate for the hemodynamic lag, so
        # the prediction count directly approximates the stimulus duration.
        video_duration = predictions.shape[0]

        # Step 7: Generate human-readable summary
        summary = generate_summary(virality_data, drops, video_duration)

        return {
            "virality_score": virality_data["overall_score"],
            "temporal_scores": virality_data["temporal_scores"],
            "category_scores": virality_data["category_scores"],
            "engagement_drops": drops,
            "summary": summary,
            "video_duration_seconds": video_duration,
            "prediction_timesteps": predictions.shape[0],
            "n_vertices_analyzed": predictions.shape[1],
        }
