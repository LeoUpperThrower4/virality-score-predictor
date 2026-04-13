"""
TRIBE v2 integration for video analysis.

Wraps the TribeModel to process uploaded videos and extract
brain-response predictions used for virality scoring.
"""

import logging
import os
import subprocess
import time
from contextlib import nullcontext
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


def _downsample_video(src: Path, fps: int = 8, size: int = 256) -> Path:
    """Reduce fps and resize to `size`x`size` via ffmpeg; keep audio untouched.

    V-JEPA2 resizes to 256 internally and 30→8 fps cuts ~4x frames, so this
    trades a bit of temporal fidelity for a big speedup in video encoding.
    """
    dst = src.with_name(src.stem + f"_ds{fps}fps.mp4")
    vf = (
        f"fps={fps},"
        f"scale={size}:{size}:force_original_aspect_ratio=decrease,"
        f"pad={size}:{size}:(ow-iw)/2:(oh-ih)/2"
    )
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(src),
            "-vf", vf,
            "-c:a", "copy",
            str(dst),
        ],
        check=True,
    )
    return dst


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

    def analyze_video(self, video_path: str, downsample: bool = False) -> dict:
        """
        Run the full analysis pipeline on a video file.

        Args:
            video_path: Path to the video file (mp4, mov, etc.).
            downsample: If True, pre-process the video to 8 fps / 256×256 before
                TRIBE sees it. Roughly 2–4× faster video-feature extraction at
                the cost of some temporal fidelity.

        Returns:
            Complete analysis results with virality score, temporal data,
            engagement drops, and recommendations.
        """
        if not self._model_loaded:
            self.load_model()

        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        logger.info(f"Analyzing video: {video_path.name} (downsample={downsample})")

        downsampled_path: Path | None = None
        if downsample:
            start = time.time()
            downsampled_path = _downsample_video(video_path)
            logger.info(f"Downsampled to 8 fps / 256² in {time.time() - start:.1f}s")
            inference_path = downsampled_path
        else:
            inference_path = video_path

        # Autocast heavy encoder + transformer work to bf16 on CUDA — 4090/A100
        # tensor cores roughly double throughput with no observable accuracy loss
        # on vision transformers. Falls back to a no-op on CPU.
        try:
            import torch
            autocast_ctx = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if torch.cuda.is_available()
                else nullcontext()
            )
        except ImportError:
            autocast_ctx = nullcontext()

        try:
            # Step 1: Extract events (audio, text, video features). This runs
            # V-JEPA2 and w2v-bert and is 97% of wall time — autocast here is
            # where the speedup comes from.
            logger.info("Extracting multimodal events...")
            start = time.time()
            with autocast_ctx:
                events_df = self.model.get_events_dataframe(video_path=str(inference_path))
            # Autocast leaves extracted features as bfloat16 tensors in the
            # events dataframe; TRIBE's predict() ultimately calls .cpu().numpy()
            # which does not support bfloat16. Cast back to fp32 here so the
            # downstream transformer runs cleanly.
            try:
                import torch
                for col in events_df.columns:
                    for i, val in enumerate(events_df[col]):
                        if isinstance(val, torch.Tensor) and val.dtype == torch.bfloat16:
                            events_df.at[i, col] = val.to(torch.float32)
            except Exception:
                logger.exception("Failed to cast events_df bfloat16 tensors to fp32")
            logger.info(
                f"Events extracted in {time.time() - start:.1f}s — {len(events_df)} events"
            )

            # Step 2: Predict brain responses. Run in fp32 because TRIBE's
            # predict() calls .cpu().numpy() on the output tensor, and numpy
            # doesn't support BFloat16 scalars.
            logger.info("Running TRIBE v2 inference...")
            start = time.time()
            predictions, segments = self.model.predict(events=events_df)
            logger.info(
                f"Inference complete in {time.time() - start:.1f}s — "
                f"predictions shape: {predictions.shape}"
            )
        finally:
            if downsampled_path is not None:
                downsampled_path.unlink(missing_ok=True)

        # Step 3: Map predictions to brain channels
        logger.info("Computing brain channel activations...")
        channel_activations = self.region_mapper.compute_channel_activations(predictions)

        # Step 4: Compute virality score (immersion + hook + peak-end signatures)
        virality_data = compute_virality_score(channel_activations)

        # Step 5: Detect engagement drops
        drops = analyze_engagement_drops(
            virality_data["temporal_scores"],
            channel_activations,
        )

        # Step 6: Estimate video duration. TRIBE v2 predictions are at 1 TR (~1s)
        # and are already offset by -5s to compensate for the hemodynamic lag, so
        # the prediction count directly approximates the stimulus duration.
        video_duration = predictions.shape[0]

        # Step 7: Generate human-readable summary
        summary = generate_summary(virality_data, drops, video_duration)

        return {
            "virality_score": virality_data["overall_score"],
            "signatures": virality_data["signatures"],
            "temporal_scores": virality_data["temporal_scores"],
            "channels": virality_data["channels"],
            "engagement_drops": drops,
            "summary": summary,
            "video_duration_seconds": video_duration,
            "prediction_timesteps": predictions.shape[0],
            "n_vertices_analyzed": predictions.shape[1],
        }
