"""
Virality Score Predictor — FastAPI Backend

Provides a REST API for uploading videos and receiving virality predictions
powered by Meta's TRIBE v2 brain-encoding model.
"""

import logging
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

# Load backend/.env before anything else reads os.environ (notably HF_TOKEN).
load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from tribe_analyzer import TribeAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
CACHE_DIR = Path(os.getenv("CACHE_DIR", "./cache"))
LOAD_MODEL_ON_STARTUP = os.getenv("LOAD_MODEL_ON_STARTUP", "true").lower() == "true"
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "500"))
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

analyzer = TribeAnalyzer(cache_folder=str(CACHE_DIR))


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle — optionally pre-loads the TRIBE v2 model."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    if LOAD_MODEL_ON_STARTUP:
        logger.info("Pre-loading TRIBE v2 model (set LOAD_MODEL_ON_STARTUP=false to skip)...")
        try:
            analyzer.load_model()
        except Exception as e:
            logger.warning(f"Model pre-load failed (will retry on first request): {e}")
    else:
        logger.info("Skipping model pre-load (will load on first request).")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Virality Score Predictor",
    description="Predict Instagram Reels virality using Meta's TRIBE v2 brain-encoding model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": analyzer._model_loaded,
    }


@app.post("/api/analyze")
async def analyze_video(file: UploadFile = File(...)):
    """
    Upload a video and receive a virality score with detailed analysis.

    The response includes:
    - **virality_score**: 0–100 composite score
    - **temporal_scores**: Per-second engagement scores
    - **category_scores**: Breakdown by brain engagement category
    - **engagement_drops**: Detected drops with causes and recommendations
    - **summary**: Human-readable analysis report
    """
    # Validate file extension
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    # Save uploaded file
    file_id = uuid.uuid4().hex[:12]
    upload_path = UPLOAD_DIR / f"{file_id}{ext}"

    try:
        file_size = 0
        with open(upload_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):  # 1 MB chunks
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE_MB * 1024 * 1024:
                    upload_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size: {MAX_FILE_SIZE_MB} MB.",
                    )
                buffer.write(chunk)

        logger.info(f"Uploaded {file.filename} ({file_size / 1024 / 1024:.1f} MB) → {upload_path.name}")

        # Run TRIBE v2 analysis
        results = analyzer.analyze_video(str(upload_path))

        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "file_size_mb": round(file_size / 1024 / 1024, 2),
            **results,
        })

    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception(f"Analysis failed for {file.filename}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        # Clean up uploaded file after processing
        upload_path.unlink(missing_ok=True)


# Serve frontend static files (production mode)
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )
