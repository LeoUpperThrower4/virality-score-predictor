# Virality Score Predictor

Predict Instagram Reels virality using Meta's **TRIBE v2** brain-encoding model. Upload a video, and the app returns a **0–100 virality score** with a detailed breakdown of when viewer engagement drops, why it happened, and what to do about it.

## How It Works

TRIBE v2 predicts how the human brain responds to audiovisual content by modeling fMRI activity across ~20,000 cortical vertices. This app maps those brain predictions to six engagement categories:

| Category | What It Measures |
|---|---|
| Visual Attention | Visual cortex activation — how much the video captures the eye |
| Auditory Engagement | Auditory cortex — how engaging the sound/music/speech is |
| Emotional Response | Cingulate cortex — emotional processing and memory encoding |
| Social Processing | Fusiform/temporal — face recognition and social cognition |
| Narrative & Language | Broca's/Wernicke's — story comprehension and language |
| Reward & Motivation | Prefrontal cortex — reward anticipation and motivation |
| Mind Wandering (DMN) | Default mode network — inverted: high = viewer drifting |

The composite score is a weighted combination of these categories, with mind wandering acting as a negative signal.

## Quick Start

### Option A: Local Setup

```bash
# 1. Clone and run setup
./setup.sh

# 2. Set your HuggingFace token
export HF_TOKEN=your_token_here

# 3. Start the server
source .venv/bin/activate
cd backend && python main.py

# 4. Open http://localhost:8000
```

### Option B: Docker

```bash
# Set your HuggingFace token
export HF_TOKEN=your_token_here

# Build and run (GPU)
docker compose up --build

# For CPU-only, remove the 'deploy.resources' block from docker-compose.yml
```

## Prerequisites

- **Python 3.10+**
- **ffmpeg** (for audio/video processing)
- **HuggingFace token** with access to LLaMA 3.2 (required by TRIBE v2)
  - Create a token at https://huggingface.co/settings/tokens
  - Accept the LLaMA 3.2 license at https://huggingface.co/meta-llama/Llama-3.2-3B
- **GPU recommended** (CUDA) — CPU works but inference is slower

## Configuration

Copy `.env.example` to `.env` in the `backend/` folder:

| Variable | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `HF_TOKEN` | — | HuggingFace API token |
| `LOAD_MODEL_ON_STARTUP` | `true` | Pre-load model at server start |
| `MAX_FILE_SIZE_MB` | `500` | Maximum upload size |
| `CACHE_DIR` | `./cache` | Model weights cache |

## Project Structure

```
├── backend/
│   ├── main.py              # FastAPI server + file upload
│   ├── tribe_analyzer.py    # TRIBE v2 inference pipeline
│   ├── brain_regions.py     # Brain region mapping + scoring
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   └── index.html           # Single-page React app
├── Dockerfile
├── docker-compose.yml
└── setup.sh
```

## API

### `POST /api/analyze`

Upload a video file (multipart form data) and receive the full analysis.

**Response:**
```json
{
  "success": true,
  "virality_score": 72.3,
  "temporal_scores": [65.1, 68.4, ...],
  "category_scores": {
    "visual_attention": { "score": 78.2, "display_name": "Visual Attention", ... },
    ...
  },
  "engagement_drops": [
    {
      "second": 12,
      "severity": 18.3,
      "primary_cause_name": "Auditory Engagement",
      "recommendation": "..."
    }
  ],
  "summary": "## Virality Score: 72.3/100 — ..."
}
```

### `GET /api/health`

Health check endpoint.

## License

Internal use only. TRIBE v2 is licensed under CC-BY-NC-4.0 by Meta.
