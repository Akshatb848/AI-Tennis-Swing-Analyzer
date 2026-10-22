# TennisIQ — AI-First Tennis Intelligence Platform

> Single-camera tennis analysis: upload a video, get court/ball/player detection, match stats and a coaching report.
>
> See the [root README](../README.md) for how to run it and for what the code does today.

## Architecture

```
tennis/
├── models/          # Pydantic data models (match, player, events, coaching, subscription)
├── engine/          # Core logic (scoring, event processor, stats, coaching, shot classifier)
├── api/             # FastAPI REST API (7 route modules + middleware)
├── ml/              # ML pipeline (model specs, ball/player/court tracking, inference)
├── video/           # Video processing (transcode, highlights, overlays)
├── dashboard/       # Streamlit analytics dashboard
├── ios/             # Swift scaffolding (capture, ML, UI, Watch, engine)
├── infra/           # Docker, Nginx, monitoring, security
└── tests/           # pytest suite
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tennis/tests/ -v

# Start API server
uvicorn tennis.api.app:app --host 0.0.0.0 --port 8000

# Launch dashboard
streamlit run tennis/dashboard/tennis_dashboard.py

# Docker (full stack)
docker-compose -f tennis/infra/docker-compose.yml up
```

## ML Model Stack (design targets, not implemented)

> These are the specifications in `tennis/ml/model_specs.py`. No trained weights ship with this repo and none of these
> parameter counts or latencies have been measured; today's pipeline uses classical OpenCV heuristics instead.

| Model | Backbone | Params | Latency | Task |
|-------|----------|--------|---------|------|
| BallNet | YOLOv8-nano | 3.2M | 8ms | Ball detection & tracking |
| PlayerNet | MoveNet Thunder | 5.4M | 12ms | Player detection + pose |
| CourtNet | ResNet18-Lite | 2.1M | 6ms | Court line detection |
| ShotNet | 1D-CNN + LSTM | 0.9M | 3ms | Shot classification |
| **Total (target)** | | **11.6M** | **29ms** | *<50ms target* |

## Subscription Tiers (planned)

| Feature | Free | Pro | Elite |
|---------|------|-----|-------|
| Live scoring | ✅ | ✅ | ✅ |
| Basic stats | ✅ | ✅ | ✅ |
| Advanced analytics | ❌ | ✅ | ✅ |
| AI coaching | ❌ | ✅ | ✅ |
| Video highlights | ❌ | ✅ | ✅ |
| Swing analysis | ❌ | ❌ | ✅ |
| Cloud storage | 5 sessions | 100 sessions | Unlimited |

## API Documentation

Start the server and visit `http://localhost:8000/docs` for interactive Swagger documentation.

## License

MIT, see [LICENSE](../LICENSE).
