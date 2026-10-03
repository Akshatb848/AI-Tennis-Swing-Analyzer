# TennisIQ: AI Tennis Swing & Match Analyzer

TennisIQ is a single-camera tennis analysis platform. You upload a match or practice video and a Python backend
works through it: it finds the court, tracks the ball and players, segments rallies, compiles match statistics,
and produces a coaching report with frame-level annotations and a timed narration script. The results are shown in a
Next.js web app and a Streamlit analytics dashboard.

All of the product code is in [`tennis/`](tennis/). See [`tennis/README.md`](tennis/README.md) for more module-level detail.
That file describes the *target* ML model stack. The [status section](#current-status--limitations) below describes what
the code actually runs today.

---

## What it does

- **Video upload and analysis jobs**: upload a video (`POST /api/v1/upload/`), then start an analysis job
  (`POST /api/v1/analyze`) and poll it for stage-by-stage progress until results, coaching, overlays and the voice script are ready.
- **Court, ball and player detection** with OpenCV: HSV colour segmentation, Hough lines and circles, frame differencing and contours.
- **Shot classification, rally segmentation and match stats** derived from the detected motion and ball activity.
- **Mixture-of-Experts coaching pipeline**: specialist agents each add their findings to a shared context, and the pipeline
  ends in a structured coaching report and narration events.
- **Supporting domain engines**: tennis scoring, line calling, event processing, stats calculation, post-match review and recording.
- **Clients**: Next.js web app (upload → configure → processing → results), a Streamlit analytics dashboard, and a SwiftUI iOS/watchOS scaffold.

## Architecture

```
            ┌──────────────────────┐        ┌──────────────────────────┐
            │  Next.js frontend    │        │  Streamlit dashboard     │
            │  tennis/frontend     │        │  tennis/dashboard        │
            └──────────┬───────────┘        └────────────┬─────────────┘
                       │ REST (/api/v1/...)              │
            ┌──────────▼─────────────────────────────────▼─────────────┐
            │  FastAPI app: tennis/api/app.py                          │
            │  routes: upload, analyze, sessions, matches, events,     │
            │  stats, coaching, video, linecalls, recording, replay,   │
            │  analytics, auth, subscriptions                          │
            └──────────┬───────────────────────────────────────────────┘
                       │ background task
            ┌──────────▼───────────────────────────────────────────────┐
            │  tennis/video/fast_analyzer.py  (job pipeline)           │
            │  metadata → court → motion → ball → shots → rallies →    │
            │  stats → MoE agents                                      │
            └──────────┬───────────────────────────────────────────────┘
                       │
            ┌──────────▼───────────────────────────────────────────────┐
            │  tennis/agents/moe_orchestrator.py                       │
            │  Court → Ball + Player → Biomechanics + Strategy →       │
            │  Coaching → Voice   (shared context dict, MoEResult)     │
            └──────────────────────────────────────────────────────────┘
```

**Agents** (`tennis/agents/`):

| Agent | Role (as implemented) |
|-------|-----------------------|
| `court_agent.py` | Court boundaries, net, baseline and service boxes from HSV segmentation + Hough line detection |
| `ball_agent.py` | Ball detection (yellow-green HSV mask + HoughCircles), trajectories, speed and bounce estimates |
| `player_agent.py` | Player detection with contours / background subtraction, plus position and movement tracking |
| `biomechanics_agent.py` | Swing and technique issues from motion intensity and position data, with frame annotations |
| `strategy_agent.py` | Shot placement zones, rally patterns, court positioning |
| `coaching_agent.py` | Combines all expert outputs into a structured, prioritised coaching report |
| `voice_agent.py` | A timed narration script built from templates (the browser plays it with the Web Speech API) |
| `moe_orchestrator.py` | Runs the agents in dependency order. One agent failing does not stop the others (`_safe_run`) |

The other packages are `tennis/engine/` (scoring, line calling, event processing, stats, shot classifier, review),
`tennis/ml/` (frame analyzer, detectors, inference pipeline, model specs), `tennis/video/` (capture, frame buffer,
overlays, highlights, avatar renderer), `tennis/models/` (Pydantic data models) and `tennis/infra/` (Docker,
Nginx, Prometheus config, an SQLite-backed `SessionStore`).

## Tech stack

- **Backend:** Python 3.11, FastAPI, Uvicorn, Pydantic v2 / pydantic-settings, OpenCV (headless), NumPy, PyJWT
- **Dashboard:** Streamlit, Plotly, pandas
- **Frontend:** Next.js 14, React 18, TypeScript, Tailwind CSS, TanStack Query, Zustand, Recharts, Axios
- **Mobile scaffold:** Swift / SwiftUI (iOS + watchOS source files, no Xcode project)
- **Infra:** Docker, docker-compose (API, Postgres, Redis, Nginx, dashboard), Prometheus config
- **Testing:** pytest

## Running locally

Run all commands from the repository root.

### 1. Backend API

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install fastapi "uvicorn[standard]" pydantic pydantic-settings python-multipart \
            httpx numpy pandas plotly "opencv-python-headless>=4.8,<5" PyJWT google-auth aiofiles \
            streamlit pytest

uvicorn tennis.api.app:app --host 0.0.0.0 --port 8000
```

- Interactive API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health

The root `requirements.txt` also lists the dependencies of the unrelated legacy code (see
[below](#note-on-root-level-legacy-code)), such as xgboost, lightgbm, chromadb and psycopg2. Installing it works, but it
is heavier than TennisIQ needs. The list above covers what `tennis/` imports.
Keep OpenCV on 4.x: OpenCV 5 changed the output shape of the Hough transforms, and the current detection code
fails with it.

Settings come from environment variables (see `tennis/config.py`), e.g. `ENVIRONMENT`, `JWT_SECRET`, `CORS_ORIGINS`,
`GOOGLE_CLIENT_ID`.

### 2. Streamlit dashboard

```bash
streamlit run tennis/dashboard/tennis_dashboard.py
```

The dashboard opens on http://localhost:8501. It builds its session list from an in-process store in
`tennis/dashboard/data_provider.py`, and the API does not populate that store when the two run as separate processes.
In practice the dashboard therefore shows generated sample data under a "Demo Data" banner.

### 3. Next.js frontend

```bash
cd tennis/frontend
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

The frontend opens on http://localhost:3000. `NEXT_PUBLIC_API_URL` defaults to `http://localhost:8000` if unset.

### 4. Docker (backend stack)

```bash
docker compose -f tennis/infra/docker-compose.yml up --build
```

This builds the API and dashboard from `tennis/infra/Dockerfile` (which installs the root `requirements.txt`) and starts
Postgres, Redis and Nginx next to them. The current API code keeps its state in memory and does not use Postgres or Redis yet.

## Running tests

```bash
python -m pytest -q tennis/tests
```

The 149 tests in `tennis/tests/` cover scoring, event processing, stats, coaching, line calling, recording, review, the ML
pipeline, the real-time pipeline, the avatar renderer and the API (through FastAPI's `TestClient`).

## Project structure

```
.
├── README.md               ← you are here
├── tennis/                 ← TennisIQ (the actual product)
│   ├── agents/             MoE agents + orchestrator
│   ├── api/                FastAPI app, routes, middleware
│   ├── engine/             scoring, line calling, stats, coaching, review
│   ├── ml/                 frame analyzer, detectors, inference pipeline, model specs
│   ├── video/              analysis job pipeline, capture, overlays, highlights
│   ├── models/             Pydantic domain models
│   ├── dashboard/          Streamlit dashboard + data provider
│   ├── frontend/           Next.js web app
│   ├── ios/                Swift/SwiftUI scaffold (iOS + Watch)
│   ├── infra/              Dockerfile, docker-compose, nginx, monitoring, SQLite session store
│   └── tests/              pytest suite
└── agents/ core/ services/ dashboard/ tests/ utils/ config/ data/ rag_docs/ chroma_db/
    main.py setup_rag.py replit.md start.ps1 attached_assets/
                            ← unrelated legacy code (see note below)
```

## Current status & limitations

TennisIQ is a working prototype and is not production-ready. Specifically:

- **Vision is heuristic, not learned.** Detection uses classical OpenCV techniques: colour masks, Hough transforms,
  contours and frame differencing. `tennis/ml/frame_analyzer.py` always runs in heuristic mode ("In current deployment, we
  use heuristic mode"). The models in `tennis/ml/model_specs.py` and `tennis/README.md` (BallNet, PlayerNet, CourtNet,
  ShotNet) are specifications only. The repo contains no trained weights, so no accuracy or latency figures are claimed.
- **Coaching and narration are rule- and template-based.** An `OPENAI_API_KEY` setting exists, but the current coaching
  code does not call an LLM.
- **Authentication is a placeholder.** `tennis/api/middleware.py` defines an `AuthMiddleware` whose token validation is a
  stub (it accepts only the literal `test_token`) and lets all requests through in development mode. The middleware is
  not registered in `tennis/api/app.py`. `routes_auth.py` issues JWTs, but passwords are hashed with salted SHA-256
  rather than a dedicated password hash such as bcrypt or argon2.
- **Storage is in memory.** Users, sessions, matches, events, coaching data and analysis jobs live in Python dicts
  and are lost on restart. A SQLite `SessionStore` exists in `tennis/infra/session_store.py` and is unit-tested, but the
  API routes do not use it yet.
- **Some dashboard panels use demo data.** In `tennis/dashboard/tennis_dashboard.py`, the player-comparison radar
  (fixed values), the serve-speed histogram (simulated around the average), the shot-placement heatmap (random) and the
  whole *Trends* view (random) are not computed from real matches. Each of these is labelled "Demo data" in the UI.
  The dashboard is also not yet wired to the API's analysis results, so in practice it runs on sample data
  (shown with a "Demo Data" banner).
- **iOS app is a scaffold.** It contains Swift source files with no Xcode project or build setup.
- **Frontend Docker image is untested.** `tennis/frontend/Dockerfile` copies `.next/standalone` and `public/`, but
  `next.config.js` does not set `output: 'standalone'` and the repo has no `public/` directory, so that image will
  not build as-is. `npm run dev` works for local use.
- **No CI** is configured in this repository.

## Note on root-level legacy code

The top-level directories `agents/`, `core/`, `services/`, `dashboard/`, `tests/`, `utils/`, `config/`, `data/`,
`rag_docs/`, `chroma_db/` and the files `main.py`, `setup_rag.py`, `replit.md` and `start.ps1` are **not part of TennisIQ**.
They belong to a separate "AI Data Scientist" multi-agent platform and came into this repo through shared git history
with [Akshatb848/data-science-agent-platform](https://github.com/Akshatb848/data-science-agent-platform), which is where that
project lives. For example, `python main.py` starts that platform's API (`services.api:app`), not TennisIQ.
This code is left in place for now and is expected to be removed in a future cleanup.
