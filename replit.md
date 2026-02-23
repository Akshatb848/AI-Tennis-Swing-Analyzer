# AI Data Scientist Platform

## Overview
A production-grade multi-agent AI system that functions as a full-stack data science team. The platform uses a 5-pillar architecture with LLM-powered analysis (Meta Llama 3.3 70B via OpenRouter), RAG knowledge retrieval (ChromaDB), automated ML pipeline with advanced algorithms (XGBoost, LightGBM, Optuna), and an interactive Streamlit dashboard.

## Architecture

### 5-Pillar Agent System
The platform uses a central Orchestrator (`agents/orchestrator.py`) that coordinates 5 specialized agents:

1. **BusinessStrategyAgent** (`agents/business_strategy_agent.py`) - Defines ML objectives, KPIs, constraints from dataset profiles. Uses LLM when available, falls back to rules.
2. **DataEngineeringAgent** (`agents/data_engineering_agent.py`) - Distribution-aware transforms (Box-Cox, Yeo-Johnson), KNN/median/mode imputation, outlier clipping (IQR), target encoding, interaction features, PCA, mutual-information feature selection, train/test split.
3. **ExploratoryAnalysisAgent** (`agents/exploratory_analysis_agent.py`) - Normality tests (Shapiro-Wilk/D'Agostino), correlation significance, ANOVA/Kruskal-Wallis, chi-square, VIF multicollinearity detection, auto-generated visualization specs.
4. **ModelingMLAgent** (`agents/modeling_ml_agent.py`) - Multi-model training (LogisticRegression, RandomForest, GradientBoosting, XGBoost, LightGBM), Optuna hyperparameter tuning, champion selection.
5. **MLOpsDeploymentAgent** (`agents/mlops_deployment_agent.py`) - Model serialization, inference script generation, monitoring config, deployment recommendations.

### LLM+RAG Architecture
- `core/llm_client.py` - OpenRouter LLM client via Replit AI Integrations. Uses Meta Llama 3.3 70B (primary) and Mistral Small 3.1 (fallback). Provides chat, dataset analysis, results analysis, improvement suggestions, and context-aware Q&A. Falls back to rule-based analysis when unavailable.
- `core/rag_client.py` - ChromaDB RAG client for knowledge retrieval, auto-indexes docs on first run. 20 chunks from ML best practices, deployment guides, troubleshooting docs.
- Chat interface feeds full pipeline context (profile, engineering, exploration, modeling, deployment) to the LLM for deeply context-aware responses.

### Technology Stack
- **Frontend**: Streamlit dashboard on port 5000 (dark sidebar, Inter font, professional UI)
- **Backend API**: FastAPI on port 8000 (optional, for programmatic access)
- **Database**: PostgreSQL for metadata tracking (datasets, experiments, model artifacts)
- **ML**: scikit-learn, XGBoost, LightGBM, Optuna
- **RAG**: ChromaDB with default embedding function
- **LLM**: Meta Llama 3.3 70B via OpenRouter (Replit AI Integrations, no API key needed)
- **Visualization**: Plotly

## Project Structure
```
core/               - Core layer (Pydantic models, LLM client, RAG client)
agents/             - 5-pillar agent modules + base class + orchestrator
services/           - API layer (FastAPI) and metadata service (PostgreSQL)
dashboard/          - Streamlit dashboard application
dashboard/components/ - Chat interface, visualization components
utils/              - Shared utilities (csv_loader for robust file parsing)
config/             - Application settings and configuration
rag_docs/           - RAG knowledge base documents (ML best practices, deployment, troubleshooting)
tests/              - Unit tests
models/             - Saved model artifacts
data/uploads/       - Uploaded dataset storage
```

## How It Runs
- Streamlit dashboard: `streamlit run dashboard/app.py --server.port 5000 --server.address 0.0.0.0`
- FastAPI backend: `python -m services.api` (port 8000)
- Users upload datasets or select samples, pipeline runs automatically through all 5 pillars

## Key Design Decisions
- LLM client uses OpenRouter via Replit AI Integrations (env vars AI_INTEGRATIONS_OPENROUTER_BASE_URL, AI_INTEGRATIONS_OPENROUTER_API_KEY auto-managed)
- App fully functional with rule-based fallback when LLM unavailable
- RAG auto-indexes knowledge base documents on first run if not already indexed
- Chat interface provides 3 quick-action buttons: Analyze Results, Suggest Improvements, Explain Features
- Full pipeline context (all 6 data sources) passed to LLM for context-aware responses
- DeepSeek R1 thinking tags stripped from responses
- Optuna tuning is optional via checkbox (defaults to enabled)
- CSV loader handles all encodings, delimiters, BOM, bad lines, quoted fields, duplicate columns

## Configuration
- Database connection via `DATABASE_URL` environment variable (PostgreSQL)
- LLM access via Replit AI Integrations (OpenRouter) - auto-configured, no manual API keys
- Streamlit configuration in `.streamlit/config.toml`

## Recent Changes
- 2026-02-23: Replaced Ollama with OpenRouter LLM (Llama 3.3 70B) via Replit AI Integrations
- 2026-02-23: Enhanced chat interface with pipeline-aware context, results analysis, improvement suggestions, quick-action buttons
- 2026-02-23: Full pipeline context (profile, engineering, exploration, modeling, mlops) fed to LLM for contextual answers
- 2026-02-23: Complete 5-pillar architecture rebuild with LLM/RAG integration, XGBoost/LightGBM/Optuna, ChromaDB RAG
- 2026-02-23: Professional dashboard UI (dark sidebar, Inter font, 6 tabs, structured cards, no emojis)
- 2026-02-22: Robust CSV loader with encoding/delimiter auto-detection, 31 tests passing
