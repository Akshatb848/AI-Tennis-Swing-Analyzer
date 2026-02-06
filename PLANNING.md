# AI Data Scientist Platform — Architecture & Implementation Plan

## 1. Current State Assessment

### What Exists Today

| Layer | Component | Status |
|-------|-----------|--------|
| **Frontend** | Streamlit 3-step wizard (project → upload → analysis) | Working |
| **Orchestration** | `CoordinatorAgent` with regex intent detection, workflow DAG | Working (no LLM) |
| **Data Cleaning** | Duplicate removal, median/mode imputation, IQR outlier clipping | Basic |
| **EDA** | Dataset profiling, correlation, skewness, kurtosis, auto-insights | Basic |
| **Feature Engineering** | DateTime extraction, interactions, log-transform, one-hot/freq encoding, scaling | Basic |
| **Modeling** | 6 classification + 6 regression sklearn models, cross-validation | Basic |
| **AutoML** | Dataset characterization, model recommendation scoring | Basic |
| **Visualization** | Plotly histograms, correlation heatmaps, scatter plots | Basic |
| **Dashboard** | KPI cards, multi-section layout config | Partial |

### What Works Well

- Clean agent abstraction (`BaseAgent` → `execute()`) with retry, timeout, state machine
- Async-first architecture with `asyncio`
- Standardized `TaskResult` / `AgentMessage` inter-agent protocol
- Modular agent registry pattern in the coordinator
- Workflow DAG with dependency tracking

### Critical Gaps vs. an Expert Data Scientist

| Capability | Expert Data Scientist | Current Platform |
|-----------|----------------------|-----------------|
| Natural language reasoning | Understands context, asks clarifying questions, explains "why" | Regex pattern matching — no LLM at all |
| Problem framing | Defines success metrics, understands business context | None |
| Data sourcing | SQL, APIs, web scraping, data warehouses | File upload + URL only |
| Statistical testing | t-test, chi-squared, ANOVA, Kolmogorov-Smirnov | None |
| Advanced feature engineering | Feature selection (RFE, mutual info, Boruta), target encoding, embeddings | Fixed pipeline, no selection |
| Gradient boosting | XGBoost, LightGBM, CatBoost | Only sklearn GradientBoosting |
| Deep learning | Neural nets for tabular, NLP, vision | None |
| Hyperparameter tuning | Grid/random/Bayesian search, Optuna | Default hyperparams only |
| Ensemble methods | Stacking, blending, voting | None |
| Time series | ARIMA, Prophet, LSTM, seasonal decomposition | None |
| NLP | Text classification, sentiment, NER, embeddings | None |
| Model interpretability | SHAP, LIME, partial dependence plots | Feature importance only |
| Model persistence | Save/load/version models | None |
| Report generation | PDF/HTML reports, executive summaries | Streamlit display only |
| Iterative refinement | Re-runs experiments, compares approaches, backtracks | Single-pass pipeline |
| Code generation | Writes and explains Python/SQL code | None |

---

## 2. Architecture Recommendations

### 2.1 LLM Integration (Highest Priority)

The single most impactful change is wiring an LLM into the `CoordinatorAgent`. The `llm_client` parameter already exists but is unused. This transforms the platform from a *fixed pipeline executor* into an *intelligent reasoning agent*.

**What to implement:**

```
┌─────────────────────────────────────────────────┐
│                   User (NL query)               │
│                        │                        │
│                        ▼                        │
│              ┌──────────────────┐               │
│              │ CoordinatorAgent │               │
│              │  (LLM-powered)   │               │
│              │                  │               │
│              │ • Intent parsing │               │
│              │ • Plan generation│               │
│              │ • Tool selection │               │
│              │ • Result explain │               │
│              └────────┬─────────┘               │
│                       │                         │
│         ┌─────────────┼─────────────┐           │
│         ▼             ▼             ▼           │
│  ┌────────────┐ ┌──────────┐ ┌──────────┐      │
│  │ DataClean  │ │   EDA    │ │ Modeling │ ...   │
│  └────────────┘ └──────────┘ └──────────┘      │
└─────────────────────────────────────────────────┘
```

**Specific changes:**

1. **Create `llm/` module** with an LLM client abstraction supporting OpenAI, Anthropic, and local models (Ollama).
2. **Replace regex intent detection** in `CoordinatorAgent.analyze_user_intent()` with LLM-based intent + entity extraction.
3. **Make workflow planning dynamic** — instead of the hardcoded 7-step pipeline in `plan_workflow()`, have the LLM generate a plan based on the data characteristics and user goal.
4. **Add conversational memory** — the `conversation_history` list exists but is never populated. Store user queries and agent results so the LLM has context for multi-turn interactions.
5. **Result interpretation** — after each agent produces results, have the LLM summarize findings in plain language and recommend next steps.

**File changes:**
- New: `llm/__init__.py`, `llm/client.py`, `llm/prompts.py`
- Modified: `agents/coordinator_agent.py` (wire up `llm_client`)

### 2.2 New Agent: StatisticalTestingAgent

An expert data scientist validates hypotheses with statistical rigor. Add:

- **Normality tests**: Shapiro-Wilk, Anderson-Darling, D'Agostino-Pearson
- **Comparison tests**: t-test (independent & paired), Mann-Whitney U, Wilcoxon signed-rank
- **Categorical tests**: Chi-squared, Fisher's exact
- **Multi-group tests**: ANOVA (one-way, two-way), Kruskal-Wallis
- **Correlation tests**: Pearson, Spearman, Kendall with p-values
- **Stationarity tests**: ADF, KPSS (for time series data)

All of these are available in `scipy.stats` (already a dependency).

**File:** `agents/statistical_testing_agent.py`

### 2.3 New Agent: AdvancedFeatureSelectionAgent

Current `FeatureEngineerAgent` creates features but never *selects* them. Expert data scientists prune ruthlessly:

- **Filter methods**: Variance threshold, mutual information, chi-squared scores
- **Wrapper methods**: Recursive Feature Elimination (RFE) with cross-validation
- **Embedded methods**: L1-based feature selection, tree-based importance
- **Advanced**: Boruta algorithm (requires `boruta` package)
- **Multicollinearity detection**: VIF (Variance Inflation Factor) calculation
- **Output**: Ranked feature list with selection rationale

**File:** `agents/feature_selection_agent.py`

### 2.4 Upgrade ModelTrainerAgent — Gradient Boosting & Hyperparameter Tuning

The current 6 sklearn models with default hyperparameters is the biggest modeling gap.

**Add models:**
- XGBoost (`xgboost`)
- LightGBM (`lightgbm`)
- CatBoost (`catboost`) — handles categoricals natively
- Stacking ensemble (sklearn `StackingClassifier`/`StackingRegressor`)
- Voting ensemble

**Add hyperparameter tuning:**
- `optuna` for Bayesian optimization (lightweight, no extra infra)
- Define search spaces per model
- Early stopping for boosting models
- Budget-aware tuning (time or trial limits)

**Add to `requirements.txt`:**
```
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0
optuna>=3.4.0
```

**File changes:** `agents/model_trainer_agent.py`, new `agents/hyperparameter_tuner_agent.py`

### 2.5 New Agent: ModelExplainabilityAgent

Move beyond feature importance to true interpretability:

- **SHAP values** (global and local explanations): `shap` library
- **LIME** (local interpretable explanations): `lime` library
- **Partial Dependence Plots (PDP)** and **Individual Conditional Expectation (ICE)**: sklearn built-in
- **Permutation importance**: sklearn built-in (model-agnostic)
- **Summary reports**: "Feature X increases prediction by Y when above threshold Z"

**Add to `requirements.txt`:**
```
shap>=0.43.0
lime>=0.2.0
```

**File:** `agents/model_explainability_agent.py`

### 2.6 New Agent: TimeSeriesAgent

Time series is an entire discipline that's currently missing:

- **Detection**: Auto-detect if dataset is time series (datetime index, sorted timestamps)
- **Decomposition**: Trend, seasonality, residual (via `statsmodels`)
- **Stationarity**: ADF test, differencing
- **Classical models**: ARIMA, SARIMA, Exponential Smoothing
- **ML models**: Prophet (`prophet`), XGBoost with lag features
- **Evaluation**: Walk-forward validation, MASE, MAPE metrics

**Add to `requirements.txt`:**
```
statsmodels>=0.14.0
prophet>=1.1.0
```

**File:** `agents/time_series_agent.py`

### 2.7 New Agent: NLPAgent

For text-heavy datasets:

- **Detection**: Auto-detect text columns (high cardinality strings, average token length)
- **Preprocessing**: Tokenization, stopword removal, lemmatization
- **Feature extraction**: TF-IDF, word count, sentiment scores
- **Embeddings**: Sentence-BERT (`sentence-transformers`) for dense features
- **Classification**: Text classification with fine-tuned transformers or sklearn + TF-IDF
- **Summarization / NER**: Via LLM API calls

**Add to `requirements.txt`:**
```
nltk>=3.8.0
sentence-transformers>=2.2.0
```

**File:** `agents/nlp_agent.py`

### 2.8 Model Persistence & Registry

Currently, trained models are lost when the session ends.

**Implement:**
- Save models using `joblib` (already in sklearn dependencies)
- Model registry: `models/` directory with metadata JSON per model
- Version tracking: model name + timestamp + metrics hash
- Load & predict: expose a `predict` action on `ModelTrainerAgent`
- Export: allow downloading trained models

**File:** `utils/model_registry.py`, updates to `agents/model_trainer_agent.py`

### 2.9 Data Connectors

Expert data scientists pull from everywhere:

- **SQL databases**: PostgreSQL, MySQL, SQLite via `sqlalchemy`
- **Cloud storage**: S3 via `boto3`, GCS via `gcsfs`
- **APIs**: Generic REST client with authentication
- **Data warehouses**: BigQuery, Snowflake (optional, heavy dependencies)

Start with SQLAlchemy + SQLite for immediate value:

**Add to `requirements.txt`:**
```
sqlalchemy>=2.0.0
```

**File:** `utils/data_connectors.py`

### 2.10 Report Generation Agent

Expert data scientists communicate results clearly:

- **Automated report**: Compile EDA, modeling, and interpretation into a structured document
- **Formats**: HTML (Jinja2 templates), PDF (via `weasyprint` or `fpdf2`), Markdown
- **Sections**: Executive summary, data overview, key findings, model performance, recommendations
- **LLM-powered narrative**: Use the LLM to write natural language summaries around charts and tables
- **Export**: Download button in Streamlit UI

**File:** `agents/report_generator_agent.py`, `templates/report.html`

### 2.11 Conversational Chat Interface

Replace the rigid 3-step wizard with a chat-based UI:

- **Streamlit chat**: Use `st.chat_input()` and `st.chat_message()` (available since Streamlit 1.26)
- **Multi-turn**: User can ask questions about results, request re-runs, drill into specific analyses
- **Mixed mode**: Keep the sidebar for structured config, use main area for conversation
- **Code display**: Show generated Python code alongside results so users can learn

**File changes:** Major refactor of `app.py`

### 2.12 Testing Infrastructure

Zero tests currently exist. Add:

- **Unit tests**: Per-agent test files (test each `execute()` with mock data)
- **Integration tests**: End-to-end pipeline with sample datasets
- **Framework**: `pytest` + `pytest-asyncio` for async agent tests
- **CI**: GitHub Actions workflow

**Structure:**
```
tests/
├── conftest.py           # Shared fixtures (sample DataFrames)
├── test_data_cleaner.py
├── test_eda.py
├── test_feature_engineer.py
├── test_model_trainer.py
├── test_coordinator.py
└── test_integration.py
```

---

## 3. Phased Implementation Roadmap

### Phase 1: Intelligence Layer (Foundation)

**Goal**: Transform from fixed pipeline to intelligent reasoning agent.

| # | Task | Files | Effort |
|---|------|-------|--------|
| 1.1 | LLM client abstraction (OpenAI/Anthropic/Ollama) | `llm/client.py` | Medium |
| 1.2 | Prompt templates for coordinator reasoning | `llm/prompts.py` | Medium |
| 1.3 | Wire LLM into CoordinatorAgent (replace regex) | `agents/coordinator_agent.py` | Medium |
| 1.4 | Conversational memory (populate `conversation_history`) | `agents/coordinator_agent.py` | Small |
| 1.5 | LLM-powered result interpretation after each agent | `agents/coordinator_agent.py` | Medium |
| 1.6 | Dynamic workflow planning via LLM | `agents/coordinator_agent.py` | Medium |
| 1.7 | Chat-based Streamlit UI | `app.py` | Large |
| 1.8 | Unit tests for LLM integration (mock responses) | `tests/` | Medium |

**Outcome**: Users can describe their problem in natural language. The coordinator reasons about which agents to invoke, interprets results, and suggests next steps.

### Phase 2: Statistical & Analytical Depth

**Goal**: Match the analytical rigor of an expert data scientist.

| # | Task | Files | Effort |
|---|------|-------|--------|
| 2.1 | StatisticalTestingAgent (hypothesis tests, normality, ANOVA) | `agents/statistical_testing_agent.py` | Medium |
| 2.2 | AdvancedFeatureSelectionAgent (RFE, mutual info, VIF) | `agents/feature_selection_agent.py` | Medium |
| 2.3 | Upgrade ModelTrainerAgent with XGBoost/LightGBM/CatBoost | `agents/model_trainer_agent.py` | Medium |
| 2.4 | HyperparameterTunerAgent with Optuna | `agents/hyperparameter_tuner_agent.py` | Medium |
| 2.5 | Ensemble methods (stacking, voting) | `agents/model_trainer_agent.py` | Small |
| 2.6 | ModelExplainabilityAgent (SHAP, LIME, PDP) | `agents/model_explainability_agent.py` | Medium |
| 2.7 | Model persistence & registry | `utils/model_registry.py` | Medium |
| 2.8 | Tests for all Phase 2 agents | `tests/` | Medium |

**Outcome**: The platform can run proper hypothesis tests, select features rigorously, train state-of-the-art gradient boosting models with tuned hyperparameters, and explain predictions.

### Phase 3: Domain Specialization

**Goal**: Handle time series, NLP, and diverse data sources.

| # | Task | Files | Effort |
|---|------|-------|--------|
| 3.1 | TimeSeriesAgent (decomposition, ARIMA, Prophet) | `agents/time_series_agent.py` | Large |
| 3.2 | NLPAgent (text preprocessing, TF-IDF, embeddings) | `agents/nlp_agent.py` | Large |
| 3.3 | Auto-detect dataset type (tabular / time series / text) | `agents/coordinator_agent.py` | Medium |
| 3.4 | SQL data connector (SQLAlchemy) | `utils/data_connectors.py` | Medium |
| 3.5 | ReportGeneratorAgent (HTML/PDF reports) | `agents/report_generator_agent.py` | Medium |
| 3.6 | Coordinator routes to specialized agents based on data type | `agents/coordinator_agent.py` | Medium |
| 3.7 | Tests for Phase 3 agents | `tests/` | Medium |

**Outcome**: The platform can handle time series forecasting, text classification, connect to databases, and produce professional reports.

### Phase 4: Production Readiness

**Goal**: Make it deployable, testable, and maintainable.

| # | Task | Files | Effort |
|---|------|-------|--------|
| 4.1 | Dockerize the application | `Dockerfile`, `docker-compose.yml` | Medium |
| 4.2 | REST API layer (FastAPI) alongside Streamlit | `api/` | Large |
| 4.3 | Authentication & multi-user support | `api/auth.py` | Medium |
| 4.4 | Model serving endpoint (predict API) | `api/predict.py` | Medium |
| 4.5 | CI/CD with GitHub Actions | `.github/workflows/` | Medium |
| 4.6 | Logging & monitoring (structured logs, metrics) | `utils/logging.py` | Medium |
| 4.7 | Configuration management (YAML-based) | `config/` | Small |
| 4.8 | Comprehensive integration test suite | `tests/` | Large |

**Outcome**: The platform is a production-grade service with API access, containerized deployment, and CI/CD.

---

## 4. Immediate Next Steps (Recommended Starting Points)

If you want to start building today, here's the order of maximum impact:

### Step 1 — LLM Client + Coordinator Integration

Create `llm/client.py` with a simple interface:

```python
class LLMClient:
    async def chat(self, messages: list[dict], tools: list[dict] = None) -> str:
        """Send messages to the LLM and return the response."""
        ...
```

Support at least one provider (OpenAI or Anthropic) and wire it into `CoordinatorAgent`. This immediately makes the platform *intelligent* instead of pattern-matching.

### Step 2 — Add XGBoost/LightGBM + Optuna

Biggest modeling upgrade for the least effort. XGBoost and LightGBM are the go-to algorithms for tabular data competitions and real-world problems. Optuna adds smart hyperparameter tuning.

### Step 3 — SHAP Explainability

After training better models, add SHAP explanations. This is what separates a tool from an expert — the ability to explain *why* the model made a prediction.

### Step 4 — Chat UI

Replace the rigid wizard with `st.chat_input()` / `st.chat_message()`. This enables the iterative, conversational workflow that defines expert data science (ask question → get answer → refine → repeat).

---

## 5. Updated Dependency List

```
# requirements.txt (proposed)

# Web Framework
streamlit>=1.30.0

# Data Processing
pandas>=2.0.0
numpy>=1.24.0

# Machine Learning - Core
scikit-learn>=1.3.0

# Machine Learning - Gradient Boosting
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2.0

# Hyperparameter Tuning
optuna>=3.4.0

# Model Interpretability
shap>=0.43.0
lime>=0.2.0

# Visualization
plotly>=5.18.0
matplotlib>=3.8.0
seaborn>=0.13.0

# Statistical Analysis
scipy>=1.11.0
statsmodels>=0.14.0

# Time Series
prophet>=1.1.0

# NLP
nltk>=3.8.0
sentence-transformers>=2.2.0

# Data Connectors
sqlalchemy>=2.0.0

# LLM Integration
openai>=1.10.0
anthropic>=0.18.0

# Report Generation
jinja2>=3.1.0

# Utilities
python-dotenv>=1.0.0
joblib>=1.3.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.23.0
```

---

## 6. Key Architecture Principles

1. **LLM as the brain, agents as the hands.** The coordinator reasons and plans; specialized agents execute. Never hard-code workflow order when an LLM can decide dynamically.

2. **Every agent must be independently testable.** Each agent receives a dict, returns a `TaskResult`. No hidden dependencies between agents.

3. **Fail gracefully, explain clearly.** When a model fails or data is unsuitable, produce a clear explanation rather than a stack trace. The LLM coordinator should translate errors into actionable advice.

4. **Preserve the async foundation.** The current `asyncio` architecture is correct. Keep it — it enables parallel agent execution (e.g., run EDA and visualization concurrently).

5. **Data in, insights out.** The platform should not just produce metrics — it should produce *narratives*. "Your best model is Random Forest with 94% accuracy" is a metric. "Customer churn is primarily driven by account age and support ticket frequency; targeting customers with accounts < 6 months old with proactive outreach could reduce churn by an estimated 15%" is an insight.

6. **Keep it modular.** New agents should be droppable into the registry without changing the coordinator. The coordinator discovers capabilities from agent metadata.
