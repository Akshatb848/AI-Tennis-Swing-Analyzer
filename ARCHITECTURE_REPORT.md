# Architecture & Failure-First Blueprint — Intelligent Data Scientist Agent

## 0) Scope & Non‑Negotiables
This report audits the current repository and designs a failure‑proof, production‑grade Intelligent Data Scientist Agent. It focuses on correctness, robustness, explicit failure handling, and UI determinism (especially Streamlit element ID safety).

---

## 1) Current State Forensic Analysis

### 1.1 Current Agent & Execution Architecture (Observed)
**Entry point & UI**
- `app.py` is a Streamlit app that wires in agents, collects user input, and renders results (charts, KPIs, narratives). The UI is a chat‑style interface that triggers analysis and displays results. 
- UI rendering is stateful via `st.session_state`, and charts are rendered repeatedly inside loops without explicit Streamlit `key`s. This is the direct cause of `StreamlitDuplicateElementId` for dynamic or repeated charts.

**Coordinator / Orchestration**
- `CoordinatorAgent` is the master orchestrator with workflow planning, state, and conversational memory scaffolding.
- The coordinator uses an LLM client (with a fallback) for intent analysis and result interpretation. It stores conversation history and analysis history, but it does **not** enforce strict workflow contracts, explicit retry/fallback boundaries per step, or UI‑safe execution checkpoints.

**Specialized Agents**
- Agents exist for data cleaning, EDA, feature engineering, modeling, AutoML, dashboards, data visualization, forecasting, insights, and report generation.
- Outputs are rendered immediately in the UI, which couples agent execution to UI rendering. There is no central “render registry” or explicit chart ID management.

### 1.2 Streamlit UI Failure Root Cause (Duplicate Element IDs)
**Primary failure**: `streamlit.errors.StreamlitDuplicateElementId`
- Root cause: `st.plotly_chart` is invoked inside loops without deterministic `key`s. Streamlit auto‑generates element IDs, and when the same chart is re‑rendered on rerun (or multiple similar charts appear in a loop), it produces collisions.
- The current UI renders charts in multiple paths (e.g., `_render_charts`, dashboard sections, model comparison, target analysis). None of these specify `key`s or register an element ID policy.

### 1.3 Hidden Assumptions & Coupling Points
- **Assumption**: reruns are harmless. In Streamlit, reruns can re‑create identical elements without IDs, leading to collisions.
- **Assumption**: UI rendering is stateless. In reality, `st.session_state` and automatic reruns create hidden dependencies between render order and element IDs.
- **Assumption**: a single pass pipeline is sufficient. The coordinator lacks a robust step registry with explicit completion/rollback and does not isolate UI rendering from execution steps.
- **Coupling**: agent execution directly triggers UI rendering, which couples computation to UI lifecycle and rerun order.

### 1.4 Failure‑Aware Current Architecture Map (Textual)
```
User
  │
  ▼
Streamlit UI (app.py)
  │  ├─ Session state
  │  ├─ Direct chart rendering (plotly_chart w/o keys)  <-- failure hotspot
  │  └─ User input -> coordinator
  ▼
CoordinatorAgent
  │  ├─ LLM intent analysis (fallback client)
  │  ├─ Static-ish workflow routing
  │  ├─ Agent dispatch
  │  └─ Result interpretation
  ▼
Specialized Agents
  │  ├─ Data cleaning
  │  ├─ EDA
  │  ├─ Feature engineering
  │  ├─ Modeling/AutoML
  │  ├─ Visualization/Dashboard
  │  └─ Report/Insights
  ▼
UI Rendering
  ├─ Charts & KPIs (re-render on rerun)
  └─ No explicit keying/ID registry
```

---

## 2) Target Architecture Blueprint (Failure‑Proof by Design)

### 2.1 Layered Architecture (Required)
1) **User Interaction Layer**
   - Deterministic rendering with explicit `key`s
   - UI registry that tracks rendered elements per run
   - Consistent rerun ordering

2) **Intent & Task Intelligence Layer**
   - LLM intent classification with confidence and ambiguity detection
   - Task decomposition and sequencing
   - Clarification questions only when needed

3) **Orchestration & Control Layer**
   - State machine with explicit step boundaries
   - Retry/rollback logic per step
   - Execution checkpoints before UI rendering

4) **Data Understanding Layer**
   - Dataset profiling, schema inference, quality scoring
   - Dataset identity hashing (for UI key determinism)

5) **Data Science Execution Layer**
   - Modular EDA, feature engineering, modeling, visualization
   - Separation between computation and rendering

6) **Reasoning & Explanation Layer**
   - LLM‑based explanation with assumptions and confidence
   - Decision rationale and actionable next steps

7) **Memory & State Layer**
   - Short‑term session state
   - Long‑term project memory
   - Render history registry

8) **Production & Safety Layer**
   - Structured logging
   - Error classification and user‑safe messaging
   - UI collision prevention and throttling

---

## 3) UI & Streamlit Failure Immunity Strategy (Mandatory)

### 3.1 Deterministic Keying Strategy (Non‑Optional)
Every UI element **must** have a stable key that includes:
- Dataset hash (or project ID)
- Execution step ID
- Chart type / content signature
- Optional rerun index (when multiple identical charts are expected)

**Key example**:
```
key = f"{dataset_hash}:{workflow_step_id}:{chart_type}:{chart_signature}"\
       f":{render_index}"
```

### 3.2 UI Registry & Collision Detection
- Maintain a render registry in `st.session_state.render_registry`.
- On each render request:
  - Generate key
  - Check registry for collision
  - If collision: increment index or append UUID
  - Register the new key
- If collision resolution fails, render a safe fallback (e.g., text summary) and log the error.

### 3.3 Render Safety Boundaries
- **No UI rendering inside agent execution**; agents produce pure data structures only.
- UI rendering should be performed in one deterministic pass at the UI layer.
- Reruns must replay the exact same rendering order using stored render metadata.

---

## 4) Data Robustness by Design

| Data Type | Detection Strategy | Decision Logic | Fallback Behavior | Failure Messaging |
|---|---|---|---|---|
| Clean/Dirty CSV | Schema scan, missingness, dtype mix | If dirty: route through cleaner | Highlight cleaning choices | "Data required cleaning: X rows fixed" |
| Missing values | Missingness per column | Impute vs drop based on rate | Keep report of imputation | "Missing values imputed with median/mode" |
| Mixed datatypes | dtype conflict detection | Cast or split columns | Keep raw copy | "Mixed dtypes detected; coercion applied" |
| High‑cardinality categories | cardinality % | Encode using target/frequency | Cap rare categories | "High cardinality; frequency encoding used" |
| Time series | datetime index / order | Route to time‑series engine | Use generic modeling | "Time series detected; forecasting mode enabled" |
| Imbalanced datasets | label distribution | Use reweighting / resample | Warn user | "Class imbalance detected; reweighting applied" |
| JSON / nested | JSON parser + flattening | Normalize into tables | Keep raw extraction | "Nested JSON flattened; check schema" |
| Text | avg token length + cardinality | Route to NLP agent | Bag‑of‑words fallback | "Text columns detected; NLP pipeline used" |
| Empty / corrupted | file read validation | Block pipeline | Provide instructions | "Dataset empty or unreadable" |

---

## 5) Professional Data Scientist Behavior Guarantees
- **Clarification gate**: If intent confidence < threshold, ask clarifying questions before execution.
- **Metric gating**: Choose metrics based on task type (classification vs regression).
- **Unsafe request refusal**: detect PII leakage or nonsensical tasks.
- **Decision rationale**: provide a “why” explanation after each step.
- **Expectation check**: detect unrealistic performance goals and warn.

---

## 6) Failure‑First System Design

### 6.1 Failure Handling Matrix
| Failure | Detection | Response | User Messaging | Observability |
|---|---|---|---|---|
| LLM failure | API error / timeout | Fallback client | "LLM unavailable; using fallback reasoning" | Logged with error code |
| Tool failure | Exception | Retry → fallback | "Step failed; retrying / fallback used" | Traceable step ID |
| UI collision | Key collision | Resolve or fallback text | "Chart rendering skipped due to UI collision" | Render registry audit |
| Data leakage | Target leakage detection | Stop pipeline | "Potential leakage detected; adjust features" | Logged & surfaced |
| Partial execution | Missing step outputs | Resume / replan | "Resuming from last safe checkpoint" | Step‑level audit |

---

## 7) Exhaustive Testing Framework (Zero Exceptions)

### 7.1 Data Testing
- Clean/dirty CSV, missing values, mixed dtypes, high cardinality, time series, imbalanced, JSON, text, empty/corrupted.
- Each test validates detection, routing, fallback, and messaging.

### 7.2 Intent Testing
- Explicit tasks, ambiguous goals, contradictory goals, mid‑execution redirects.

### 7.3 Pipeline Testing
- Partial execution, failed model training, metric mismatch, leakage detection, evaluation misuse.

### 7.4 UI / Frontend Testing
- Re‑render loops, multiple charts of same type, dynamic chart generation, conditional UI blocks, session restarts.

### 7.5 Failure Injection
- Tool failures, LLM hallucinations, timeouts, memory corruption, UI state desync.

Each test defines: **Detection mechanism**, **Expected behavior**, **Recovery strategy**, **User‑visible messaging**.

---

## 8) Gap‑Driven Refactor Plan

### 8.1 Key Gaps
1. **UI element ID collisions (Critical)**
   - Root cause: no deterministic keys; UI rendering inside loops
   - Fix: explicit key strategy + render registry + separation of compute/render

2. **Weak orchestration boundaries (High)**
   - Root cause: agent execution and rendering coupled
   - Fix: orchestrator outputs structured render payloads; UI renders in a deterministic pass

3. **Missing failure classification (High)**
   - Root cause: exceptions not categorized
   - Fix: error taxonomy + structured logging

4. **No dataset identity tracking (Medium)**
   - Root cause: no dataset hashing
   - Fix: hash dataset + use hash in UI keys and memory

5. **No explicit clarification policy (Medium)**
   - Root cause: intent routing lacks confidence gating
   - Fix: enforce confidence thresholds and clarification prompts

### 8.2 Refactor Strategy (Incremental)
1. **Introduce UI render registry** in `app.py`
2. **Separate render payloads from agent outputs**
3. **Add dataset hash to session state**
4. **Enforce key generation for every Streamlit element**
5. **Add structured error taxonomy & logging**

---

## 9) Production Readiness Checklist
- [ ] Deterministic UI rendering with explicit keys
- [ ] Render registry and collision detection
- [ ] Step‑level orchestration logs
- [ ] Retry/rollback boundaries
- [ ] Dataset identity hashing
- [ ] Clarification gating
- [ ] Structured error taxonomy
- [ ] Test suite covering UI collisions and failure injection

---

## 10) Migration Plan (Safe, Incremental)
1. Add render registry + deterministic key builder
2. Introduce dataset hashing and attach to render keys
3. Refactor agent outputs to pure data payloads
4. Add orchestration checkpoints and failure taxonomy
5. Add UI collision tests + failure injection tests
6. Integrate clarification gating and confidence thresholds

---

## 11) Deliverables Summary
- Current architecture map and failure analysis
- Target architecture blueprint
- UI‑safe rendering strategy
- Failure handling matrix
- Exhaustive testing plan
- Production readiness checklist
- Migration plan
