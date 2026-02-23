# Troubleshooting Guide

## Data Loading Issues

### CSV encoding errors
- Symptom: UnicodeDecodeError or garbled characters
- Solution: The robust CSV loader auto-detects encoding. If manual override needed, try UTF-8, Latin-1, or CP1252

### Delimiter detection failure
- Symptom: All data in a single column
- Solution: Specify delimiter explicitly. Common delimiters: comma, semicolon, tab, pipe

### Empty dataset after loading
- Symptom: DataFrame has 0 rows
- Solution: Check if the file has data rows below the header. Verify the file isn't empty.

### Memory errors with large files
- Symptom: MemoryError or process killed
- Solution: Use sample_rows parameter to load a subset. Typical limit: 20,000 rows for profiling.

## Model Training Issues

### XGBoost/LightGBM not available
- Symptom: ImportError for xgboost or lightgbm
- Solution: Install with pip: `pip install xgboost lightgbm`. Falls back to sklearn models if unavailable.

### Training takes too long
- Symptom: Pipeline hangs during modeling step
- Solution: Reduce Optuna trials (set use_optuna=False), reduce dataset size, or use fewer models

### Poor model performance
- Symptom: Low accuracy/R-squared scores
- Solution: Check for data quality issues, try feature engineering, check for class imbalance, consider more data

### Convergence warnings
- Symptom: ConvergenceWarning from sklearn
- Solution: Increase max_iter for linear models, or try different solver. Usually not critical.

## Dashboard Issues

### Streamlit not loading
- Symptom: Blank page or connection error
- Solution: Check that Streamlit is running on the correct port (5000). Verify no other process uses the port.

### Charts not rendering
- Symptom: Empty chart areas or Plotly errors
- Solution: Ensure data is not empty. Check for NaN values in chart data. Verify column names match.

### Session state errors
- Symptom: KeyError for session state variables
- Solution: Always use st.session_state.get() with defaults instead of direct access.

## LLM Integration Issues

### Ollama not connected
- Symptom: "LLM not available" messages
- Solution: The app works fully without LLM. Install Ollama and pull a model: `ollama pull deepseek-r1:7b`

### Slow LLM responses
- Symptom: Chat takes > 30 seconds
- Solution: Use a smaller model (7B parameters). Ensure adequate RAM (8GB+ recommended).

## Database Issues

### PostgreSQL connection failed
- Symptom: psycopg2 OperationalError
- Solution: Verify DATABASE_URL is set correctly. Check if the database server is running.

### SQLite file locked
- Symptom: sqlite3.OperationalError: database is locked
- Solution: Ensure only one process writes to the database at a time. Consider switching to PostgreSQL.
