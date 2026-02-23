# Machine Learning Best Practices

## Data Preparation
- Always perform exploratory data analysis before modeling
- Handle missing values appropriately: use median for numeric, mode for categorical
- Remove or flag outliers using IQR method or isolation forest
- Scale numeric features using StandardScaler for distance-based algorithms
- One-hot encode categorical features with reasonable cardinality (< 10 categories)
- Remove ID columns and high-cardinality text columns before modeling

## Feature Engineering
- Create interaction features for known relationships
- Use PCA for dimensionality reduction when features exceed 50
- Apply feature selection using mutual information or L1 regularization
- Consider target encoding for high-cardinality categoricals
- Handle datetime features by extracting year, month, day, day-of-week

## Model Selection
- For classification: start with LogisticRegression as baseline, then try RandomForest, GradientBoosting, XGBoost, LightGBM
- For regression: start with LinearRegression, then try RandomForest, GradientBoosting, XGBoost, LightGBM
- For clustering: use KMeans with silhouette score, or DBSCAN for non-spherical clusters
- For time series: consider ARIMA, Prophet, or LSTM networks
- Always compare against a simple baseline model

## Hyperparameter Tuning
- Use Optuna for Bayesian hyperparameter optimization
- Limit trials to 30-50 per model to keep runtime reasonable
- Use 5-fold cross-validation during tuning
- Focus on the most impactful hyperparameters first
- Monitor for overfitting: compare train vs validation scores

## Model Evaluation
- Classification metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Regression metrics: RMSE, MAE, R-squared, MAPE
- Always use cross-validation for robust estimates
- Generate confusion matrix for classification problems
- Check feature importance to understand model decisions

## Common Pitfalls
- Data leakage: never use test data for training or feature engineering
- Class imbalance: use SMOTE, class weights, or stratified sampling
- Overfitting: use regularization, early stopping, or simpler models
- Underfitting: try more complex models or add more features
- Feature scaling: tree-based models don't need it, but linear models do
