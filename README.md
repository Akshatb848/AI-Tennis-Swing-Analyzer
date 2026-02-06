# 🔬 Data Science Agent Platform

An AI-powered multi-agent system that works like an expert data scientist.

## 🌟 Features

- **Multi-Agent Architecture**: Coordinator, Data Cleaner, EDA, Feature Engineer, Model Trainer, AutoML agents
- **Any Data Type Support**: Tabular, text, time-series
- **Automated Analysis**: Data cleaning, EDA, feature engineering, model training
- **Interactive Dashboard**: Professional visualizations

## 🚀 Quick Start

### Local Installation

```bash
# Clone the repo
git clone https://github.com/Akshatb848/data-science-agent-platform.git
cd data-science-agent-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Usage

1. **Step 1**: Create or select a project
   - `"create project"` or `"make new project"`

2. **Step 2**: Upload your dataset
   - `"upload dataset"` or use sample data

3. **Step 3**: Run analysis
   - `"proceed"` or `"start analysis"`

## 🏗️ Architecture

```
├── agents/
│   ├── base_agent.py           # Base agent class
│   ├── coordinator_agent.py    # Master orchestrator
│   ├── data_cleaner_agent.py   # Data preprocessing
│   ├── eda_agent.py            # Exploratory analysis
│   ├── feature_engineer_agent.py
│   ├── model_trainer_agent.py
│   ├── automl_agent.py
│   ├── dashboard_builder_agent.py
│   └── data_visualizer_agent.py
├── utils/
│   └── helpers.py
├── app.py                      # Main Streamlit app
├── requirements.txt
└── README.md
```

## 🤖 Supported Models

### Classification
- Logistic Regression, Random Forest, Gradient Boosting
- Decision Tree, KNN, Naive Bayes

### Regression
- Linear Regression, Ridge, Lasso
- Random Forest, Gradient Boosting, Decision Tree

## 📊 Sample Datasets

- Iris (Classification)
- Housing (Regression)
- Titanic (Classification)
- Random Data

## 📝 License

MIT License

## 👤 Author

Akshat Banga - [@Akshatb848](https://github.com/Akshatb848)
