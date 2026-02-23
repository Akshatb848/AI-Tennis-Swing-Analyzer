# Model Deployment Guide

## Streamlit Community Cloud (Recommended)
- Free unlimited public apps with direct GitHub integration
- 1GB RAM, sufficient for most dashboards
- Steps:
  1. Push code to GitHub repository
  2. Go to share.streamlit.io and connect your repo
  3. Select the main app file (dashboard/app.py)
  4. Set any required secrets in the Streamlit dashboard
  5. Deploy - auto-updates on push to main branch

## Hugging Face Spaces
- Free CPU instances (2 vCPU, 16GB RAM)
- Supports Streamlit and Gradio interfaces
- Steps:
  1. Create a new Space on huggingface.co
  2. Choose Streamlit as the SDK
  3. Push your code to the Space repository
  4. Add requirements.txt with all dependencies
  5. The app deploys automatically

## Replit Deployment
- Built-in deployment with custom domain support
- Steps:
  1. Configure the run command for Streamlit
  2. Set environment variables in Replit Secrets
  3. Click Deploy in the Replit interface

## Production Checklist
- Save models using joblib for sklearn models
- Use torch.save for PyTorch models
- Create a predict.py inference script
- Set up monitoring for model drift
- Log predictions with timestamps
- Implement health check endpoints
- Use environment variables for configuration
- Never hardcode API keys or credentials

## Model Serving Best Practices
- Use pickle/joblib for serialization
- Load models once at startup, not per-request
- Implement input validation
- Add error handling for malformed inputs
- Cache predictions when appropriate
- Monitor memory usage and response times

## Monitoring in Production
- Track prediction distributions over time
- Compute KL divergence or PSI for drift detection
- Alert when drift exceeds threshold (typically > 0.2)
- Log feature distributions alongside predictions
- Retrain when performance degrades significantly
- Keep a record of all model versions and their metrics
