import streamlit as st
from agents.coordinator import CoordinatorAgent
from agents.agent7_dashboard import DashboardAgent
from streamlit_option_menu import option_menu
import plotly.io as pio


st.set_page_config(page_title="Expert Data Science Agent", layout="wide")
st.title("🧑‍🔬 Expert Multi-Agent Data Scientist")

# Chat-style flow
if "step" not in st.session_state:
    st.session_state.step = 1
    st.session_state.project_id = None
    st.session_state.dataset = None
    st.session_state.task = ""

chat_input = st.chat_input("Enter your response...")

if chat_input:
    if st.session_state.step == 1:
        if "create" in chat_input.lower() or "new" in chat_input.lower():
            st.session_state.project_id = chat_input.split()[-1] if len(chat_input.split()) > 1 else "new_project"
            st.chat_message("assistant").write(f"Project created: {st.session_state.project_id}")
            st.session_state.step = 2
        else:
            st.chat_message("assistant").write("Step 1: Say 'create project [ID]' or 'new project'.")

    elif st.session_state.step == 2:
        # Dataset: Handle upload or creation
        uploaded = st.file_uploader("Upload dataset (CSV/Excel)", key="dataset_uploader")
        if uploaded:
            if uploaded.name.endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.session_state.dataset = df
            st.chat_message("assistant").write("Dataset uploaded!")
            st.session_state.step = 3
        elif "create dataset" in chat_input.lower():
            # Simple creation (e.g., sample)
            st.session_state.dataset = pd.DataFrame({"col1": [1,2], "col2": [3,4]})
            st.chat_message("assistant").write("Sample dataset created!")
            st.session_state.step = 3
        else:
            st.chat_message("assistant").write("Step 2: Upload dataset or say 'create dataset'.")

    elif st.session_state.step == 3:
        st.session_state.task = chat_input
        st.chat_message("assistant").write("Proceeding with task...")
        coord = CoordinatorAgent(st.session_state.project_id)
        proposal = coord.run_pipeline(st.session_state.task)  # Dynamic run
        st.success(f"Model Proposal: {proposal}")

        if st.button("Load Dashboard"):
            dashboard = DashboardAgent(st.session_state.project_id).run()
            st.session_state["dashboard"] = dashboard

if "dashboard" in st.session_state:
    data = st.session_state["dashboard"]
    with st.sidebar:
        selected = option_menu("Navigation", ["Overview", "EDA", "Features", "Models", "Insights"])

    if selected == "Overview":
        st.header("Project Overview")
        AgGrid(data["project"], enable_enterprise_modules=True)  # Interactive table

    if selected == "EDA":
        st.header("Exploratory Data Analysis")
        # Load plots from artifacts
        hist_html = pio.read_json("artifacts/test_project/eda/hist.html")  # Adjust path
        st.plotly_chart(hist_html)

    # Similarly for others: Use AgGrid for filters/drills, Plotly for viz
    if selected == "Models":
        st.header("Best Model Results")
        AgGrid(data["models"])
        st.image("artifacts/test_project/shap.png")  # SHAP plot
