from langchain_groq import ChatGroq
import streamlit as st

def get_llm():
    return ChatGroq(
        model="llama-3.1-70b-versatile",  # Fast, capable for planning/proposals
        temperature=0.5,
        api_key=st.secrets["groq"]["api_key"]
    )

def query_llm(prompt: str, system: str = "You are an expert data scientist AI.") -> str:
    llm = get_llm()
    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    response = llm.invoke(messages)
    return response.content
