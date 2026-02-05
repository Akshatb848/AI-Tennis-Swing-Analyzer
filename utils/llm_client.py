from langchain_groq import ChatGroq
import streamlit as st

# utils/llm_client.py

def query_llm(prompt: str) -> str:
    """
    LLM disabled for stability phase.
    This function is a safe stub.
    """
    return "LLM disabled (stability mode)"


#def query_llm(prompt: str, system: str = "You are an expert data scientist AI.") -> str:
    #llm = get_llm()
    #messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
    #response = llm.invoke(messages)
    #return response.content
