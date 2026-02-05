# utils/rag_retriever.py

from pathlib import Path

RAG_DOCS_PATH = Path("rag_docs")


#def query_rag(query: str) -> str:
    #"""
    #Very simple RAG retriever (v1):
    #Reads markdown files and returns relevant context.
    #"""

    #if not RAG_DOCS_PATH.exists():
        #return "No RAG documents found."

    #docs = []
    #for md_file in RAG_DOCS_PATH.glob("*.md"):
     #   docs.append(md_file.read_text())

    #context = "\n\n".join(docs)

    #return f"""
#Context:
#{context}

#User Query:
#{query}
#"""

def query_rag(query: str) -> str:
    """
    RAG disabled for stability phase.
    """
    return "RAG disabled (stability mode)"


