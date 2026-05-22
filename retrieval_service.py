from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.prebuilt import create_react_agent

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")
RAG_TOP_K = int(os.getenv("RAG_TOP_K"))


def build_model():
    model_name = os.getenv("GEMINI_MODEL")
    api_key = os.getenv("GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(model=model_name, temperature=0, api_key=api_key)


def build_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def load_vector_db(db_path: str = CHROMA_DB_PATH):
    return Chroma(persist_directory=db_path, embedding_function=build_embeddings())


def retrieve_docs(query: str, db_path: str = CHROMA_DB_PATH):
    vector_db = load_vector_db(db_path)
    return vector_db.similarity_search(query, k=RAG_TOP_K)


def format_context(docs) -> str:
    if not docs:
        return "No relevant policy context was found in the knowledge base."

    lines = []
    for index, doc in enumerate(docs, start=1):
        metadata = doc.metadata or {}
        source = metadata.get("source", "unknown")
        page = metadata.get("page_number", metadata.get("page", "unknown"))
        lines.append(f"[{index}] Source: {source} | Page: {page}\n{doc.page_content.strip()}")

    return "\n\n".join(lines)


def content_to_text(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text", ""))
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(text)
                else:
                    parts.append(str(item))
        return " ".join(part for part in parts if part).strip()
    return str(content).strip()


@tool("policy_retrieval")
def policy_retrieval(query: str) -> str:
    docs = retrieve_docs(query)
    return format_context(docs)


def build_policy_agent():
    prompt = (
        "You are a policy assistant. For every policy question, call the policy_retrieval tool first. "
        "Use the retrieved context to answer concisely and mention source numbers when useful."
    )
    return create_react_agent(build_model(), tools=[policy_retrieval], prompt=prompt)

def answer_policy_query(query: str) -> str:
    agent = build_policy_agent()
    result = agent.invoke({"messages": [HumanMessage(content=query)]})
    last_message = result["messages"][-1]
    return content_to_text(getattr(last_message, "content", last_message))