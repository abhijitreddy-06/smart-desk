from __future__ import annotations

import os

import streamlit as st
from dotenv import load_dotenv

from agent import handle_query

load_dotenv()

st.set_page_config(page_title="Smart Desk", page_icon="💬", layout="centered")

st.title("Smart Desk")
st.caption("Policy questions go to RAG. Everything else gets a direct LLM reply.")

default_prompt = os.getenv("DEFAULT_USER_QUERY", "Ask a policy question or a general question...")
query = st.text_area("Your question", value="", placeholder=default_prompt, height=140)

if st.button("Ask", type="primary"):
    if not query.strip():
        st.warning("Enter a question first.")
    else:
        try:
            with st.spinner("Thinking..."):
                result = handle_query(query.strip())
        except RuntimeError as exc:
            st.error(str(exc))
            st.info(
                "Add the key in .env, for example:\n"
                "GEMINI_API_KEY=your_key_here\n"
                "GEMINI_MODEL=gemini-1.5-flash"
            )
            st.stop()

        mode_label = "RAG" if result["is_policy"] else "Direct LLM"
        st.success(f"Mode: {mode_label}")
        st.write(result["response"])