from __future__ import annotations
import os
import sys
from typing import TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from retrieval_service import answer_policy_query


load_dotenv()

PROMPT = """You are a strict router and assistant.

Return only structured output.

Task:
- Set is_policy to true only if the query is about e-commerce policy topics such as return, refund, cancellation, shipping policy, platform fees, extra charges, warranty, or similar policy rules.
- Set is_policy to false for general questions, coding questions, greetings, explanations, or anything not about e-commerce policy.
- If is_policy is true, set response to an empty string.
- If is_policy is false, write a short helpful answer in response.
"""


class Decision(BaseModel):
    is_policy: bool = Field(description="True when the query is about e-commerce policy.")
    response: str = Field(default="", description="Answer only when the query is not a policy question.")


class QueryResult(TypedDict):
    is_policy: bool
    response: str
    mode: str


def build_model() -> ChatGoogleGenerativeAI:
    model_name = os.getenv("GEMINI_MODEL")
    api_key = os.getenv("GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(model=model_name, temperature=0, api_key=api_key)


def classify_query(query: str) -> Decision:
    model = build_model().with_structured_output(Decision)
    return model.invoke(
        [
            SystemMessage(content=PROMPT),
            HumanMessage(content=query),
        ]
    )


def build_direct_response(query: str) -> str:
    response = build_model().invoke(
        [
            SystemMessage(content="You are a helpful assistant. Reply briefly and clearly."),
            HumanMessage(content=query),
        ]
    )
    return response.content.strip()


class AgentState(TypedDict, total=False):
    query: str
    is_policy: bool
    response: str


def route_node(state: AgentState) -> AgentState:
    decision = classify_query(state["query"])
    return {"is_policy": decision.is_policy}


def direct_node(state: AgentState) -> AgentState:
    return {"response": build_direct_response(state["query"])}


def rag_node(state: AgentState) -> AgentState:
    return {"response": answer_policy_query(state["query"])}


def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("route", route_node)
    graph.add_node("direct", direct_node)
    graph.add_node("rag", rag_node)
    graph.add_edge(START, "route")
    graph.add_conditional_edges(
        "route",
        lambda state: "rag" if state.get("is_policy") else "direct",
        {"rag": "rag", "direct": "direct"},
    )
    graph.add_edge("direct", END)
    graph.add_edge("rag", END)
    return graph.compile()


def handle_query(query: str) -> QueryResult:
    result = build_agent().invoke({"query": query})
    return {
        "is_policy": bool(result.get("is_policy")),
        "response": result.get("response", ""),
        "mode": "rag" if result.get("is_policy") else "direct",
    }


def run(query: str) -> str:
    return handle_query(query)["response"]

