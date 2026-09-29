"""LangGraph workflow used by the facade. Callers outside ai/ never import this."""

from typing import Literal, TypedDict

from langchain.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field


class TicketState(TypedDict):
    ticket_text: str
    category: str
    draft_reply: str
    approved: bool


class Classification(BaseModel):
    category: Literal["billing", "bug", "other"] = Field(
        description="Single best-fitting category."
    )


def route_ticket(state: TicketState) -> str:
    """Pure routing — unit-test this with dicts, no model."""
    return state["category"]


def build_triage_graph(model: BaseChatModel):
    def classify(state: TicketState) -> dict:
        result = model.with_structured_output(Classification).invoke(
            f"Classify this support ticket:\n\n{state['ticket_text']}"
        )
        return {"category": result.category}

    def draft_billing(state: TicketState) -> dict:
        reply = model.invoke(
            "Short empathetic billing reply. Do not promise a specific refund.\n\n"
            f"{state['ticket_text']}"
        )
        return {"draft_reply": reply.text}

    def draft_bug(state: TicketState) -> dict:
        reply = model.invoke(
            "Short bug acknowledgement; ask for repro steps.\n\n"
            f"{state['ticket_text']}"
        )
        return {"draft_reply": reply.text}

    def validate(state: TicketState) -> dict:
        reply = state["draft_reply"]
        ok = len(reply) < 1500 and "guarantee" not in reply.lower()
        return {"approved": ok}

    builder = StateGraph(TicketState)
    builder.add_node("classify", classify)
    builder.add_node("billing", draft_billing)
    builder.add_node("bug", draft_bug)
    builder.add_node("validate", validate)
    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        route_ticket,
        {"billing": "billing", "bug": "bug", "other": END},
    )
    builder.add_edge("billing", "validate")
    builder.add_edge("bug", "validate")
    builder.add_edge("validate", END)
    return builder.compile()
