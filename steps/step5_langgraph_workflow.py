"""Step 5 — A LangGraph WORKFLOW: you decide the path, the model fills steps.

Step 4's agent: the MODEL chooses which tool to call next.
This step's workflow: YOUR CODE chooses the next node. The model only
does work *inside* a node (classify, draft). Mixing the two is the
production pattern: rails where the process is known, autonomy where it isn't.

Mental model: an agent is a taxi (driver picks the route). A workflow is
a train (you laid the tracks). Take the train when you know the process.

The four lessons:
  1. State is a TypedDict that flows through every node.
  2. A node is (state) -> partial update. LangGraph merges the keys you return.
  3. Conditional edges are plain Python you can unit-test with no LLM.
  4. Nodes do not have to call a model — validators, DB lookups, and
     policy checks belong between LLM steps.

Run with:  uv run python steps/step5_langgraph_workflow.py
"""

import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

MODEL = os.getenv("MODEL", "openai:gpt-5-nano")
model = init_chat_model(MODEL, timeout=30, max_retries=2)


class TicketState(TypedDict):
    ticket_text: str
    category: str
    draft_reply: str
    approved: bool


class Classification(BaseModel):
    category: Literal["billing", "bug", "other"] = Field(
        description="The single best-fitting category."
    )


def classify(state: TicketState) -> dict:
    """LLM node: fill in category from the ticket text."""
    result = model.with_structured_output(Classification).invoke(
        f"Classify this support ticket:\n\n{state['ticket_text']}"
    )
    return {"category": result.category}


def draft_billing_reply(state: TicketState) -> dict:
    reply = model.invoke(
        "Draft a short, empathetic reply to this billing issue. "
        f"Do not promise a refund amount.\n\n{state['ticket_text']}"
    )
    return {"draft_reply": reply.text}


def draft_bug_reply(state: TicketState) -> dict:
    reply = model.invoke(
        "Draft a short reply acknowledging this bug and asking for "
        f"repro steps.\n\n{state['ticket_text']}"
    )
    return {"draft_reply": reply.text}


def validate(state: TicketState) -> dict:
    """Deterministic node — no LLM. Policy lives in code."""
    reply = state["draft_reply"]
    ok = len(reply) < 1500 and "guarantee" not in reply.lower()
    return {"approved": ok}


def route_by_category(state: TicketState) -> str:
    """Return the NAME of the next node. This is a unit-testable function."""
    return state["category"]


def build_graph():
    builder = StateGraph(TicketState)
    builder.add_node("classify", classify)
    builder.add_node("billing", draft_billing_reply)
    builder.add_node("bug", draft_bug_reply)
    builder.add_node("validate", validate)

    builder.add_edge(START, "classify")
    builder.add_conditional_edges(
        "classify",
        route_by_category,
        {"billing": "billing", "bug": "bug", "other": END},
    )
    builder.add_edge("billing", "validate")
    builder.add_edge("bug", "validate")
    builder.add_edge("validate", END)
    return builder.compile()


def main() -> None:
    graph = build_graph()
    print("nodes:", list(graph.nodes))
    print("This is a CompiledStateGraph — same type as create_agent.\n")

    tickets = [
        "I was charged twice this month and I want it fixed!",
        "The export button does nothing when I click it.",
        "What are your office hours?",
    ]
    for text in tickets:
        out = graph.invoke(
            {
                "ticket_text": text,
                "category": "",
                "draft_reply": "",
                "approved": False,
            },
            {"recursion_limit": 10},
        )
        print("=" * 60)
        print(f"ticket:   {text}")
        print(f"category: {out['category']}")
        print(f"approved: {out['approved']}")
        if out["draft_reply"]:
            preview = out["draft_reply"][:180].replace("\n", " ")
            print(f"draft:    {preview}...")
        else:
            print("draft:    (none — routed to END)")


if __name__ == "__main__":
    main()
