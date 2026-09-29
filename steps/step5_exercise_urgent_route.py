"""Step 5 EXERCISE — add a branch you own, then unit-test it without an LLM.

Goals:
  A) Extend TicketState with `urgent: bool`.
  B) Classification should also extract urgent (angry customer, outage,
     legal threat, double-charge with chat down, etc.).
  C) Add an `escalate` node that sets draft_reply to a fixed escalation
     template and approved=False (a human will take it). No LLM in escalate.
  D) Change routing: if urgent → escalate, else the existing category map.
     "other" + not urgent still goes to END.
  E) Write test_route() below (plain dicts, no model) proving:
       - urgent billing → "escalate"
       - non-urgent billing → "billing"
       - non-urgent other → "other"

Then run:
  uv run python steps/step5_exercise_urgent_route.py
  uv run pytest steps/step5_exercise_urgent_route.py -q

The pytest run is the point: workflow control flow is testable like any
other Python. Try that with create_agent and you will feel the difference.
"""

from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

# You may use a real model in classify/draft nodes. Routing tests must not.


class TicketState(TypedDict):
    ticket_text: str
    category: str
    draft_reply: str
    approved: bool
    # TODO 1: add urgent: bool


class Classification(BaseModel):
    category: Literal["billing", "bug", "other"]
    # TODO 2: add urgent: bool with a real Field description


# TODO 3: implement classify, draft_billing_reply, escalate, validate
# (copy from the lesson and adapt). escalate must NOT call a model.


def route(state: TicketState) -> str:
    """Return the next node name: escalate | billing | bug | other."""
    # TODO 4: if urgent → "escalate"; else return category
    raise NotImplementedError("TODO 4")


def build_graph():
    # TODO 5: wire START → classify → route → (escalate|billing|bug|END)
    # billing/bug → validate → END; escalate → END
    raise NotImplementedError("TODO 5")


def test_route_urgent_billing_goes_to_escalate() -> None:
    # TODO 6: call route({... urgent True, category billing ...})
    #         and assert it returns "escalate"
    raise NotImplementedError("TODO 6")


def test_route_non_urgent_billing_goes_to_billing() -> None:
    raise NotImplementedError("TODO 6 continued")


def test_route_non_urgent_other_goes_to_other() -> None:
    raise NotImplementedError("TODO 6 continued")


def main() -> None:
    graph = build_graph()
    samples = [
        "URGENT: charged twice AND support chat is down. I will sue.",
        "Small typo on the invoice, no rush.",
        "What time do you close on Friday?",
    ]
    for text in samples:
        out = graph.invoke(
            {
                "ticket_text": text,
                "category": "",
                "draft_reply": "",
                "approved": False,
                "urgent": False,
            },
            {"recursion_limit": 10},
        )
        print("=" * 60)
        print(text)
        print(out)


if __name__ == "__main__":
    main()
