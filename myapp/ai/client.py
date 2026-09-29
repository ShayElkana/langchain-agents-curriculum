"""The only module the rest of the monolith should import from ai/.

Plain Python in, plain domain types out. No AIMessage crosses this line.
"""

from langchain.agents import create_agent

from myapp.ai.graphs.triage import build_triage_graph
from myapp.ai.models import get_chat_model
from myapp.ai.tools.orders import get_order_status
from myapp.domain import TriageResult

_graph = None
_lookup_agent = None


def _triage_graph():
    global _graph
    if _graph is None:
        _graph = build_triage_graph(get_chat_model())
    return _graph


def triage_ticket(ticket_id: int, text: str) -> TriageResult:
    out = _triage_graph().invoke(
        {
            "ticket_text": text,
            "category": "",
            "draft_reply": "",
            "approved": False,
        },
        {
            "configurable": {"thread_id": f"ticket-{ticket_id}"},
            "recursion_limit": 10,
        },
    )
    return TriageResult(
        category=out["category"],
        draft_reply=out.get("draft_reply") or "",
        approved=out.get("approved") or False,
    )


def lookup_order(question: str) -> str:
    """Tiny agent behind the same facade — tools still call services.orders."""
    global _lookup_agent
    if _lookup_agent is None:
        _lookup_agent = create_agent(
            model=get_chat_model(),
            tools=[get_order_status],
            system_prompt="Look up orders with your tool. Be concise.",
        )
    result = _lookup_agent.invoke(
        {"messages": [{"role": "user", "content": question}]},
        {"recursion_limit": 10},
    )
    return result["messages"][-1].text
