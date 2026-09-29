"""Step 8 — Put the AI layer behind a facade in an existing monolith.

The rest of the company should call:

    from myapp.ai.client import triage_ticket
    result = triage_ticket(ticket_id=4812, text=raw)

and get a TriageResult — not an AIMessage, not a LangGraph state dict.

Two boundary rules:
  1. Inward: LangChain types stay inside myapp/ai/.
  2. Outward: tools call myapp.services.*, so authz, audit, and validation
     are the same as any other caller. The agent is an untrusted client.

This repo's toy layout:

    myapp/domain.py           # plain result types
    myapp/services/orders.py  # existing business logic — no langchain
    myapp/ai/client.py        # THE facade
    myapp/ai/models.py        # model factory (config-driven)
    myapp/ai/graphs/          # workflows
    myapp/ai/tools/           # adapters

Run with:  uv run python steps/step8_monolith_facade.py
Then:      rg langchain myapp --glob '*.py'
            (hits should live only under myapp/ai/)
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from myapp.ai.client import lookup_order, triage_ticket
from myapp.domain import TriageResult
from myapp.services.orders import get_order


def main() -> None:
    print("=" * 60)
    print("Service layer still works with no AI involved")
    print("=" * 60)
    order = get_order("A123")
    print(f"  {order}")

    print("\n" + "=" * 60)
    print("Facade: triage_ticket → TriageResult")
    print("=" * 60)
    result = triage_ticket(4812, "I was charged twice this month!")
    print(f"  type:     {type(result).__name__}")
    print(f"  isinstance TriageResult: {isinstance(result, TriageResult)}")
    print(f"  category: {result.category}")
    print(f"  approved: {result.approved}")
    print(f"  draft:    {result.draft_reply[:160].replace(chr(10), ' ')}...")

    print("\n" + "=" * 60)
    print("Facade: lookup_order → str (tool hit the SERVICE, not a raw dict)")
    print("=" * 60)
    print(" ", lookup_order("Where is order B456?"))

    print("\n^ Call sites in api/ or jobs/ look like any other service call.")


if __name__ == "__main__":
    main()
