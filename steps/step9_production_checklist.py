"""Step 9 — Production shape: config, caps, fake-model tests, thread identity.

You already have the pieces. This step names the checklist and shows two
practices that belong in a real monolith from day one:

  1. Deterministic tests with a fake chat model — CI must not call OpenAI.
  2. thread_id derived from a domain primary key, plus recursion_limit
     on every invoke.

The rest of the checklist (observability, PostgresSaver, HITL, evals,
background jobs) is in the comments at the bottom — implement when the
toy grows up, not as extra magic in this file.

Run with:  uv run python steps/step9_production_checklist.py
           uv run pytest tests/test_fake_tool_loop.py -q
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, ToolMessage

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from myapp.ai.testing import ScriptedChatModel  # noqa: E402
from myapp.domain import TriageResult  # noqa: E402


@tool
def get_order_status(order_id: str) -> str:
    """Look up an order. Used only to prove the fake model can request it."""
    return f"Order {order_id}: shipped."


def part_1_fake_model_runs_the_real_loop() -> None:
    """The agent runtime is real. The LLM is a scripted iterator."""
    print("=" * 60)
    print("PART 1: fake model + real create_agent loop")
    print("=" * 60)

    fake = ScriptedChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_order_status",
                            "args": {"order_id": "A123"},
                            "id": "call_fake_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="Order A123 has shipped."),
            ]
        )
    )
    agent = create_agent(
        model=fake,
        tools=[get_order_status],
        system_prompt="Use tools.",
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Where is A123?"}]},
        {"recursion_limit": 10},
    )
    names = [type(m).__name__ for m in result["messages"]]
    print("message types:", names)
    print("final:", result["messages"][-1].content)
    assert any(isinstance(m, ToolMessage) for m in result["messages"])
    print("\n^ This test is free, deterministic, and belongs in CI.")


def part_2_thread_id_is_a_domain_key() -> None:
    print("\n" + "=" * 60)
    print("PART 2: thread_id from a primary key")
    print("=" * 60)
    ticket_id = 4812
    thread_id = f"ticket-{ticket_id}"
    print(f"  ticket_id={ticket_id} → thread_id={thread_id!r}")
    print("  Store that mapping in YOUR tables. Retention/PII rules apply.")
    print("  Recursion cap lives next to invoke, not as a global maybe.")


def part_3_checklist() -> None:
    print("\n" + "=" * 60)
    print("PART 3: production checklist (wire these as the toy becomes real)")
    print("=" * 60)
    items = [
        "Facade: domain types only (Step 8).",
        "Right layer: structured output < workflow < create_agent < deep agent.",
        "Tools = service adapters; mutations idempotent; errors as messages.",
        "Checkpointer: PostgresSaver in prod, thread_id = domain id, retention.",
        "HITL interrupt before consequential writes (Step 6).",
        "timeout + max_retries on the model; recursion_limit on every invoke.",
        "Long graphs: background job, not a 30s HTTP handler.",
        "Tracing: LangSmith/Langfuse; join run id to app logs.",
        "Tests: unit tools/routers, fake-model loop (this file), nightly evals.",
        "Model name from config (MODEL env). Prompts reviewed like code.",
        "Treat model output as untrusted input. Caps in code, not only prompts.",
    ]
    for i, item in enumerate(items, 1):
        print(f"  {i:2}. {item}")
    print(f"\n  MODEL currently: {os.getenv('MODEL', 'openai:gpt-5-nano')}")
    print(f"  TriageResult is importable: {TriageResult.__name__}")


if __name__ == "__main__":
    part_1_fake_model_runs_the_real_loop()
    part_2_thread_id_is_a_domain_key()
    part_3_checklist()
