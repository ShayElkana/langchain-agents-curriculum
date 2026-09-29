"""Step 6 — Checkpoints and human-in-the-loop.

Two new capabilities, one mechanism: a checkpointer.

A checkpointer snapshots graph state after each superstep, keyed by
thread_id. That single feature buys you:
  1. Multi-turn memory without a manual history list (Step 1's lesson,
     now the runtime's job).
  2. Pause for a human, let the process die, resume days later on a
     different worker — same thread_id.

Mental model: a save-game. Every step auto-saves. Load slot = thread_id.

The four lessons:
  1. thread_id is the session name. Map it to a domain entity
     (ticket-4812), never reuse it across unrelated conversations.
  2. Without a checkpointer, thread_id does nothing useful.
  3. interrupt(payload) freezes the run; Command(resume=...) continues it.
  4. Code BEFORE interrupt() in that node/tool re-runs on resume —
     keep side effects AFTER the interrupt, or make them idempotent.

Run with:  uv run python steps/step6_checkpoints_and_hitl.py
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

load_dotenv()

MODEL = os.getenv("MODEL", "openai:gpt-5-nano")


def part_1_memory_is_the_checkpointer() -> None:
    """Same agent, two turns, no manual history list."""
    print("=" * 60)
    print("PART 1: multi-turn memory via thread_id")
    print("=" * 60)

    agent = create_agent(
        model=MODEL,
        tools=[],
        system_prompt="You are concise. Remember facts the user tells you in this thread.",
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "user-shay"}, "recursion_limit": 10}

    agent.invoke(
        {"messages": [{"role": "user", "content": "My name is Shay. I work at Acme."}]},
        config,
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is my name and where do I work?"}]},
        config,
    )
    print("turn 2:", result["messages"][-1].text)
    print(
        "\n^ The second invoke did not re-send turn 1. The checkpointer did.\n"
        "  A different thread_id would be a blank slate."
    )


@tool
def issue_refund(order_id: str, amount: float) -> str:
    """Issue a customer refund. Requires a human to approve first."""
    decision = interrupt(
        {"action": "refund", "order_id": order_id, "amount": amount}
    )
    if decision != "approve":
        return f"Refund of ${amount} for {order_id} was rejected by a reviewer."
    return f"Refunded ${amount} for order {order_id}."


def part_2_human_in_the_loop() -> None:
    """The run pauses inside the tool until you resume it."""
    print("\n" + "=" * 60)
    print("PART 2: interrupt → human → Command(resume=...)")
    print("=" * 60)

    agent = create_agent(
        model=MODEL,
        tools=[issue_refund],
        system_prompt=(
            "You are a support agent. If the customer asks for a refund, "
            "call issue_refund. Do not invent a refund without the tool."
        ),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "ticket-4812"}, "recursion_limit": 15}

    paused = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Please refund $49.99 on order A123. I was double-charged.",
                }
            ]
        },
        config,
    )
    interrupts = paused.get("__interrupt__")
    print("paused. interrupts:", interrupts)

    resumed = agent.invoke(Command(resume="approve"), config)
    print("after approve:", resumed["messages"][-1].text)
    print(
        "\n^ In production the gap between those two invokes is a ticket in "
        "your admin UI, and the second invoke may run tomorrow on another pod."
    )


if __name__ == "__main__":
    part_1_memory_is_the_checkpointer()
    part_2_human_in_the_loop()
