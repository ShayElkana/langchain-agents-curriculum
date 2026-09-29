"""Step 4 — A simple agent: the Step 3 loop, automated.

In Step 3 you wrote this by hand:
    call model → if it requested tools, run them → send results back → repeat

That loop IS an agent. create_agent is a hardened version of the same
thing: keep going until the model answers with plain text (no tool_calls).

The defining trait: the MODEL chooses the path at runtime. You define
the track (the tools + the system prompt). Nobody writes the sequence
"check status, then carrier, then open a claim" — the model does.

The four lessons:
  1. create_agent == your Step 3 loop, plus "keep going until done".
  2. The message trail is the audit log. Always read it.
  3. Always set a recursion/step cap so a confused agent cannot loop forever.
  4. create_agent is a LangGraph graph under the hood — later steps
     (checkpoints, interrupts) work on it for that reason.

Run with:  uv run python steps/step4_simple_agent.py
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

MODEL = os.getenv("MODEL", "openai:gpt-5-nano")


@tool
def get_order_status(order_id: str) -> str:
    """Look up the current status of an order by its ID."""
    return f"Order {order_id}: delayed at carrier since Monday."


@tool
def get_carrier_info(order_id: str) -> str:
    """Get detailed carrier tracking information for an order."""
    return "Carrier: FedEx. Last scan 5 days ago. Status: possibly lost."


@tool
def open_lost_package_claim(order_id: str) -> str:
    """Open a lost-package claim. Only when carrier data suggests the package is lost."""
    return f"Claim CLM-789 opened for {order_id}."


def print_trail(messages: list) -> None:
    """Print every step so the model's chosen path is visible."""
    print("\n--- message trail ---")
    for m in messages:
        kind = type(m).__name__
        if getattr(m, "tool_calls", None):
            calls = ", ".join(f"{c['name']}({c['args']})" for c in m.tool_calls)
            print(f"  {kind}: requested {calls}")
        else:
            text = (m.text if hasattr(m, "text") else str(m.content)).strip()
            preview = text if len(text) < 160 else text[:157] + "..."
            print(f"  {kind}: {preview}")
    print("--- end trail ---\n")


def part_1_the_model_chooses_the_path() -> None:
    """Nobody wrote the sequence. The model picked it from the tools."""
    print("=" * 60)
    print("PART 1: the model steers; you defined the track")
    print("=" * 60)

    agent = create_agent(
        model=MODEL,
        tools=[get_order_status, get_carrier_info, open_lost_package_claim],
        system_prompt=(
            "You are a support agent for Acme. Investigate order issues using "
            "your tools before answering. Only open claims when the carrier "
            "data suggests the package is lost."
        ),
    )

    # recursion_limit caps graph steps (model call + tool batch ≈ a few
    # steps each). Without it, a confused agent can loop until the bill
    # hurts. Always set one.
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Order A123 hasn't arrived, help!"}]},
        {"recursion_limit": 25},
    )
    print_trail(result["messages"])
    print("final:", result["messages"][-1].text)
    print(
        "\n^ Typical path: status → carrier → claim → answer. "
        "That sequence is not in your code."
    )


def part_2_it_is_a_graph() -> None:
    """create_agent compiles a tiny LangGraph: model node + tools node."""
    print("\n" + "=" * 60)
    print("PART 2: under the hood it is a graph")
    print("=" * 60)

    agent = create_agent(
        model=MODEL,
        tools=[get_order_status],
        system_prompt="You are a support agent. Use tools before answering.",
    )
    print(f"compiled type: {type(agent).__name__}")
    print(f"nodes:         {list(agent.nodes)}")
    print(
        "\nConditional edge: if the model message has tool_calls → tools; "
        "else → end. That is the loop you wrote in Step 3."
    )


if __name__ == "__main__":
    part_1_the_model_chooses_the_path()
    part_2_it_is_a_graph()
