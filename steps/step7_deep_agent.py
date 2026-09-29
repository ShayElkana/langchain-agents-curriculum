"""Step 7 — Deep Agents: a harness for long-horizon work.

A plain create_agent degrades on long tasks for one reason: everything
piles into a single message history. The context window fills, the plan
gets buried, quality collapses.

Deep Agents (package: deepagents) wrap create_agent with extra machinery
aimed at that failure mode:
  - a virtual filesystem (ls/read/write/edit) so bulky notes live in
    files, not in the conversation
  - subagents (a `task` tool) so noisy research burns a *child* context
    and only a summary returns to the parent
  - optional todo-list planning

Mental model: a plain agent is a contractor working from memory. A deep
agent is that contractor plus a whiteboard, a filing cabinet, and junior
staff they can delegate to.

When NOT to use this: a 5-step support agent, a classifier, a structured
extraction. Start simpler; move up only when you watch the plan get lost.

This lesson uses a fake "search" tool so you do not need a second API key.
The invoke is deliberately small — deep-agent runs are slower and dearer
than create_agent.

Run with:  uv run python steps/step7_deep_agent.py
"""

import os

from deepagents import create_deep_agent
from dotenv import load_dotenv
from langchain.tools import tool

load_dotenv()

MODEL = os.getenv("MODEL", "openai:gpt-5-nano")

KB = {
    "langgraph": (
        "LangGraph is a durable graph runtime for LLM steps: nodes, edges, "
        "state, checkpoints, interrupts. Best when you need pause/resume."
    ),
    "temporal": (
        "Temporal is a general workflow engine (not LLM-specific): durable "
        "execution, timers, retries, worker fleets. Heavier ops footprint."
    ),
    "create_agent": (
        "create_agent is a small LangGraph: model ⇄ tools until the model "
        "stops calling tools. Right default for short tool-using tasks."
    ),
}


@tool
def knowledge_search(topic: str) -> str:
    """Look up a canned note about an orchestration topic.

    Known topics: langgraph, temporal, create_agent.
    """
    key = topic.strip().lower().replace(" ", "_")
    for k, v in KB.items():
        if k in key or key in k:
            return v
    return f"No note for {topic!r}. Try: langgraph, temporal, create_agent."


def main() -> None:
    researcher = {
        "name": "researcher",
        "description": "Looks up one orchestration topic and returns a short summary.",
        "system_prompt": (
            "You research one topic using knowledge_search. "
            "Return a concise sourced summary, then stop."
        ),
        "tools": [knowledge_search],
    }

    agent = create_deep_agent(
        model=MODEL,
        tools=[knowledge_search],
        system_prompt=(
            "You are a research analyst. For comparison questions: look up "
            "each topic (yourself or via the researcher subagent), write "
            "short notes with your file tools, then put a 5-line comparison "
            "in /comparison.md. Do not run shell commands."
        ),
        subagents=[researcher],
    )
    print("compiled type:", type(agent).__name__)
    print("nodes:", list(agent.nodes))

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Compare LangGraph, Temporal, and create_agent for "
                        "orchestrating LLM work. Write the comparison to "
                        "/comparison.md."
                    ),
                }
            ]
        },
        {"recursion_limit": 40},
    )

    print("\nresult keys:", sorted(result.keys()))
    files = result.get("files") or {}
    if files:
        print("\n--- virtual files ---")
        for path, content in files.items():
            print(f"\n## {path}\n{content[:1500]}")
    else:
        print("\n(no 'files' key — read the trail; the agent may have used ls/read_file)")
        for m in result.get("messages", [])[-4:]:
            kind = type(m).__name__
            text = getattr(m, "text", None) or str(getattr(m, "content", ""))
            print(f"  {kind}: {text[:300]}")


if __name__ == "__main__":
    main()
