"""Step 7 EXERCISE — use the harness, then notice when it is overkill.

Goals:
  A) create_deep_agent with knowledge_search (copy the stub from the lesson
     or write your own 3-entry dict).
  B) Give it one subagent. Ask it to produce /report.md with three bullets.
  C) Print the virtual file if present, else the final message.
  D) Re-run THE SAME prompt with plain create_agent (no filesystem, no
     subagents) and compare: latency, trail length, answer quality.

The comparison is the exercise. On a 3-bullet lookup, create_agent will
often win on speed and cost. That is the correct conclusion — deep agents
earn their keep on long, messy, multi-document work, not on this toy.

Run with:  uv run python steps/step7_exercise_deep_vs_plain.py
"""

import os
import time

from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("MODEL", "openai:gpt-5-nano")
PROMPT = (
    "Write /report.md with exactly three bullets comparing LangGraph, "
    "Temporal, and create_agent. Use your search tool. Be terse."
)


# TODO 1: knowledge_search @tool + tiny KB (or import the lesson's idea).


def run_deep() -> None:
    # TODO 2: create_deep_agent, invoke with recursion_limit, print files
    #         or final message, print elapsed seconds.
    raise NotImplementedError("TODO 2")


def run_plain() -> None:
    # TODO 3: create_agent with the SAME search tool only. Same prompt
    #         (ask for three bullets in the reply, no files). Print elapsed.
    raise NotImplementedError("TODO 3")


if __name__ == "__main__":
    t0 = time.perf_counter()
    run_deep()
    print(f"\ndeep elapsed: {time.perf_counter() - t0:.1f}s")
    t0 = time.perf_counter()
    run_plain()
    print(f"plain elapsed: {time.perf_counter() - t0:.1f}s")
