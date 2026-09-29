"""Step 6 EXERCISE — isolation by thread_id, then approve/reject.

Goals:
  A) Build a create_agent with InMemorySaver OR SqliteSaver (see TODO).
  B) Prove thread isolation: tell thread "user-a" your name, ask
     thread "user-b" for that name — B must not know it. Then ask A again.
  C) Refund tool with interrupt(). First invoke pauses. Resume once with
     "approve" and once (a different thread) with "reject". Print both
     tool outcomes.

Sqlite stretch (worth doing): swap InMemorySaver for SqliteSaver, run,
quit the process, run again with the same thread_id, and confirm A still
remembers your name. That is crash-recovery in one file.

    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver
    conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()

Run with:  uv run python steps/step6_exercise_threads_and_resume.py
"""

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt

load_dotenv()

MODEL = os.getenv("MODEL", "openai:gpt-5-nano")


# TODO 1: issue_refund with interrupt(), same pattern as the lesson.


def make_agent():
    # TODO 2: return create_agent(..., checkpointer=InMemorySaver())
    #         (or SqliteSaver — see docstring)
    raise NotImplementedError("TODO 2")


def part_a_thread_isolation() -> None:
    print("=" * 60)
    print("A: two threads must not share memory")
    print("=" * 60)
    # TODO 3:
    #   agent = make_agent()
    #   config_a = {"configurable": {"thread_id": "user-a"}, "recursion_limit": 10}
    #   config_b = {"configurable": {"thread_id": "user-b"}, "recursion_limit": 10}
    #   tell A your name; ask B "what is my name?"; ask A again.
    raise NotImplementedError("TODO 3")


def part_b_approve_and_reject() -> None:
    print("\n" + "=" * 60)
    print("B: resume approve vs reject")
    print("=" * 60)
    # TODO 4: two thread_ids (ticket-ok, ticket-no).
    #   Each asks for a refund. Resume one with "approve", one with "reject".
    raise NotImplementedError("TODO 4")


if __name__ == "__main__":
    part_a_thread_isolation()
    part_b_approve_and_reject()
