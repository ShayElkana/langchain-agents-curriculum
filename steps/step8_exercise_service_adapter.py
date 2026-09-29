"""Step 8 EXERCISE — a new mutating tool that still goes through services.

You will add a tiny notes feature the "legacy" app already hypothetically
owns, then expose it to an agent without letting LangChain into services/.

Goals:
  A) Implement myapp/services/notes.py:
       add_note(order_id: str, body: str) -> str
       Unknown order_id → raise the existing OrderNotFound from orders.py
       (call get_order first). Store notes in a module-level dict.
  B) Implement myapp/ai/tools/notes.py:
       @tool add_order_note(order_id, body) that catches OrderNotFound and
       returns an error MESSAGE. Never raise across the tool boundary.
  C) Add myapp.ai.client.add_note_via_agent(question: str) -> str that
     create_agent's with that tool (and only that tool).
  D) From THIS file, call the facade. Do not import langchain here.
  E) Run:  rg "langchain" myapp --glob "*.py"
     Allowed hits: myapp/ai/** only. services/ and domain.py must be clean.

Run with:  uv run python steps/step8_exercise_service_adapter.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# TODO: after C, this import should work:
# from myapp.ai.client import add_note_via_agent


def main() -> None:
    # TODO: print add_note_via_agent("Add a note on A123: customer called, still waiting.")
    # TODO: print add_note_via_agent("Add a note on Z999: this id is fake.")
    raise NotImplementedError("Complete TODOs A–D, then delete this line.")


if __name__ == "__main__":
    main()
