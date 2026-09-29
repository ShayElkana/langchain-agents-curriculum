"""Step 9 EXERCISE — a CI test that never needs an API key.

Goals:
  A) Make tests/test_fake_tool_loop.py pass (already written):
         uv run pytest tests/test_fake_tool_loop.py -q
  B) Add tests/test_route_ticket.py that imports
         route_ticket from myapp.ai.graphs.triage
     and asserts a hand-built state with category "other" routes to "other".
     No model involved.
  C) Add tests/test_recursion_limit.py: use ScriptedChatModel from
     myapp.ai.testing that ALWAYS emits the same get_order_status tool call,
     create_agent with recursion_limit=4, and assert
     langgraph.errors.GraphRecursionError.

Run with:  uv run pytest tests -q
"""

def main() -> None:
    print(__doc__)


if __name__ == "__main__":
    main()
