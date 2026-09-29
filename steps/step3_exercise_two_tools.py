"""Step 3 EXERCISE — tool selection and generic dispatch (your turn).

Goals:
  A) Add a second tool and watch the model pick the right one per request.
  B) Replace the hardcoded "call get_order_status" with GENERIC dispatch —
     look the tool up by the name the model requested. (This is exactly
     what create_agent's tools node does internally.)
  C) The docstring experiment: once A+B work, replace both docstrings with
     a single meaningless word like "Tool." and re-run. Watch tool
     selection degrade. Docstrings are load-bearing.

Stretch: send "Check the status of orders A123 AND B456" and print how many
entries ai_msg.tool_calls has. (Parallel tool calls — one message, several
requests. Your dispatch loop should already handle it.)

Run with:  uv run python steps/step3_exercise_two_tools.py
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool

load_dotenv()

model = init_chat_model(os.getenv("MODEL", "openai:gpt-5-nano"), timeout=30, max_retries=2)

ORDERS = {
    "A123": {"status": "shipped", "eta": "Thursday"},
    "B456": {"status": "processing", "eta": "next Monday"},
}


@tool
def get_order_status(order_id: str) -> str:
    """Look up the current status and ETA of a customer order by its ID.

    Order IDs are one letter followed by three digits, e.g. 'A123'.
    """
    order = ORDERS.get(order_id)
    if order is None:
        return f"No order found with ID {order_id!r}. IDs look like 'A123'."
    return f"Order {order_id}: {order['status']}, ETA {order['eta']}."


# TODO 1: implement cancel_order(order_id: str) -> str as a @tool.
#   - Write a real docstring (the model chooses tools by reading it!).
#   - If the ID is unknown, return a helpful error MESSAGE (don't raise).
#   - If the order status is "shipped", return a message that it can no
#     longer be cancelled. Otherwise remove it from ORDERS and confirm.
#   Note the design: the "already shipped" rule lives in the TOOL's code,
#   not in a prompt. Hard rules belong in code (remember the step 4
#   curriculum exercise to come).

@tool
def cancel_order(order_id: str) -> str:
    """Cancel a customer order by its ID.

    Order IDs are one letter followed by three digits, e.g. 'A123'.
    """
    order = ORDERS.get(order_id)
    if order is None:
        return f"No order found with ID {order_id!r}. IDs look like 'A123'."

    if order['status'] == 'shipped':
        return f"Order {order_id} has already been shipped and cannot be cancelled."

    del ORDERS[order_id]
    return f"Order {order_id} has been cancelled."

def run_conversation(user_text: str) -> None:
    """One full tool-calling round-trip, printed step by step."""
    print("\n" + "=" * 60)
    print(f"user: {user_text}")
    print("=" * 60)

    # TODO 2: build the pieces:
    #   tools = [get_order_status, cancel_order]
    #   tools_by_name = {t.name: t for t in tools}     <- generic dispatch table
    #   model_with_tools = model.bind_tools(tools)
    #
    tools = [get_order_status, cancel_order]
    tools_by_name = {t.name: t for t in tools}
    model_with_tools = model.bind_tools(tools)
    # TODO 3: the loop:
    #   1. messages = [HumanMessage(user_text)]
    #   2. ai_msg = model_with_tools.invoke(messages); append it
    #   3. for each tc in ai_msg.tool_calls:
    #        - print which tool it chose and with what args
    #        - result = tools_by_name[tc["name"]].invoke(tc["args"])
    #        - append ToolMessage(content=result, tool_call_id=tc["id"])
    #   4. final = model_with_tools.invoke(messages); print final.text
    messages = [HumanMessage(user_text)]
    ai_msg = model_with_tools.invoke(messages)
    messages.append(ai_msg) 
    for tc in ai_msg.tool_calls:
        print(f"Tool {tc['name']} called with args {tc['args']}")
        result = tools_by_name[tc["name"]].invoke(tc["args"])
        messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))
    final = model_with_tools.invoke(messages)
    print(final.text)
    #raise NotImplementedError("Complete TODOs 1-3, then delete this line.")


if __name__ == "__main__":
    # Two requests that should each hit a DIFFERENT tool:
    run_conversation("Where is my order B456?")
    run_conversation("Please cancel my order B456, I ordered by mistake.")
    # And one the tool itself must refuse (shipped):
    run_conversation("Cancel order A123 please.")
