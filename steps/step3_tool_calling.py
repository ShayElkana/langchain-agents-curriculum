"""Step 3 — Tool calling: the model requests, YOUR code executes.

The problem: models can't check your database, your order system, or
today's date. Tools connect language to systems.

The critical insight of this step (worth over-learning):
    The model NEVER executes anything. It emits a structured request —
    "please call get_order_status(order_id='A123')" — and YOUR code
    decides whether to run it, runs it, and sends the result back.

That request/execute/respond loop is built by hand below, once. In step 4,
create_agent automates this exact loop — after this file, it holds no magic.

The four lessons:
  1. @tool turns a function into a schema: docstring = description the
     model reads, type hints = argument schema. Docstrings are load-bearing.
  2. The model's "answer" can be a tool REQUEST instead of text
     (AIMessage.tool_calls).
  3. You execute and reply with a ToolMessage carrying the tool_call_id.
  4. Tools should return error MESSAGES, not raise — a described failure
     lets the model self-correct; an exception kills the run.

Run with:  uv run python steps/step3_tool_calling.py
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, ToolMessage
from langchain.tools import tool

load_dotenv()

model = init_chat_model(os.getenv("MODEL", "openai:gpt-5-nano"), timeout=30, max_retries=2)

# Fake "database" so the tool has something real to do.
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
        # Lesson 4: describe the failure instead of raising. The model can
        # read this, tell the user, or retry with a corrected ID.
        return f"No order found with ID {order_id!r}. IDs look like 'A123'."
    return f"Order {order_id}: {order['status']}, ETA {order['eta']}."


def part_1_what_the_model_sees() -> None:
    """@tool converted your function into name + description + schema."""
    print("=" * 60)
    print("PART 1: what @tool made of your function")
    print("=" * 60)
    print(f"name:        {get_order_status.name}")
    print(f"description: {get_order_status.description}")
    print(f"args schema: {get_order_status.args}")
    print("\n^ This — not your code — is ALL the model ever sees of the tool.")


def part_2_the_model_requests() -> None:
    """The model does not answer. It asks you to run something."""
    print("\n" + "=" * 60)
    print("PART 2: the model emits a REQUEST, not an answer")
    print("=" * 60)

    model_with_tools = model.bind_tools([get_order_status])
    ai_msg = model_with_tools.invoke([HumanMessage("Where is my order A123?")])

    print(f"\ntext content: {ai_msg.text!r}")
    print(f"tool_calls:   {ai_msg.tool_calls}")
    print("\n^ Empty-ish text, populated tool_calls: it wants YOU to act.")


def part_3_the_full_loop() -> None:
    """Execute the request, reply with a ToolMessage, get the real answer."""
    print("\n" + "=" * 60)
    print("PART 3: the full request → execute → respond loop, by hand")
    print("=" * 60)

    model_with_tools = model.bind_tools([get_order_status])
    messages = [HumanMessage("Where is my order A123?")]

    # Round 1: model requests.
    ai_msg = model_with_tools.invoke(messages)
    messages.append(ai_msg)  # the request itself stays in the history

    # WE execute. (It may request several tools at once — hence the loop.)
    for tc in ai_msg.tool_calls:
        print(f"model requested: {tc['name']}({tc['args']})")
        result = get_order_status.invoke(tc["args"])
        print(f"we executed and got: {result!r}")
        # tool_call_id pairs this result with that specific request.
        messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    # Round 2: model now sees the result and can actually answer.
    final = model_with_tools.invoke(messages)
    print(f"\nfinal answer: {final.text}")


def part_4_errors_as_messages() -> None:
    """A described failure lets the model recover gracefully."""
    print("\n" + "=" * 60)
    print("PART 4: tool 'errors' the model can work with")
    print("=" * 60)

    model_with_tools = model.bind_tools([get_order_status])
    messages = [HumanMessage("Where is my order 999?")]  # bad ID format

    ai_msg = model_with_tools.invoke(messages)
    messages.append(ai_msg)
    for tc in ai_msg.tool_calls:
        result = get_order_status.invoke(tc["args"])  # returns the error STRING
        print(f"tool returned: {result!r}")
        messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

    final = model_with_tools.invoke(messages)
    print(f"\nmodel's graceful recovery: {final.text}")
    print("\n^ Had the tool RAISED instead, the whole run would be dead.")


if __name__ == "__main__":
    part_1_what_the_model_sees()
    part_2_the_model_requests()
    part_3_the_full_loop()
    part_4_errors_as_messages()
