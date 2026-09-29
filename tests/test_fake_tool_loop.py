"""Deterministic agent-loop test — no network, no API key."""

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import AIMessage, ToolMessage

from myapp.ai.testing import ScriptedChatModel


@tool
def get_order_status(order_id: str) -> str:
    """Look up an order by id."""
    return f"Order {order_id}: shipped."


def test_create_agent_executes_scripted_tool_call() -> None:
    fake = ScriptedChatModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "get_order_status",
                            "args": {"order_id": "A123"},
                            "id": "call_1",
                            "type": "tool_call",
                        }
                    ],
                ),
                AIMessage(content="Shipped."),
            ]
        )
    )
    agent = create_agent(model=fake, tools=[get_order_status], system_prompt="Use tools.")
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "status of A123?"}]},
        {"recursion_limit": 10},
    )
    assert any(isinstance(m, ToolMessage) for m in result["messages"])
    assert "Shipped" in result["messages"][-1].content
