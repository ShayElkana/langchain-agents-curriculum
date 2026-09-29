"""Step 1 — A simple LLM call.

The two lessons of this step:
  1. Everything is *messages* — a call is a list of messages in, one AIMessage out.
  2. The model is *stateless* — it remembers nothing between calls. "Memory"
     is always you re-sending the history.

Run with:  uv run python steps/step1_simple_llm_call.py
Requires:  OPENAI_API_KEY in a .env file (see .env.example)
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage

# Load OPENAI_API_KEY (and optionally MODEL) from the .env file.
load_dotenv()

# init_chat_model is the provider-agnostic factory: "openai:gpt-5-nano" picks
# the provider package (langchain-openai) and the model. Swapping providers
# later is a one-string change. Model name is config, not code — so we read
# it from the environment with a cheap default for learning.
MODEL_NAME = os.getenv("MODEL", "openai:gpt-5-nano")

# timeout and max_retries are production hygiene: never call a network
# service without them. max_retries handles transient errors (rate limits).
model = init_chat_model(MODEL_NAME, timeout=30, max_retries=2)


def part_1_single_call() -> None:
    """One request, one response. The atom everything else is built from."""
    print("=" * 60)
    print("PART 1: a single call")
    print("=" * 60)

    response = model.invoke(
        [
            # SystemMessage frames the model's behavior for the whole call.
            SystemMessage("You are a concise assistant for the engineering team at Acme."),
            # HumanMessage is the user's turn.
            HumanMessage("Explain idempotency in one paragraph."),
        ]
    )

    # .invoke() returned an AIMessage. .content is the text ...
    print(f"\n{response.content}\n")
    # ... and usage_metadata is your token counts — i.e. your COST.
    # input_tokens = everything you sent; output_tokens = what it generated.
    print(f"usage: {response.usage_metadata}")


def part_2_statelessness() -> None:
    """Proof that the model remembers nothing between calls."""
    print("\n" + "=" * 60)
    print("PART 2: the model is stateless")
    print("=" * 60)

    # Call 1: tell it your name.
    model.invoke([HumanMessage("My name is Shay. Remember it!")])

    # Call 2: a brand-new call. The previous call might as well never
    # have happened — nothing links them.
    response = model.invoke([HumanMessage("What's my name?")])
    print(f"\nSecond call, asked 'What's my name?':\n{response.content}")
    print("\n^ It has no idea. Each .invoke() is a blank slate.")


def part_3_memory_is_manual() -> None:
    """The fix: 'memory' means re-sending the whole history every call."""
    print("\n" + "=" * 60)
    print("PART 3: memory = re-sending history")
    print("=" * 60)

    history: list[SystemMessage | HumanMessage | AIMessage] = [
        SystemMessage("You are a concise assistant."),
    ]

    # Turn 1
    history.append(HumanMessage("My name is Shay. Remember it!"))
    reply = model.invoke(history)  # send EVERYTHING so far
    history.append(reply)          # keep the model's answer in the history

    # Turn 2 — the question now travels WITH the earlier turns.
    history.append(HumanMessage("What's my name?"))
    reply = model.invoke(history)
    print(f"\nWith history re-sent:\n{reply.content}")

    print(f"\ninput tokens this turn: {reply.usage_metadata['input_tokens']}")
    print("^ Note: we PAID to re-send the old turns. History growth = cost growth.")
    print("  Every chat product you've ever used works exactly this way.")


if __name__ == "__main__":
    part_1_single_call()
    part_2_statelessness()
    part_3_memory_is_manual()
