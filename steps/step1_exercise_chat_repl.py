"""Step 1 EXERCISE — build a terminal chat REPL (your turn).

Goal: a loop that reads your input, keeps the full message history, and
prints CUMULATIVE token usage after each turn. Watch the input-token number
climb every turn — internalize *why* (you re-send the whole history).

Run with:  uv run python steps/step1_exercise_chat_repl.py
Type 'quit' to exit.

Fill in the three TODOs. Peek at step1_simple_llm_call.py part 3 if stuck.
"""

import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage

load_dotenv()

model = init_chat_model(os.getenv("MODEL", "openai:gpt-5-nano"), timeout=30, max_retries=2)


def main() -> None:
    history = [SystemMessage("You are a helpful, concise assistant.")]
    total_input_tokens = 0
    total_output_tokens = 0

    while True:
        user_input = input("\nyou> ").strip()
        if user_input.lower() in {"quit", "exit"}:
            break
        if not user_input:
            continue

        # TODO 1: append the user's message to `history`.
        history.append(HumanMessage(user_input))
        # TODO 2: call the model with the FULL history and append its
        #         reply (the AIMessage itself) back onto `history`.
        assistant = model.invoke(history)   
        history.append(assistant)
        # TODO 3: print the reply text, then update and print the running
        #         token totals using reply.usage_metadata
        #         (keys: "input_tokens", "output_tokens").
        print(f"\nWith history re-sent:\n{assistant.content}")
        usage = assistant.usage_metadata
        total_input_tokens += usage["input_tokens"]
        total_output_tokens += usage["output_tokens"]
        print(f"[this turn: in={usage['input_tokens']} out={usage['output_tokens']} | "
            f"cumulative: in={total_input_tokens} out={total_output_tokens}]")
        #raise NotImplementedError("Complete TODOs 1-3, then delete this line.")


if __name__ == "__main__":
    main()
