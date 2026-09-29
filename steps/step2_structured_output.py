"""Step 2 — Structured output.

The problem: a model's native output is prose, and production code cannot
branch on prose. The solution: hand the model a schema (a Pydantic class)
and get back a *validated Python object* instead of text.

The three lessons of this step:
  1. A Pydantic class + with_structured_output() = typed, validated results.
  2. Field descriptions and Literal types ARE prompt engineering — the model
     reads them. Write them like docs for a new hire.
  3. Validated is not the same as correct. The shape is guaranteed;
     the judgment inside it is not.

Run with:  uv run python steps/step2_structured_output.py
"""

import json
import os
from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

load_dotenv()

model = init_chat_model(os.getenv("MODEL", "openai:gpt-5-nano"), timeout=30, max_retries=2)


# ---------------------------------------------------------------------------
# The schema. Three things to notice:
#   - The docstring and every Field(description=...) are sent to the model.
#     They are instructions, not comments.
#   - Literal[...] doesn't just validate the category — it CONSTRAINS the
#     model's choices. It cannot answer "billing-ish".
#   - ge=1, le=5 gives Pydantic something to reject if the model misbehaves.
# ---------------------------------------------------------------------------
class TicketTriage(BaseModel):
    """Triage assessment of a customer support ticket."""

    category: Literal["billing", "bug", "feature_request", "other"] = Field(
        description="The single best-fitting category for the ticket."
    )
    severity: int = Field(
        ge=1, le=5,
        description="Impact on the customer: 1=trivial, 3=degraded, 5=critical outage.",
    )
    summary: str = Field(description="One-sentence summary in neutral tone.")
    needs_human: bool = Field(
        description="True if a human should review before any reply is sent "
        "(angry customer, refund request, legal threat, security issue)."
    )


TICKET = (
    "Subject: CHARGED TWICE?!?!\n\n"
    "I just checked my statement and you charged me twice this month, "
    "$49.99 on the 1st AND the 14th. Also your support chat widget won't "
    "even load so I couldn't reach anyone. Fix this or I'm cancelling."
)


def part_1_basic_triage() -> None:
    """One call: messy human text in, validated typed object out."""
    print("=" * 60)
    print("PART 1: text in, typed object out")
    print("=" * 60)

    # .with_structured_output() returns a NEW runnable whose .invoke()
    # returns a TicketTriage instance instead of an AIMessage.
    triager = model.with_structured_output(TicketTriage)
    result = triager.invoke(f"Triage this support ticket:\n\n{TICKET}")

    # This is a real Pydantic object. Your code can branch on it safely.
    print(f"\ntype:        {type(result).__name__}")
    print(f"category:    {result.category}")
    print(f"severity:    {result.severity}")
    print(f"summary:     {result.summary}")
    print(f"needs_human: {result.needs_human}")

    # ...and this is the production point: no regex, no string parsing.
    if result.needs_human or result.severity >= 4:
        print("\n→ routed to human review queue")


def part_2_under_the_hood() -> None:
    """What actually gets sent: your class becomes a JSON Schema."""
    print("\n" + "=" * 60)
    print("PART 2: under the hood — the schema IS the prompt")
    print("=" * 60)

    # This is (roughly) what with_structured_output sends to the provider.
    # Find your Field descriptions and the Literal enum in here — the model
    # reads all of it.
    print(json.dumps(TicketTriage.model_json_schema(), indent=2))

    # include_raw=True lets you see both the raw AIMessage and the parsed
    # object — useful for debugging "why did parsing fail" in real systems.
    triager = model.with_structured_output(TicketTriage, include_raw=True)
    out = triager.invoke(f"Triage this support ticket:\n\n{TICKET}")
    print(f"\nraw message type:   {type(out['raw']).__name__}")
    print(f"parsed result type: {type(out['parsed']).__name__}")
    print(f"parsing_error:      {out['parsing_error']}")
    print(f"tokens:             {out['raw'].usage_metadata}")


def part_3_validated_is_not_correct() -> None:
    """The shape is guaranteed. The judgment is not."""
    print("\n" + "=" * 60)
    print("PART 3: validated ≠ correct")
    print("=" * 60)

    ambiguous = (
        "The export button does nothing. I mean, it works if you wait, "
        "but honestly at this price it should be instant. Can you add a "
        "progress bar or something?"
    )
    # Is that a bug or a feature request? Reasonable people disagree —
    # and so will the model, run to run. It will ALWAYS give you a valid
    # TicketTriage. It will not always give you the same one, or the one
    # a human would pick. That's why evals exist (curriculum Part 5).
    triager = model.with_structured_output(TicketTriage)
    for i in range(2):
        r = triager.invoke(f"Triage this support ticket:\n\n{ambiguous}")
        print(f"run {i + 1}: category={r.category!r}, severity={r.severity}")
    print("\nValid shape every time — but check whether the runs even agree.")


if __name__ == "__main__":
    part_1_basic_triage()
    part_2_under_the_hood()
    part_3_validated_is_not_correct()
