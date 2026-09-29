"""Step 2 EXERCISE — extract structured code-review findings from a diff.

Goal: define a CodeReviewFinding schema, ask the model to review the diff
below, and get back a LIST of validated findings you could feed into a
review tool.

Two-part exercise:
  A) Fill in the TODOs and get typed findings printing.
  B) The degradation experiment: once (A) works, delete all the Field
     descriptions and change severity from Literal to plain str. Re-run.
     Compare the quality and consistency of what comes back. Lesson:
     schema quality = output quality.

Run with:  uv run python steps/step2_exercise_code_review.py
"""

import os
from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, Field

load_dotenv()

model = init_chat_model(os.getenv("MODEL", "openai:gpt-5-nano"), timeout=60, max_retries=2)

SAMPLE_DIFF = '''\
--- a/app/services/payments.py
+++ b/app/services/payments.py
@@ -10,18 +10,22 @@
 def process_refund(order_id, amount):
-    order = db.get_order(order_id)
-    if order is None:
-        raise OrderNotFound(order_id)
-    if amount > order.total:
-        raise InvalidRefund("refund exceeds order total")
+    order = db.get_order(order_id)
     gateway.refund(order.payment_id, amount)
     db.mark_refunded(order_id, amount)
     return True
+
+def process_refund_batch(order_ids, amount):
+    results = {}
+    for oid in order_ids:
+        try:
+            results[oid] = process_refund(oid, amount)
+        except:
+            pass
+    return results
'''


# TODO 1: define the schema.
#   class CodeReviewFinding(BaseModel):
#       - file: which file the finding is in
#       - line: approximate line number (int)
#       - severity: Literal["info", "warning", "critical"]
#       - explanation: what is wrong and why it matters
#       - suggested_fix: a concrete, minimal fix
#   Write a real docstring and real Field descriptions — part B will show
#   you why they matter.

class CodeReviewFinding(BaseModel):
    file: str = Field(description="The file the finding is in")
    line: int = Field(description="The approximate line number of the finding")
    severity: Literal["info", "warning", "critical"] = Field(description="The severity of the finding")
    explanation: str = Field(description="What is wrong and why it matters")
    suggested_fix: str = Field(description="A concrete, minimal fix")

# TODO 2: the model must return a LIST of findings, but
#   with_structured_output takes ONE schema class. The standard trick is a
#   wrapper model:
#       class CodeReview(BaseModel):
#           findings: list[CodeReviewFinding]
#   Define it.
class CodeReview(BaseModel):
    findings: list[CodeReviewFinding] = Field(description="A list of code review findings")


def main() -> None:
    # TODO 3: build `reviewer = model.with_structured_output(CodeReview)`,
    #   invoke it with a prompt containing SAMPLE_DIFF, then loop over
    #   result.findings and print each one as:
    #       [{severity}] {file}:{line} — {explanation}
    #             fix: {suggested_fix}
    #
    # Hint: the diff hides at least three genuine problems. See if the
    # model catches what you caught (read it yourself first!).
    reviewer = model.with_structured_output(CodeReview)
    # The schema defines the SHAPE of the answer; the prompt defines the QUESTION.
    result = reviewer.invoke(
        "Review this code diff and report all genuine problems "
        f"(bugs, removed safety checks, error-handling issues):\n\n{SAMPLE_DIFF}"
    )
    for finding in result.findings:
        print(f"[{finding.severity}] {finding.file}:{finding.line} — {finding.explanation}")
        print(f"fix: {finding.suggested_fix}")
    print(f"Total findings: {len(result.findings)}")
    return
    #raise NotImplementedError("Complete TODOs 1-3, then delete this line.")


if __name__ == "__main__":
    main()
