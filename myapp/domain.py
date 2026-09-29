"""Domain result types. The rest of the monolith speaks these, not AIMessage."""

from pydantic import BaseModel


class TriageResult(BaseModel):
    category: str
    draft_reply: str
    approved: bool
