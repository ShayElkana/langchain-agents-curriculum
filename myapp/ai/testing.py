"""Test doubles for the agent runtime. Safe to import in CI — no API keys."""

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel


class ScriptedChatModel(GenericFakeChatModel):
    """GenericFakeChatModel does not implement bind_tools; create_agent needs it."""

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001
        return self
