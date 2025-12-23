import pytest

from core.api.codebase_intelligence import CodebaseIntelligence


class _Msg:
    def __init__(self, msg_type: str, content: str):
        self.type = msg_type
        self.content = content


class _StubCheckpointer:
    def __init__(self):
        # Mirrors the attribute the implementation probes for.
        self.storage: dict[str, dict] = {}

    async def aget(self, config):  # matches `checkpointer.aget(...)` usage
        thread_id = config["configurable"]["thread_id"]
        return self.storage.get(thread_id)


class _StubInnerAgent:
    def __init__(self, checkpointer: _StubCheckpointer):
        self.checkpointer = checkpointer


class _StubRagAgent:
    def __init__(self, checkpointer: _StubCheckpointer):
        self.agent = _StubInnerAgent(checkpointer)


def _initialized_ci_with_checkpointer() -> tuple[CodebaseIntelligence, _StubCheckpointer]:
    ci = CodebaseIntelligence()
    ci._initialized = True
    cp = _StubCheckpointer()
    ci.agent = _StubRagAgent(cp)  # type: ignore[assignment]
    return ci, cp


@pytest.mark.asyncio
async def test_get_conversation_history_empty_when_missing():
    ci, _ = _initialized_ci_with_checkpointer()
    history = await ci.get_conversation_history("missing")
    assert history == []


@pytest.mark.asyncio
async def test_get_conversation_history_formats_messages():
    ci, cp = _initialized_ci_with_checkpointer()

    cp.storage["c1"] = {
        "id": "checkpoint-1",
        "channel_values": {"messages": [_Msg("human", "hi"), _Msg("ai", "hello")]},
        "metadata": {},
    }

    history = await ci.get_conversation_history("c1")
    assert [m["role"] for m in history] == ["human", "ai"]
    assert [m["content"] for m in history] == ["hi", "hello"]


@pytest.mark.asyncio
async def test_list_conversations_reads_checkpointer_storage_keys():
    ci, cp = _initialized_ci_with_checkpointer()
    cp.storage["c1"] = {"channel_values": {"messages": []}}
    cp.storage["c2"] = {"channel_values": {"messages": []}}

    conversations = await ci.list_conversations()
    assert set(conversations) == {"c1", "c2"}


@pytest.mark.asyncio
async def test_get_conversation_state_contains_basic_fields():
    ci, cp = _initialized_ci_with_checkpointer()
    cp.storage["c1"] = {
        "id": "checkpoint-1",
        "channel_values": {"messages": [_Msg("human", "hi")]},
        "metadata": {"foo": "bar"},
    }

    state = await ci.get_conversation_state("c1")
    assert state is not None
    assert state["conversation_id"] == "c1"
    assert state["checkpoint_id"] == "checkpoint-1"
    assert state["message_count"] == 1
    assert state["metadata"] == {"foo": "bar"}


@pytest.mark.asyncio
async def test_clear_conversation_removes_matching_keys():
    ci, cp = _initialized_ci_with_checkpointer()
    cp.storage["c1"] = {"channel_values": {"messages": []}}
    cp.storage["other"] = {"channel_values": {"messages": []}}

    assert await ci.clear_conversation("c1") is True
    assert "c1" not in cp.storage
    assert "other" in cp.storage


@pytest.mark.asyncio
async def test_get_conversation_summary_counts_roles_and_previews():
    ci, cp = _initialized_ci_with_checkpointer()
    cp.storage["c1"] = {
        "id": "checkpoint-1",
        "channel_values": {
            "messages": [_Msg("human", "hello"), _Msg("ai", "world")]
        },
        "metadata": {},
    }

    summary = await ci.get_conversation_summary("c1")
    assert summary["exists"] is True
    assert summary["message_count"] == 2
    assert summary["role_counts"] == {"human": 1, "ai": 1}
    assert summary["first_message"] == "hello"
    assert summary["last_message"] == "world"
