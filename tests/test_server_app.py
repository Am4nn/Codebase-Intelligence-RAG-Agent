from fastapi.testclient import TestClient

import server.app as server_app


class FakeCodebaseIntelligence:
    def __init__(self):
        self._initialized = False
        self.repo_path = "fake-repo"
        self.persist_dir = "fake-persist"

    async def initialize(self, *args, **kwargs):
        self._initialized = True
        return self

    def is_initialized(self) -> bool:
        return self._initialized

    async def query(self, question: str, conversation_id: str) -> str:
        return f"answer: {question} ({conversation_id})"

    def export_change_log(self, output_file: str = "change_log.json") -> None:
        return None

    async def list_conversations(self):
        return ["c1"]

    async def get_conversation_history(self, conversation_id: str):
        return [{"role": "human", "content": "hi", "timestamp": None}]

    async def get_conversation_state(self, conversation_id: str):
        return {"conversation_id": conversation_id, "checkpoint_id": "x"}

    async def get_conversation_summary(self, conversation_id: str):
        return {
            "conversation_id": conversation_id,
            "exists": True,
            "message_count": 1,
            "role_counts": {"human": 1},
            "first_message": "hi",
            "last_message": "hi",
        }

    async def clear_conversation(self, conversation_id: str) -> bool:
        return True


def test_health_and_status_and_query_and_conversations(monkeypatch):
    # Prevent startup from requiring OpenAI keys or indexing.
    monkeypatch.setattr(server_app, "CodebaseIntelligence", FakeCodebaseIntelligence)

    with TestClient(server_app.app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

        r = client.get("/status")
        assert r.status_code == 200
        assert r.json()["initialized"] is True

        r = client.post("/query", json={"question": "what?", "conversation_id": "c1"})
        assert r.status_code == 200
        assert "answer:" in r.json()["answer"]

        r = client.get("/conversations")
        assert r.status_code == 200
        assert r.json()["conversations"] == ["c1"]

        r = client.get("/conversations/c1/history")
        assert r.status_code == 200
        assert r.json()["message_count"] == 1

        r = client.get("/conversations/c1/state")
        assert r.status_code == 200
        assert r.json()["exists"] is True

        r = client.get("/conversations/c1/summary")
        assert r.status_code == 200
        assert r.json()["exists"] is True

        r = client.delete("/conversations/c1")
        assert r.status_code == 200
        assert r.json()["success"] is True
