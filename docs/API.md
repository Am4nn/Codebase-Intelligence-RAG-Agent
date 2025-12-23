# API

Base URL: `http://localhost:8000`

## Health

`GET /health`

Response:
```json
{ "status": "healthy", "system_ready": true }
```

## Status

`GET /status`

Response:
```json
{ "initialized": true, "repo_path": "...", "persist_dir": "..." }
```

## Query

`POST /query`

Body:
```json
{ "question": "How does X work?", "conversation_id": "optional" }
```

Response:
```json
{ "answer": "...", "question": "How does X work?" }
```

## Conversations

These endpoints expose the LangGraph checkpoint state stored by the checkpointer (currently `InMemorySaver`).

### List conversations

`GET /conversations`

Response:
```json
{ "conversations": ["default", "my-session"], "count": 2 }
```

### History

`GET /conversations/{conversation_id}/history`

Response:
```json
{
  "conversation_id": "default",
  "messages": [{ "role": "human", "content": "...", "timestamp": null }],
  "message_count": 1
}
```

### State

`GET /conversations/{conversation_id}/state`

Response:
```json
{ "conversation_id": "default", "exists": true, "state": { "message_count": 3, "metadata": {} } }
```

### Summary

`GET /conversations/{conversation_id}/summary`

Response:
```json
{ "conversation_id": "default", "exists": true, "message_count": 3, "role_counts": {"human": 2, "ai": 1} }
```

### Clear

`DELETE /conversations/{conversation_id}`

Response:
```json
{ "success": true, "conversation_id": "default" }
```

## Export

`POST /export?output_file=change_log.json`

Response:
```json
{ "success": true, "file_path": "change_log.json" }
```
