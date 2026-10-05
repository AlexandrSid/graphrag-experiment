from graphrag_lab.agents.cursor_client import CursorCliClient

c = CursorCliClient()
res = c.chat_json("cursor-grok-4.6-high-fast", [{"role": "user", "content": "Return JSON: {\"status\": \"ok\"}"}])
print("РЕЗУЛЬТАТ:", res)
