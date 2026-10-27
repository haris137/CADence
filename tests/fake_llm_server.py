"""A tiny OpenAI-compatible HTTP server used by the tests (no network, no API key needed).

Behaviour is selected by the *model name* the client sends:
  good          answers from the FACTS block, citing a real tag
  ratelimit     always HTTP 429 (tests provider fallback)
  badtag        first answer cites a tag that does not exist, the retry is correct (tests the citation checker)
  empty         returns an empty answer when max_tokens < 1000 (tests the thinking-budget retry)
  noreasoning   HTTP 400 if 'reasoning_effort' is sent (tests optional-parameter removal)
Streaming (stream=true) is served as server-sent events like the real API.
"""
from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Tuple

STATE: Dict[str, int] = {}


def _answer(messages: List[dict], model: str) -> str:
    user = next((m["content"] for m in reversed(messages) if m["role"] == "user" and isinstance(m["content"], str)), "")
    if "Return one JSON object" in user or "Map each header" in user:
        if "Map each header" in user:
            return json.dumps({"mapping": {"Cust. Ref": "customer"}})
        return json.dumps({"part_number": "2280", "material": "AL 6061-T6", "scale": "1:1",
                           "notes": ["BREAK ALL EDGES 0.5", "HEAT TREAT TO 60 HRC"],  # 2nd note is invented
                           "holes": ["4X Ø8.5 THRU"], "overall_dimensions": None})
    all_user = "\n".join(m["content"] for m in messages if m["role"] == "user" and isinstance(m["content"], str))
    q = all_user.split("QUESTION:", 1)[-1].lower()
    facts = re.findall(r"^\[([A-Z]\d+)\] (.*)$", all_user, re.MULTILINE)
    if model == "badtag" and "failed automatic verification" not in user:
        return "The material is AL 6061-T6 [E999]."
    if "price" in q or "cost" in q:
        return "Not found in the provided files."
    want = "material" if "material" in q else ("dimension" if "dimension" in q else ("hole" if "hole" in q else ""))
    for tag, line in facts:
        low = line.lower()
        if want and want in low:
            value = line.split(" = ", 1)[-1] if " = " in line else line
            return f"From the {'Excel record' if tag[0] == 'E' else 'drawing' if tag[0] == 'P' else 'model'}: {value} [{tag}]."
    return "Not found in the provided files."


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):  # noqa: N802
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        model = req.get("model", "good")
        STATE[model] = STATE.get(model, 0) + 1
        if not self.path.endswith("/chat/completions"):
            return self._send(404, {"error": {"message": "not found"}})
        if model == "ratelimit":
            return self._send(429, {"error": {"message": "rate limited", "type": "rate_limit"}})
        if model == "noreasoning" and "reasoning_effort" in req:
            return self._send(400, {"error": {"message": "unknown field reasoning_effort"}})
        text = _answer(req.get("messages", []), model)
        if model == "empty" and int(req.get("max_tokens") or 0) < 1000:
            text = ""
        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for i in range(0, len(text), 12):
                chunk = {"id": "x", "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
                         "choices": [{"index": 0, "delta": {"content": text[i:i + 12]}, "finish_reason": None}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            end = {"id": "x", "object": "chat.completion.chunk", "created": int(time.time()), "model": model,
                   "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
            self.wfile.write(f"data: {json.dumps(end)}\n\ndata: [DONE]\n\n".encode())
            return
        return self._send(200, {"id": "x", "object": "chat.completion", "created": int(time.time()), "model": model,
                                "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                             "finish_reason": "stop"}],
                                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}})


def start() -> Tuple[str, ThreadingHTTPServer]:
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_address[1]}/v1", srv
