#!/usr/bin/env python3
"""Dry-run stand-in for an OpenAI-compatible chat-completions endpoint.

Lets reviewers exercise CTX's `--with-model` plumbing (request building,
response parsing, usage accounting, grading, persistence, exit codes) with
zero API spend and fully deterministic answers. It is a PERFECT simulated
model: it answers correctly iff the needle keyword survived in the messages
it receives — so expect no off-diagonal divergences from a mock run. Real
divergences need a real backend.

Usage:
    python benchmarks/mock_model_server.py [port]   # default 18080

    $env:CTX_MODEL_API_KEY = "dry-run"
    $env:CTX_MODEL_BASE_URL = "http://127.0.0.1:18080"
    $env:CTX_MODEL_NAME = "mock-perfect"
    cargo run --example bench -- --with-model --model-limit 6 --model-pause-ms 0

Stdlib only.
"""

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

KEYWORDS = ["$500", "peanuts", "amara", "train", "epi pen"]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            self.send_error(400, "bad json")
            return
        texts = " ".join(m.get("content", "") for m in req.get("messages", []))
        hit = next((k for k in KEYWORDS if k in texts.lower()), None)
        if hit:
            answer = "Based on the conversation history, the answer is: {}.".format(hit)
        else:
            answer = "I don't have that information in the conversation."
        prompt_tokens = max(1, len(texts) // 4)
        completion_tokens = max(1, len(answer) // 4)
        resp = {
            "id": "mock-1",
            "choices": [{"message": {"role": "assistant", "content": answer}}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
        data = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 18080
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
