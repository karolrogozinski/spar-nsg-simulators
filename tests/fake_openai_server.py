"""Minimal OpenAI-compatible chat server for offline tests of backends and the runner.

    python tests/fake_openai_server.py [port] [fail_every]

Answers simulator prompts with a deterministic YES / NO (a hash of the prompt), wrapped in a
<think> block and an [ANSWER] section; every `fail_every`-th request returns HTTP 500.
Requests asking for an explanation text (renderer prompts) get a fixed sentence back.
"""

import hashlib
import itertools
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
FAIL_EVERY = int(sys.argv[2]) if len(sys.argv) > 2 else 7
calls = itertools.count(1)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if FAIL_EVERY and next(calls) % FAIL_EVERY == 0:
            self.send_response(500)
            self.end_headers()
            return
        prompt = body["messages"][-1]["content"]
        if "Write the explanation" in prompt:
            content = "The patient is sent to a doctor if the patient has a fever."
        else:
            answer = "YES" if int(hashlib.sha256(prompt.encode()).hexdigest(), 16) % 2 else "NO"
            content = f"<think>[ANSWER] maybe</think>\n[ANSWER]\n{answer}\n\n[CONFIDENCE]\nHIGH"
        out = json.dumps({
            "id": "fake", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


if __name__ == "__main__":
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
