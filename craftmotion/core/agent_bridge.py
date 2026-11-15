"""Local HTTP bridge so agents can drive the running editor over MCP.

POST http://127.0.0.1:<port>/mcp with a JSON-RPC 2.0 body and
`Authorization: Bearer <token>`. Every tool call lands in the same Registry the
GUI is using, so an agent can edit the project a human is looking at, live.
"""
from __future__ import annotations

import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional

from .registry import Registry


class AgentBridge:
    def __init__(self, registry: Registry, host: str = "127.0.0.1", port: int = 8787,
                 on_call: Optional[Callable[[str], None]] = None):
        self.reg = registry
        self.host = host
        self.port = port
        self.token = secrets.token_hex(16)
        self.on_call = on_call
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.calls = 0

    @property
    def running(self) -> bool:
        return self._server is not None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/mcp"

    def start(self) -> dict:
        if self._server:
            return {"running": True, "url": self.url, "token": self.token}
        from ..mcp import MCPServer
        server_ref = self
        mcp = MCPServer(self.reg)

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_a):  # silence
                pass

            def _json(self, code: int, payload: dict):
                data = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(data)

            def do_OPTIONS(self):
                self.send_response(204)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "authorization, content-type")
                self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
                self.end_headers()

            def do_GET(self):
                if self.path in ("/health", "/"):
                    self._json(200, {"ok": True, "server": "craftmotion", "tools": len(mcp.tools),
                                     "calls": server_ref.calls})
                else:
                    self._json(404, {"error": "not found"})

            def do_POST(self):
                if self.path.rstrip("/") not in ("/mcp", "/message"):
                    self._json(404, {"error": "not found"})
                    return
                auth = self.headers.get("Authorization", "")
                if not auth.endswith(server_ref.token):
                    self._json(401, {"error": "bad token"})
                    return
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    body = json.loads(self.rfile.read(length).decode("utf-8"))
                except Exception as exc:
                    self._json(400, {"jsonrpc": "2.0", "id": None,
                                     "error": {"code": -32700, "message": f"parse: {exc}"}})
                    return
                messages = body if isinstance(body, list) else [body]
                responses = [r for r in (mcp.handle(m) for m in messages) if r is not None]
                for m in messages:
                    if m.get("method") == "tools/call":
                        server_ref.calls += 1
                        if server_ref.on_call:
                            try:
                                server_ref.on_call(str((m.get("params") or {}).get("name", "")))
                            except Exception:
                                pass
                if responses:
                    self._json(200, responses if isinstance(body, list) else responses[0])
                else:
                    self.send_response(202)
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()

        try:
            self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        except OSError:
            self.port = 0
            self._server = ThreadingHTTPServer((self.host, self.port), Handler)
            self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return {"running": True, "url": self.url, "token": self.token, "port": self.port}

    def stop(self) -> None:
        if self._server:
            try:
                self._server.shutdown()
                self._server.server_close()
            except Exception:
                pass
        self._server = None
