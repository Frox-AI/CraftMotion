"""MCP server protocol tests (JSON-RPC over the handler, no subprocess needed)."""
import json

from craftmotion.core.registry import Registry
from craftmotion.mcp import MCPServer


def make():
    return MCPServer(Registry())


def test_initialize_and_capabilities():
    s = make()
    r = s.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert r["result"]["protocolVersion"]
    caps = r["result"]["capabilities"]
    assert "tools" in caps and "resources" in caps and "prompts" in caps


def test_tools_list_matches_commands():
    from craftmotion.core.registry import COMMANDS
    s = make()
    r = s.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = [t["name"] for t in r["result"]["tools"]]
    assert len(names) == len(COMMANDS)
    assert "layer_add" in names and "keyframe_set" in names


def test_tool_call_roundtrip():
    s = make()
    r = s.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                  "params": {"name": "project_new", "arguments": {"title": "A", "width": 200, "height": 100}}})
    assert not r["result"]["isError"]
    r = s.handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                  "params": {"name": "layer_add", "arguments": {"kind": "text", "text": "hey"}}})
    data = json.loads(r["result"]["content"][0]["text"])
    assert data["layer"]["kind"] == "text"


def test_tool_call_error_is_reported_not_raised():
    s = make()
    r = s.handle({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                  "params": {"name": "layer_get", "arguments": {"layer_id": 999}}})
    assert r["result"]["isError"] is True


def test_resources_and_prompts():
    s = make()
    r = s.handle({"jsonrpc": "2.0", "id": 6, "method": "resources/list"})
    uris = {x["uri"] for x in r["result"]["resources"]}
    assert "craftmotion://project" in uris and "craftmotion://am-format" in uris
    r = s.handle({"jsonrpc": "2.0", "id": 7, "method": "resources/read",
                  "params": {"uri": "craftmotion://timeline"}})
    assert "layers" in json.loads(r["result"]["contents"][0]["text"])
    r = s.handle({"jsonrpc": "2.0", "id": 8, "method": "prompts/list"})
    assert len(r["result"]["prompts"]) == 3
    r = s.handle({"jsonrpc": "2.0", "id": 9, "method": "prompts/get",
                  "params": {"name": "edit_like_a_human", "arguments": {"goal": "a reel"}}})
    assert "reel" in r["result"]["messages"][0]["content"]["text"]


def test_notifications_get_no_response():
    s = make()
    assert s.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_unknown_method_error():
    s = make()
    r = s.handle({"jsonrpc": "2.0", "id": 10, "method": "nope/nothing"})
    assert r["error"]["code"] == -32601
