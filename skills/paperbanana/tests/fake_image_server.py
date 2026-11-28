#!/usr/bin/env python3
"""Offline MCP fixture; no network or model calls."""
import json
import os
from pathlib import Path
import shutil
import sys


def main():
    directory = Path(os.environ["FAKE_IMAGE_DIR"])
    mode = os.environ.get("FAKE_IMAGE_MODE", "async")
    task_output = None
    for line in sys.stdin:
        message = json.loads(line)
        with (directory / "calls.jsonl").open("a") as log:
            log.write(json.dumps({"pid": os.getpid(), **message}) + "\n")
        method = message.get("method")
        if "id" not in message:
            continue
        response = {"jsonrpc": "2.0", "id": message["id"]}
        if method == "initialize":
            response["result"] = {"protocolVersion": "2024-11-05", "capabilities": {},
                                  "serverInfo": {"name": "fixture", "version": "1"}}
        elif method == "tools/list":
            response["result"] = {"tools": [
                {"name": "generate_image", "inputSchema": {"type": "object", "required": ["path"],
                 "properties": {"path": {"type": "string"}, "prompt": {"type": "string"}}}},
                {"name": "fetch_task" if mode == "alternate_poll" else "generation_result",
                 "inputSchema": {"type": "object", "required": ["task_id" if mode == "alternate_poll" else "id"]}},
            ]}
        elif method == "tools/call":
            name = message["params"]["name"]
            if name == "generate_image":
                task_output = Path(message["params"]["arguments"]["path"])
                if mode == "rpc_error":
                    response["error"] = {"code": -32000, "message": "fixture RPC failure"}
                elif mode == "error":
                    response["result"] = {"isError": True, "content": [{"type": "text", "text": "fixture provider failure"}]}
                elif mode == "crash":
                    return
                elif mode == "sync":
                    shutil.copyfile(directory / "source.png", task_output)
                    response["result"] = {"content": [{"type": "text", "text": "saved " + str(task_output)}]}
                else:
                    response["result"] = {"content": [{"type": "text", "text": 'Task started with id "fixture-task"'}]}
            elif name in {"generation_result", "fetch_task"}:
                if mode == "pending":
                    response["result"] = {"content": [{"type": "text", "text": 'Task running with id "fixture-task"'}]}
                elif task_output:
                    shutil.copyfile(directory / "source.png", task_output)
                    response["result"] = {"content": [{"type": "text", "text": "saved " + str(task_output)}]}
                else:
                    response["error"] = {"code": -32000, "message": "no task in this process"}
        if mode == "usage" and task_output and task_output.is_file() and "result" in response:
            response["result"]["usage"] = {"total_cost_usd": 0.042}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
