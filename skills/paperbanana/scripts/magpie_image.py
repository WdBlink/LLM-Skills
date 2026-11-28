#!/usr/bin/env python3
"""Use an installed Magpie image MCP over one stdio session, without a UI."""
from __future__ import annotations

import argparse
from collections import deque
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
import time


class TransportError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise TransportError(message)


def find_executable(explicit=None):
    selected = explicit or os.environ.get("PAPERBANANA_MAGPIE_BIN") or shutil.which("magpie")
    if not selected:
        # This is an executable discovery location, never an image/model default.
        candidate = Path("/Applications/magpie.app/Contents/MacOS/magpie")
        selected = str(candidate) if candidate.is_file() else None
    require(selected is not None, "Magpie is unavailable; set PAPERBANANA_MAGPIE_BIN or request.executable")
    resolved = shutil.which(selected) or selected
    require(Path(resolved).is_file() and os.access(resolved, os.X_OK), "Magpie executable is not runnable")
    return str(Path(resolved).resolve())


def positive(value, label):
    require(type(value) in (float, int) and math.isfinite(value) and value > 0,
            label + " must be a finite positive number")
    return float(value)


def output_target(value):
    require(isinstance(value, str) and value.strip(), "Declare an output path")
    path = Path(value).expanduser().absolute()
    require(not path.is_symlink(), "Output must not be a symlink: " + str(path))
    # macOS /tmp and /var are directory aliases. Normalize the chosen parent,
    # while refusing an existing or linked destination file.
    path = path.parent.resolve() / path.name
    require(not path.exists(), "Output already exists; use a new versioned path: " + str(path))
    return path


class Server:
    def __init__(self, executable, history):
        self.history = history
        self.messages = queue.Queue()
        self.diagnostics = deque(maxlen=40)
        self.next_id = 0
        self.process = subprocess.Popen(
            [executable, "mcp", "image"], stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.stderr_reader = threading.Thread(target=self._read_stderr, daemon=True)
        self.reader.start()
        self.stderr_reader.start()

    def _read(self):
        try:
            for line in self.process.stdout:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    self.diagnostics.append(line[:1000].rstrip())
                    continue
                self.messages.put(value)
        finally:
            self.messages.put(None)

    def _read_stderr(self):
        for line in self.process.stderr:
            self.diagnostics.append(line[:1000].rstrip())

    def send(self, value):
        self.process.stdin.write(json.dumps(value, allow_nan=False) + "\n")
        self.process.stdin.flush()

    def call(self, method, params, timeout):
        self.next_id += 1
        request_id = self.next_id
        message = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        self.history.append({"request": message})
        self.send(message)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                response = self.messages.get(timeout=max(0.001, deadline - time.monotonic()))
            except queue.Empty:
                break
            require(response is not None, "Image MCP closed before answering; inspect retained receipt")
            self.history.append({"response": response})
            require(isinstance(response, dict), "Invalid MCP response")
            if response.get("id") != request_id:
                continue
            require("error" not in response, "MCP JSON-RPC error: " + str(response.get("error")))
            require("result" in response, "MCP response has no result")
            return response["result"]
        raise TransportError("MCP reply timed out; usage is unknown. Do not resubmit generation automatically")

    def initialize(self):
        result = self.call("initialize", {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "paperbanana", "version": "0.1.0"},
        }, 30)
        require(isinstance(result, dict) and "serverInfo" in result, "MCP initialize failed")
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        tools = []
        cursor = None
        seen = set()
        while True:
            listing = self.call("tools/list", {"cursor": cursor} if cursor else {}, 30)
            require(isinstance(listing, dict) and isinstance(listing.get("tools"), list), "Invalid tools/list response")
            tools.extend(listing["tools"])
            cursor = listing.get("nextCursor")
            if not cursor:
                return tools
            require(cursor not in seen, "Repeated tools/list pagination cursor")
            seen.add(cursor)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.reader.join(timeout=2)
        self.stderr_reader.join(timeout=2)
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            stream.close()


def pending_id(result):
    structured = result.get("structuredContent") or {}
    if isinstance(structured, dict) and structured.get("status") in {"pending", "running", "queued"}:
        task_id = structured.get("task_id") or structured.get("id")
        if isinstance(task_id, str) and task_id:
            return task_id
    for block in result.get("content", []):
        if block.get("type") == "text":
            match = re.search(r'with id "([^"\n]+)"', block.get("text", ""))
            if match:
                return match.group(1)
    return None


def tool_result(server, name, arguments, timeout):
    result = server.call("tools/call", {"name": name, "arguments": arguments}, timeout)
    require(isinstance(result, dict), "Invalid image-tool result")
    if result.get("isError"):
        text = "\n".join(item.get("text", "") for item in result.get("content", []) if isinstance(item, dict))
        raise TransportError("Image tool failed: " + text[:2000])
    return result


def validate_arguments(tool, arguments):
    require(isinstance(arguments, dict), "arguments must be an object")
    schema = tool.get("inputSchema", {})
    missing = set(schema.get("required", [])) - set(arguments)
    require(not missing, "Missing required tool arguments: " + ", ".join(sorted(missing)))
    if schema.get("additionalProperties") is False:
        unknown = set(arguments) - set(schema.get("properties", {}))
        require(not unknown, "Unknown tool arguments: " + ", ".join(sorted(unknown)))


def run(request, receipt_path):
    require(isinstance(request, dict), "Request must be a JSON object")
    receipt_path = output_target(str(receipt_path))
    tool_name = request.get("tool")
    require(tool_name is None or isinstance(tool_name, str) and tool_name.strip(), "tool must be a nonempty string")
    arguments = request.get("arguments", {})
    require(isinstance(arguments, dict), "arguments must be a JSON object")
    image_path = output_target(request.get("output_path")) if tool_name else None
    if image_path:
        from PIL import Image
        require(image_path != receipt_path, "Image and receipt need different paths")
        # The current Magpie schema uses path; reject obvious output overwrites
        # before calling the server, while retaining schema discovery below.
        for key in ("path", "output_path"):
            if key in arguments:
                require(output_target(arguments[key]) == image_path,
                        "Tool output and output_path must identify the same new image")
    timeout = positive(request.get("timeout", 180), "timeout")
    poll_timeout = positive(request.get("poll_timeout", 300), "poll_timeout")
    interval = positive(request.get("poll_interval", 2), "poll_interval")
    poll_tool = request.get("poll_tool", "generation_result")
    poll_id_argument = request.get("poll_id_argument", "id")
    require(isinstance(poll_tool, str) and poll_tool.strip(), "poll_tool must be a name from tools/list")
    require(isinstance(poll_id_argument, str) and poll_id_argument.strip(), "poll_id_argument must be a schema property")
    executable = find_executable(request.get("executable"))
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"schema_version": 1, "status": "started", "usage": None,
               "executable": executable, "history": []}
    server = None
    # Reserve the receipt before contacting the service. A concurrent invocation
    # cannot overwrite the same attempt's evidence.
    with receipt_path.open("x", encoding="utf-8") as stream:
        def save():
            stream.seek(0)
            json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.truncate()
            stream.flush()

        try:
            server = Server(executable, receipt["history"])
            tools = server.initialize()
            summary = {"ok": True, "receipt": str(receipt_path)}
            if not tool_name:
                summary["tools"] = tools
            if tool_name:
                by_name = {item["name"]: item for item in tools}
                require(tool_name in by_name, "Tool is not exposed by this server: " + tool_name)
                validate_arguments(by_name[tool_name], arguments)
                require(arguments.get("n", 1) == 1, "Archive one image per version; use n=1 for this transport")
                output_fields = set(by_name[tool_name].get("inputSchema", {}).get("properties", {})) & {"path", "output_path"}
                require(not output_fields or any(key in arguments for key in output_fields),
                        "Pass the declared output path using the tool's path argument")
                require(poll_tool in by_name, "No configured result tool; select poll_tool from tools/list before generation")
                validate_arguments(by_name[poll_tool], {poll_id_argument: "schema-check"})
                image_path.parent.mkdir(parents=True, exist_ok=True)
                result = tool_result(server, tool_name, arguments, timeout)
                usage_records = []

                def capture_usage(value):
                    structured = value.get("structuredContent")
                    for container in (value, structured):
                        if isinstance(container, dict) and container.get("usage") is not None:
                            usage_records.append(container["usage"])
                    if usage_records:
                        receipt["usage"] = usage_records[-1]
                        receipt["usage_reports"] = usage_records.copy()

                capture_usage(result)
                save()
                deadline = time.monotonic() + poll_timeout
                while True:
                    task_id = pending_id(result)
                    if not task_id:
                        break
                    receipt["pending_task_id"] = task_id
                    save()
                    remaining = deadline - time.monotonic()
                    require(remaining > 0, "Asynchronous task timed out; usage is unknown. Do not resubmit")
                    time.sleep(min(interval, remaining))
                    remaining = deadline - time.monotonic()
                    require(remaining > 0, "Asynchronous task timed out; usage is unknown. Do not resubmit")
                    result = tool_result(server, poll_tool, {poll_id_argument: task_id}, min(60, remaining))
                    capture_usage(result)
                require(image_path.is_file() and not image_path.is_symlink(),
                        "Image tool returned without the declared output file; inspect receipt before recovery")
                with Image.open(image_path) as image:
                    image.load()
                    summary.update(image=str(image_path), pixels=list(image.size), format=image.format)
                summary["result"] = result
            receipt["status"] = "completed"
            return summary
        except BaseException as error:
            receipt["status"] = "failed_or_unknown"
            receipt["error"] = str(error)
            raise
        finally:
            try:
                if server:
                    server.close()
                    receipt["diagnostics"] = list(server.diagnostics)
            finally:
                save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, help="Runtime JSON request; {} discovers actual tools")
    parser.add_argument("--receipt", required=True, help="New path for this attempt's receipt")
    args = parser.parse_args(argv)
    try:
        request = json.loads(Path(args.request).read_text(encoding="utf-8"))
        result = run(request, args.receipt)
    except (ValueError, OSError, ImportError) as error:
        print("magpie_image: " + str(error), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
