"""Behaviour checks against an offline stdio server, never an image provider."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image

HERE = Path(__file__).resolve().parent
CLI = HERE.parent / "scripts" / "magpie_image.py"


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="paperbanana-mcp-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.executable = self.root / "fake-magpie"
        source = (HERE / "fake_image_server.py").read_text()
        self.executable.write_text("#!" + sys.executable + "\n" + source)
        self.executable.chmod(0o700)
        Image.new("RGB", (36, 18), "white").save(self.root / "source.png")
        self.environment = {**os.environ, "FAKE_IMAGE_DIR": str(self.root)}

    def invoke(self, request, mode="async"):
        request = {"executable": str(self.executable), **request}
        (self.root / "request.json").write_text(json.dumps(request))
        environment = {**self.environment, "FAKE_IMAGE_MODE": mode}
        result = subprocess.run([sys.executable, str(CLI), "--request", str(self.root / "request.json"),
                                 "--receipt", str(self.root / "receipt.json")],
                                env=environment, capture_output=True, text=True, timeout=12)
        return result

    def request(self):
        target = str(self.root / "candidate.png")
        return {"tool": "generate_image", "arguments": {"path": target, "prompt": "fixture"},
                "output_path": target, "poll_interval": 0.01, "poll_timeout": 0.1}

    def calls(self):
        return [json.loads(line) for line in (self.root / "calls.jsonl").read_text().splitlines()]

    def test_discovery_does_not_generate(self):
        result = self.invoke({})
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("generate_image", result.stdout)
        self.assertFalse(any(item["method"] == "tools/call" for item in self.calls()))

    def test_async_generation_uses_one_process_and_actual_file(self):
        result = self.invoke(self.request())
        self.assertEqual(0, result.returncode, result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual([36, 18], summary["pixels"])
        calls = self.calls()
        self.assertEqual(1, len({item["pid"] for item in calls}))
        names = [item["params"]["name"] for item in calls if item["method"] == "tools/call"]
        self.assertEqual(["generate_image", "generation_result"], names)
        receipt = json.loads((self.root / "receipt.json").read_text())
        self.assertEqual("completed", receipt["status"])
        self.assertIsNone(receipt["usage"])

    def test_sync_result_is_decoded(self):
        result = self.invoke(self.request(), "sync")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual([36, 18], json.loads(result.stdout)["pixels"])

    def test_json_rpc_error_is_a_failure_with_receipt(self):
        result = self.invoke(self.request(), "rpc_error")
        self.assertEqual(2, result.returncode, result.stderr)
        self.assertIn("RPC error", result.stderr)
        receipt = json.loads((self.root / "receipt.json").read_text())
        self.assertEqual("failed_or_unknown", receipt["status"])
        self.assertIn("fixture RPC failure", json.dumps(receipt))

    def test_provider_error_is_not_an_image(self):
        result = self.invoke(self.request(), "error")
        self.assertEqual(2, result.returncode)
        self.assertIn("provider failure", result.stderr)
        self.assertFalse((self.root / "candidate.png").exists())

    def test_pending_timeout_does_not_resubmit_generation(self):
        result = self.invoke(self.request(), "pending")
        self.assertEqual(2, result.returncode, result.stderr)
        receipt = json.loads((self.root / "receipt.json").read_text())
        self.assertEqual("fixture-task", receipt["pending_task_id"])
        calls = self.calls()
        names = [item["params"]["name"] for item in calls if item["method"] == "tools/call"]
        self.assertEqual(1, names.count("generate_image"))
        self.assertIn("unknown", result.stderr)

    def test_unexpected_server_exit_preserves_attempt(self):
        result = self.invoke(self.request(), "crash")
        self.assertEqual(2, result.returncode)
        self.assertIn("closed", result.stderr)
        self.assertEqual("failed_or_unknown", json.loads((self.root / "receipt.json").read_text())["status"])

    def test_existing_image_refused_before_server_start(self):
        request = self.request()
        (self.root / "candidate.png").write_bytes(b"existing")
        result = self.invoke(request)
        self.assertEqual(2, result.returncode)
        self.assertIn("already exists", result.stderr)
        self.assertFalse((self.root / "calls.jsonl").exists())

    def test_unknown_tool_does_not_call_generation(self):
        request = self.request()
        request["tool"] = "invented_tool"
        result = self.invoke(request)
        self.assertEqual(2, result.returncode)
        self.assertFalse(any(item["method"] == "tools/call" for item in self.calls()))

    def test_reported_usage_is_preserved_without_inventing_totals(self):
        result = self.invoke(self.request(), "usage")
        self.assertEqual(0, result.returncode, result.stderr)
        receipt = json.loads((self.root / "receipt.json").read_text())
        self.assertEqual({"total_cost_usd": 0.042}, receipt["usage"])

    def test_configured_poll_schema_is_used_in_same_process(self):
        request = {**self.request(), "poll_tool": "fetch_task", "poll_id_argument": "task_id"}
        result = self.invoke(request, "alternate_poll")
        self.assertEqual(0, result.returncode, result.stderr)
        calls = [item["params"] for item in self.calls() if item["method"] == "tools/call"]
        self.assertEqual("fetch_task", calls[-1]["name"])
        self.assertEqual({"task_id": "fixture-task"}, calls[-1]["arguments"])

    def test_missing_poll_tool_is_refused_before_generation(self):
        result = self.invoke(self.request(), "alternate_poll")
        self.assertEqual(2, result.returncode)
        self.assertFalse(any(item["method"] == "tools/call" for item in self.calls()))

    def test_directory_alias_is_normalized(self):
        alias = self.root / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        request = self.request()
        request["output_path"] = str(alias / "candidate.png")
        request["arguments"]["path"] = request["output_path"]
        result = self.invoke(request)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(str(self.root / "candidate.png"), json.loads(result.stdout)["image"])

    def test_output_file_symlink_is_refused_before_start(self):
        (self.root / "candidate.png").symlink_to(self.root / "source.png")
        result = self.invoke(self.request())
        self.assertEqual(2, result.returncode)
        self.assertIn("symlink", result.stderr)
        self.assertFalse((self.root / "calls.jsonl").exists())


if __name__ == "__main__":
    unittest.main()
