"""Exercise credential routing without a network client or model call."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import types
import unittest
from unittest.mock import Mock, patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "generate_weekly_report.py"
spec = importlib.util.spec_from_file_location("weekly_report", SCRIPT)
weekly = importlib.util.module_from_spec(spec)
spec.loader.exec_module(weekly)


class ProviderConfigTests(unittest.TestCase):
    def call(self, env):
        response = Mock()
        response.json.return_value = {"choices": [{"message": {"content": '{"summary":"test"}'}}]}
        post = Mock(return_value=response)
        stderr = io.StringIO()
        with patch.dict(os.environ, env, clear=True), patch.dict(
            "sys.modules", {"requests": types.SimpleNamespace(post=post)}
        ), contextlib.redirect_stderr(stderr):
            result = weekly.call_llm("synthetic work log")
        return result, post, stderr.getvalue()

    def test_each_provider_keeps_its_key_endpoint_and_model(self):
        cases = [
            ("MINIMAX", "https://api.minimaxi.com/v1/chat/completions"),
            ("OPENAI", "https://api.openai.com/v1/chat/completions"),
            ("SILICONFLOW", "https://api.siliconflow.cn/v1/chat/completions"),
        ]
        for prefix, url in cases:
            with self.subTest(prefix=prefix):
                result, post, error = self.call({f"{prefix}_API_KEY": "fake-key", f"{prefix}_MODEL": "test-model"})
                self.assertEqual(result, {"summary": "test"})
                self.assertEqual(error, "")
                args, kwargs = post.call_args
                self.assertEqual(args, (url,))
                self.assertEqual(kwargs["headers"]["Authorization"], "Bearer fake-key")
                self.assertEqual(kwargs["json"]["model"], "test-model")
                self.assertFalse(kwargs["allow_redirects"])

    def test_unrelated_provider_settings_are_not_used(self):
        result, post, _ = self.call({
            "MINIMAX_API_KEY": "selected-key",
            "OPENAI_API_KEY": "other-key", "OPENAI_BASE_URL": "https://other.example/v1",
            "OPENAI_MODEL": "wrong-model", "OPENAI_MAX_TOKENS": "9",
            "SILICONFLOW_TEMPERATURE": "1.8",
        })
        self.assertIsNotNone(result)
        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://api.minimaxi.com/v1/chat/completions")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer selected-key")
        self.assertEqual(kwargs["json"]["model"], "MiniMax-M2.5")
        self.assertEqual(kwargs["json"]["max_tokens"], 1200)
        self.assertEqual(kwargs["json"]["temperature"], 0.4)

    def test_key_without_required_model_never_sends(self):
        for prefix in ("OPENAI", "SILICONFLOW"):
            with self.subTest(prefix=prefix):
                result, post, error = self.call({f"{prefix}_API_KEY": "secret-not-to-log"})
                self.assertIsNone(result)
                post.assert_not_called()
                self.assertIn(f"{prefix}_MODEL", error)
                self.assertNotIn("secret-not-to-log", error)

    def test_custom_config_does_not_borrow_generic_key(self):
        for field in ("BASE_URL", "MODEL"):
            result, post, error = self.call({
                f"WEEKLY_REPORT_{field}": "https://custom.example/v1" if field == "BASE_URL" else "test-model",
                "MINIMAX_API_KEY": "generic-key", "WEEKLY_REPORT_API_KEY": " ",
            })
            self.assertIsNone(result)
            post.assert_not_called()
            self.assertIn("WEEKLY_REPORT_API_KEY", error)

    def test_incomplete_custom_config_does_not_fall_through(self):
        for missing in ("BASE_URL", "MODEL"):
            env = {"WEEKLY_REPORT_API_KEY": "custom-key", "WEEKLY_REPORT_BASE_URL": "https://custom.example/v1",
                   "WEEKLY_REPORT_MODEL": "test-model", "MINIMAX_API_KEY": "other-key"}
            del env[f"WEEKLY_REPORT_{missing}"]
            result, post, _ = self.call(env)
            self.assertIsNone(result)
            post.assert_not_called()

    def test_complete_custom_config_and_full_endpoint_are_preserved(self):
        result, post, _ = self.call({
            "WEEKLY_REPORT_API_KEY": "custom-key",
            "WEEKLY_REPORT_BASE_URL": "https://custom.example/v1/chat/completions/",
            "WEEKLY_REPORT_MODEL": "custom-model", "WEEKLY_REPORT_MAX_TOKENS": "345",
            "WEEKLY_REPORT_TEMPERATURE": "0.2", "MINIMAX_API_KEY": "unused-key",
        })
        self.assertIsNotNone(result)
        args, kwargs = post.call_args
        self.assertEqual(args[0], "https://custom.example/v1/chat/completions")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer custom-key")
        self.assertEqual(kwargs["json"]["model"], "custom-model")
        self.assertEqual(kwargs["json"]["max_tokens"], 345)
        self.assertEqual(kwargs["json"]["temperature"], 0.2)

    def test_selected_provider_override_is_normalized(self):
        result, post, _ = self.call({"OPENAI_API_KEY": "key", "OPENAI_MODEL": "test",
                                   "OPENAI_BASE_URL": "http://localhost:8000/v1/"})
        self.assertIsNotNone(result)
        self.assertEqual(post.call_args.args[0], "http://localhost:8000/v1/chat/completions")

    def test_invalid_endpoint_never_sends_or_logs_credential(self):
        for url in ("file:///tmp/test", "https://user:secret@example.com/v1", "https://example.com/v1?key=secret"):
            result, post, error = self.call({"WEEKLY_REPORT_API_KEY": "secret-key",
                                           "WEEKLY_REPORT_BASE_URL": url, "WEEKLY_REPORT_MODEL": "test"})
            self.assertIsNone(result)
            post.assert_not_called()
            self.assertNotIn("secret", error)

    def test_no_key_uses_fallback_without_a_request(self):
        result, post, error = self.call({})
        self.assertIsNone(result)
        self.assertEqual(error, "")
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
