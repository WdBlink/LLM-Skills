"""Offline routing tests. Image dependencies and the API client are mocked."""

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ms_qwen_vl.py"
spec = importlib.util.spec_from_file_location("ms_qwen_vl_tested", SCRIPT)
cli = importlib.util.module_from_spec(spec)
dependencies = {name: MagicMock() for name in ("cv2", "dotenv", "numpy", "openai", "PIL")}
with patch.dict(sys.modules, {**dependencies, spec.name: cli}):
    spec.loader.exec_module(cli)


class ProviderRoutingTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.client_patch = patch.object(cli, "OpenAI")
        self.client = self.client_patch.start()
        self.addCleanup(self.client_patch.stop)
        response = self.client.return_value.chat.completions.create.return_value
        response.choices = [MagicMock()]
        response.choices[0].message.content = "offline result"

    def analyze(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return cli.analyze_image("https://example.com/image.jpg", **kwargs)

    def test_modelscope_key_reaches_only_modelscope(self):
        os.environ["MODELSCOPE_API_KEY"] = "test-modelscope-key"
        self.assertEqual(self.analyze(), "offline result")
        self.client.assert_called_once_with(
            api_key="test-modelscope-key", base_url="https://api-inference.modelscope.cn/v1/"
        )

    def test_siliconflow_only_configuration_keeps_working(self):
        os.environ["SILICONFLOW_API_KEY"] = "test-siliconflow-key"
        self.analyze()
        self.client.assert_called_once_with(
            api_key="test-siliconflow-key", base_url="https://api.siliconflow.cn/v1/"
        )

    def test_selected_provider_does_not_borrow_other_key(self):
        os.environ["MODELSCOPE_API_KEY"] = "test-modelscope-key"
        with self.assertRaisesRegex(ValueError, "SILICONFLOW_API_KEY"):
            self.analyze(provider="siliconflow")
        self.client.assert_not_called()

    def test_two_keys_require_explicit_selection(self):
        os.environ.update(MODELSCOPE_API_KEY="test-ms", SILICONFLOW_API_KEY="test-sf")
        with self.assertRaisesRegex(ValueError, "多个服务"):
            self.analyze()
        self.client.assert_not_called()

    def test_explicit_key_without_provider_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "指定服务"):
            self.analyze(api_key="test-override")
        self.client.assert_not_called()

    def test_explicit_key_and_model_override_selected_environment(self):
        os.environ.update(MODELSCOPE_API_KEY="test-env", MODELSCOPE_MODEL="env-model")
        self.analyze(provider="modelscope", api_key="test-override", model="chosen-model")
        self.client.assert_called_once_with(
            api_key="test-override", base_url="https://api-inference.modelscope.cn/v1/"
        )
        self.assertEqual(
            self.client.return_value.chat.completions.create.call_args.kwargs["model"],
            "chosen-model",
        )

    def test_model_configuration_stays_with_selected_provider(self):
        os.environ.update(
            MODELSCOPE_API_KEY="test-ms", MODELSCOPE_MODEL="ms-fast",
            MODELSCOPE_MODEL_PRECISE="ms-precise", SILICONFLOW_API_KEY="test-sf",
            SILICONFLOW_MODEL="sf-fast", SILICONFLOW_MODEL_PRECISE="sf-precise",
        )
        for provider, expected in (("modelscope", "ms"), ("siliconflow", "sf")):
            for precise, suffix in ((False, "fast"), (True, "precise")):
                with self.subTest(provider=provider, precise=precise):
                    config = cli.resolve_api_config(provider=provider, precise=precise)
                    self.assertEqual(config.model, f"{expected}-{suffix}")

    def test_cli_provider_overrides_environment_and_reaches_request(self):
        os.environ.update(
            QWEN_VL_PROVIDER="modelscope", MODELSCOPE_API_KEY="test-ms",
            SILICONFLOW_API_KEY="test-sf", SILICONFLOW_MODEL="sf-model",
        )
        argv = [str(SCRIPT), "https://example.com/image.jpg", "--provider", "siliconflow"]
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
            cli.main()
        self.client.assert_called_once_with(
            api_key="test-sf", base_url="https://api.siliconflow.cn/v1/"
        )
        self.assertEqual(
            self.client.return_value.chat.completions.create.call_args.kwargs["model"], "sf-model"
        )

    def test_cli_invalid_configuration_exits_before_input_or_network(self):
        os.environ.update(QWEN_VL_PROVIDER="siliconflow", MODELSCOPE_API_KEY="test-ms")
        with patch.object(sys, "argv", [str(SCRIPT), "missing.jpg"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                cli.main()
        self.assertEqual(error.exception.code, 2)
        self.client.assert_not_called()

    def test_video_frames_use_selected_provider(self):
        os.environ.update(MODELSCOPE_API_KEY="test-ms", SILICONFLOW_API_KEY="test-sf")
        args = SimpleNamespace(
            question=None, prompt=None, categories=None, api_key=None, model=None,
            precise=False, provider="siliconflow",
        )
        capture = MagicMock()
        capture.get.return_value = 1
        capture.read.side_effect = [(True, object()), (False, None)]
        with tempfile.TemporaryDirectory() as output:
            with patch.object(cli.cv2, "VideoCapture", return_value=capture):
                with patch.object(cli, "get_image_content", return_value="data:image/png;base64,test"):
                    with contextlib.redirect_stdout(io.StringIO()):
                        cli.process_video("video.mp4", Path(output), 5, "describe", args, {})
            self.assertEqual((Path(output) / "frame_0.txt").read_text(), "offline result")
        self.client.assert_called_once_with(
            api_key="test-sf", base_url="https://api.siliconflow.cn/v1/"
        )

    def test_unknown_provider_and_missing_key_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "provider 必须"):
            cli.resolve_api_config(provider="unknown")
        with self.assertRaisesRegex(ValueError, "MODELSCOPE_API_KEY"):
            cli.resolve_api_config()

    def test_configuration_repr_omits_key(self):
        config = cli.resolve_api_config(provider="modelscope", api_key="test-secret")
        self.assertNotIn("test-secret", repr(config))


if __name__ == "__main__":
    unittest.main()
