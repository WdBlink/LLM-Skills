import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from verify_artifacts import verify


class ArtifactChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "bundle"
        self.root.mkdir()
        self.artifact = self.root / "input.txt"
        self.artifact.write_text("sample\n", encoding="utf-8")
        self.manifest = self.root / "registration.json"
        self.data = {"benchmark_id": "TEST", "version": "1", "status": "passed",
                     "sha256": {"input.txt": hashlib.sha256(b"sample\n").hexdigest()}}
        self.save()

    def save(self):
        self.manifest.write_text(json.dumps(self.data), encoding="utf-8")

    def assert_failed(self, fragment):
        result = verify(self.manifest)
        self.assertEqual(result["status"], "artifact_verification_failed")
        self.assertEqual(result["benchmark_validation"], "not_assessed")
        self.assertIn(fragment, json.dumps(result["errors"]))

    def test_valid_is_integrity_only(self):
        result = verify(self.manifest)
        self.assertEqual(result["verified_files"], ["input.txt"])
        self.assertEqual(result["status"], "artifacts_verified")
        self.assertEqual(result["declared_status"], "passed")
        self.assertEqual(result["benchmark_validation"], "not_assessed")

    def test_design_version_and_explicit_root(self):
        self.data["design_version"] = self.data.pop("version")
        self.manifest = self.base / "external.json"
        self.save()
        self.assertEqual(verify(self.manifest, self.root)["status"], "artifacts_verified")

    def test_changed_and_missing_artifact(self):
        self.artifact.write_text("changed", encoding="utf-8")
        self.assert_failed("mismatch")
        self.artifact.unlink()
        self.assert_failed("No such file")

    def test_invalid_metadata(self):
        original = dict(self.data)
        for key, value in [("benchmark_id", ""), ("benchmark_id", 7), ("version", None),
                           ("version", []), ("sha256", {}), ("sha256", [])]:
            with self.subTest(key=key, value=value):
                self.data = dict(original)
                self.data[key] = value
                self.save()
                self.assert_failed(key)
        self.data = dict(original, design_version="2")
        self.save()
        self.assert_failed("disagree")

    def test_duplicate_keys_and_malformed_json(self):
        for raw in ['{"benchmark_id":"a","benchmark_id":"b"}',
                    '{"sha256":{"a":"x","a":"y"}}']:
            self.manifest.write_text(raw, encoding="utf-8")
            self.assert_failed("duplicate JSON key")
        self.manifest.write_text("{", encoding="utf-8")
        self.assert_failed("Expecting property name")

    def test_unsafe_paths(self):
        digest = next(iter(self.data["sha256"].values()))
        for path in ["../input.txt", str(self.artifact), "C:/file", "dir\\file", ""]:
            with self.subTest(path=path):
                self.data["sha256"] = {path: digest}
                self.save()
                self.assert_failed("path must be relative")

    def test_symlink_escape_and_directory(self):
        outside = self.base / "outside.txt"
        outside.write_bytes(self.artifact.read_bytes())
        self.artifact.unlink()
        self.artifact.symlink_to(outside)
        self.assert_failed("escapes artifact root")
        self.artifact.unlink()
        self.artifact.mkdir()
        self.assert_failed("regular file")

    def test_invalid_digest(self):
        for digest in [None, "a" * 63, "G" * 64, "A" * 64, 123]:
            self.data["sha256"]["input.txt"] = digest
            self.save()
            self.assert_failed("64 lowercase")

    def test_empty_fixture_does_not_prove_real_data(self):
        self.artifact.write_bytes(b"")
        self.data["sha256"]["input.txt"] = hashlib.sha256(b"").hexdigest()
        self.save()
        result = verify(self.manifest)
        self.assertEqual(result["status"], "artifacts_verified")
        self.assertEqual(result["benchmark_validation"], "not_assessed")

    def test_cli_from_unrelated_directory_and_failure_exit(self):
        command = [sys.executable, str(Path(__file__).with_name("verify_artifacts.py").resolve()),
                   str(self.manifest)]
        completed = subprocess.run(command, cwd=self.base, capture_output=True, text=True, timeout=10)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout)["status"], "artifacts_verified")
        self.artifact.unlink()
        completed = subprocess.run(command, cwd=self.base, capture_output=True, text=True, timeout=10)
        self.assertEqual(completed.returncode, 1)
        self.assertIn("errors", json.loads(completed.stdout))


if __name__ == "__main__":
    unittest.main()
