#!/usr/bin/env python3
"""Real CLI tests using offline images, never evidence of scientific review."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

CLI = Path(__file__).resolve().parents[1] / "scripts" / "paperbanana.py"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_image(path, mode="RGB", dpi=(150, 150)):
    color = (20, 30, 40, 90) if mode == "RGBA" else (20, 30, 40)
    image = Image.new(mode, (40, 20), color)
    image.save(path, **({"dpi": dpi} if dpi else {}))


class BackendTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="paperbanana-test-")
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name).resolve()
        self.root = self.work / "figure"
        self.source = self.work / "source.md"
        self.source.write_text("Offline fixture method. No image model or visual reviewer was called.\n")
        self.image = self.work / "returned.png"
        make_image(self.image)
        self.evidence = self.work / "inspection.md"
        self.evidence.write_text("Offline CLI fixture inspection declaration, not a scientific judgment.\n")
        self.receipt = self.work / "receipt.json"
        self.receipt.write_text('{"fixture": true, "provider_called": false}\n')
        self.call("init", "--figure-id", "F-method")
        brief = self.read("brief.json")
        brief.update(sources=[{"path": str(self.source), "sha256": digest(self.source)}],
                     no_reference_reason="Offline fixture", purpose="Exercise artifact consistency")
        self.write("brief.json", brief)

    def call(self, command, *args, expect=0, error=None):
        result = subprocess.run([sys.executable, "-B", str(CLI), command, "--root", str(self.root), *map(str, args)],
                                capture_output=True, text=True, timeout=15)
        self.assertNotIn("Traceback", result.stderr, result.stderr)
        self.assertEqual(result.returncode, expect, result.stdout + result.stderr)
        if error:
            self.assertIn(error, result.stderr)
        if expect == 2:
            self.assertEqual(result.stdout, "")
            self.assertTrue(result.stderr.strip())
            return None
        self.assertEqual(result.stderr, "", result.stderr)
        return json.loads(result.stdout)

    def read(self, name):
        return json.loads((self.root / name).read_text())

    def write(self, name, record):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2) + "\n")

    def snapshot(self):
        if not self.root.exists():
            return {}
        return {str(path.relative_to(self.root)): ("symlink", os.readlink(path)) if path.is_symlink()
                else ("dir",) if path.is_dir() else ("file", path.read_bytes(), path.stat().st_mtime_ns)
                for path in self.root.rglob("*")}

    def reject(self, command, *args, error):
        before = self.snapshot()
        self.call(command, *args, expect=2, error=error)
        self.assertEqual(before, self.snapshot(), "Refused commands must not write any artifact")

    def ingest(self, version="v1", *args, **kwargs):
        return self.call("ingest", "--version", version, "--source", self.image,
                         "--tool", "offline-fixture", *args, **kwargs)

    def review(self, version="v1", *args, verdict="pass", **kwargs):
        return self.call("critique", "--version", version, "--verdict", verdict,
                         "--evidence", self.evidence, "--method", "offline fixture declaration", *args, **kwargs)

    def select(self, version="v1", **kwargs):
        return self.call("select", "--version", version, "--reason", "Exercise selection binding", **kwargs)

    def ready(self, *args):
        self.ingest("v1", *args)
        self.review()
        self.select()

    def invalid(self, fragment=None):
        result = self.call("validate", "--require-selection", expect=1)
        self.assertFalse(result["ok"])
        if fragment:
            self.assertTrue(any(fragment in message for message in result["errors"]), result["errors"])

    def test_init_keeps_existing_brief_byte_for_byte(self):
        before = self.snapshot()
        result = self.call("init", "--figure-id", "F-method", "--route", "edit")
        self.assertFalse(result["created"])
        self.assertEqual(before, self.snapshot())
        self.reject("init", "--figure-id", "F-other", error="different figure")

    def test_absolute_external_inputs_and_identical_retry(self):
        result = self.ingest("v1", "--input-image", self.image, "--receipt", self.receipt)
        self.assertEqual(result["pixels"], [40, 20])
        self.assertEqual((self.root / result["image"]).read_bytes(), self.image.read_bytes())
        before = self.snapshot()
        self.ingest("v1", "--input-image", self.image, "--receipt", self.receipt)
        self.assertEqual(before, self.snapshot())
        self.review()
        self.select()
        self.assertTrue(self.call("validate", "--require-selection")["ok"])

    def test_same_version_rejects_changed_image_or_metadata_without_writes(self):
        self.ingest()
        self.reject("ingest", "--version", "v1", "--source", self.image,
                    "--tool", "different-tool", error="Output differs")
        Image.new("RGB", (41, 20), "red").save(self.image)
        self.reject("ingest", "--version", "v1", "--source", self.image,
                    "--tool", "offline-fixture", error="Output differs")

    def test_negative_ingest_does_not_leave_version_directory(self):
        missing = self.work / "missing.png"
        self.reject("ingest", "--version", "v1", "--source", self.image,
                    "--input-image", missing, error="Cannot decode")
        self.receipt.write_text("{")
        self.reject("ingest", "--version", "v1", "--source", self.image,
                    "--receipt", self.receipt, error="Invalid JSON")
        self.reject("ingest", "--version", "../v1", "--source", self.image, error="Version")

    def test_edit_uses_brief_and_requires_baseline_input(self):
        brief = self.read("brief.json")
        brief.update(route="edit", baseline={"path": str(self.image), "sha256": digest(self.image)},
                     permitted_changes=["Change a label"])
        self.write("brief.json", brief)
        self.reject("ingest", "--version", "v1", "--source", self.image,
                    "--operation", "generate", error="edit operation")
        self.reject("ingest", "--version", "v1", "--source", self.image,
                    "--operation", "edit", error="baseline was not passed")
        self.ingest("v1", "--operation", "edit", "--input-image", self.image)
        self.ready_version_check()

    def ready_version_check(self):
        self.review()
        self.select()
        self.assertTrue(self.call("validate", "--require-selection")["ok"])

    def test_source_and_brief_changes_invalidate_review_and_selection(self):
        self.ready()
        self.source.write_text("Changed source")
        self.invalid("source bytes changed")
        self.reject("select", "--version", "v1", "--reason", "stale", error="source bytes changed")
        self.source.write_text("Offline fixture method. No image model or visual reviewer was called.\n")
        brief = self.read("brief.json")
        brief["caption"] = "Changed caption"
        self.write("brief.json", brief)
        self.invalid("brief changed")
        self.reject("select", "--version", "v1", "--reason", "stale", error="brief changed")

    def test_archives_survive_original_cleanup_but_reject_changed_archived_bytes(self):
        self.ready("--input-image", self.image, "--receipt", self.receipt)
        self.image.unlink()
        self.receipt.unlink()
        self.evidence.unlink()
        self.assertTrue(self.call("validate", "--require-selection")["ok"])
        archived = self.root / "candidates/v1/inputs/1.png"
        original = archived.read_bytes()
        Image.new("RGB", (40, 20), "red").save(archived)
        self.invalid("input image bytes changed")
        archived.write_bytes(original)
        (self.root / "candidates/v1/receipt.json").write_text('{"fixture": "changed"}')
        self.invalid("receipt bytes changed")

    def test_baseline_change_is_detected(self):
        brief = self.read("brief.json")
        brief.update(route="edit", baseline={"path": str(self.image), "sha256": digest(self.image)},
                     permitted_changes=["Label"])
        self.write("brief.json", brief)
        self.ready("--operation", "edit", "--input-image", self.image)
        Image.new("RGB", (40, 20), "red").save(self.image)
        self.invalid("baseline bytes changed")

    def test_only_pass_can_be_selected_and_inspection_is_mandatory(self):
        self.ingest()
        self.reject("select", "--version", "v1", "--reason", "missing", error="Missing JSON")
        for verdict in ("unreviewed", "revise", "blocked"):
            with self.subTest(verdict=verdict):
                self.review(verdict=verdict)
                self.reject("select", "--version", "v1", "--reason", "not passing", error="passing")
        self.evidence.write_text(" \n\t")
        self.reject("critique", "--version", "v1", "--verdict", "pass", "--evidence", self.evidence,
                    "--method", "fixture", error="inspection evidence")
        self.evidence.unlink()
        self.reject("critique", "--version", "v1", "--verdict", "pass", "--evidence", self.evidence,
                    "--method", "fixture", error="No inspection evidence")
        self.reject("critique", "--version", "v1", "--verdict", "invented", "--evidence", self.evidence,
                    "--method", "fixture", error="invalid choice")

    def test_pass_rejects_blocking_findings_and_nonboolean_flags(self):
        self.ingest()
        findings = self.work / "findings.json"
        for blocking in (True, "false"):
            findings.write_text(json.dumps([{"issue": "Fixture defect", "blocking": blocking}]))
            self.reject("critique", "--version", "v1", "--verdict", "pass", "--evidence", self.evidence,
                        "--method", "fixture", "--findings", findings,
                        error="blocking finding" if blocking is True else "boolean")
        findings.write_text('[{"issue": "Fixture issue", "blocking": false}]')
        self.review("v1", "--findings", findings)
        self.select()

    def test_evidence_and_critique_changes_invalidate_selection(self):
        self.ready()
        archived = self.root / self.read("candidates/v1/critique.json")["inspection"]["evidence"]
        original = archived.read_bytes()
        archived.write_text("Changed archived inspection\n")
        self.invalid("inspection evidence bytes changed")
        self.reject("select", "--version", "v1", "--reason", "stale", error="evidence bytes changed")
        archived.write_bytes(original)
        self.evidence.write_text("Changed fixture declaration\n")
        self.review()
        self.invalid("selected critique changed")
        self.select()
        self.assertTrue(self.call("validate", "--require-selection")["ok"])
        before = self.snapshot()
        self.review()
        self.assertEqual(before, self.snapshot(), "Identical critique retry must preserve selected review")

    def test_record_identity_and_path_tampering_is_rejected(self):
        self.ready()
        for name, change, error in (
            ("candidates/v1/candidate.json", {"figure_id": "other"}, "figure_id"),
            ("candidates/v1/candidate.json", {"version": "v2"}, "version"),
            ("candidates/v1/critique.json", {"version": "v2"}, "version"),
            ("candidates/v1/critique.json", {"verdict": "invented"}, "verdict"),
            ("candidates/v1/critique.json", {"inspection": {}}, "inspection method"),
            ("selection.json", {"figure_id": "other"}, "figure_id"),
            ("candidates/v1/candidate.json", {"image": {"path": str(self.image), "format": "PNG"}}, "path"),
        ):
            with self.subTest(name=name, change=change):
                original = (self.root / name).read_bytes()
                record = self.read(name)
                record.update(change)
                self.write(name, record)
                self.invalid(error)
                if name != "selection.json":
                    self.reject("select", "--version", "v1", "--reason", "tampered", error=error)
                (self.root / name).write_bytes(original)
        selection = self.read("selection.json")
        selection["selected"]["image"]["path"] = str(self.image)
        self.write("selection.json", selection)
        self.invalid("selected image path")

    def test_candidate_pixels_and_metadata_changes_are_detected(self):
        self.ready()
        archived = self.root / "candidates/v1/image.png"
        make_image(archived, mode="RGBA")
        self.invalid("candidate image bytes changed")
        self.reject("select", "--version", "v1", "--reason", "stale", error="bytes changed")

    def test_require_selection_and_export_tampering(self):
        self.ingest()
        self.review()
        self.invalid("no selection.json")
        self.select()
        self.call("export", "--version", "v1", "--dpi", 300)
        self.assertTrue(self.call("validate", "--require-selection")["ok"])
        exported = self.root / "export/figure.png"
        exported.write_bytes(b"broken")
        self.invalid("export bytes changed")

    def test_malformed_json_and_shapes_are_controlled(self):
        self.ready()
        for name in ("brief.json", "candidates/v1/candidate.json", "candidates/v1/critique.json", "selection.json"):
            original = (self.root / name).read_bytes()
            for content in ("{", "[]", "null", '{"x": NaN}', '{"x":1,"x":2}'):
                with self.subTest(name=name, content=content):
                    (self.root / name).write_text(content)
                    self.invalid()
            (self.root / name).write_bytes(original)
        self.call("export", "--version", "v1", "--dpi", 300)
        for content in ("[]", "{", '{"version": []}'):
            (self.root / "export/export.json").write_text(content)
            self.invalid()

    def test_output_symlinks_and_hardlinks_are_rejected_for_every_record(self):
        self.ready("--receipt", self.receipt)
        self.call("export", "--version", "v1", "--dpi", 300)
        names = ("brief.json", "candidates/v1/candidate.json", "candidates/v1/image.png",
                 "candidates/v1/receipt.json", "candidates/v1/critique.json", "selection.json",
                 "export/export.json", "export/figure.png")
        for name in names:
            path = self.root / name
            original = path.read_bytes()
            for kind in ("symlink", "hardlink"):
                with self.subTest(name=name, kind=kind):
                    external = self.work / "linked-artifact"
                    external.write_bytes(original)
                    path.unlink()
                    if kind == "symlink":
                        path.symlink_to(external)
                    else:
                        os.link(external, path)
                    self.invalid("Symlink" if kind == "symlink" else "Hardlink")
                    self.reject("export", "--version", "v1", "--dpi", 300,
                                error="Symlink" if kind == "symlink" else "Hardlink")
                    self.assertEqual(external.read_bytes(), original)
                    path.unlink()
                    external.unlink()
                    path.write_bytes(original)

    def test_symlink_version_directories_and_export_parent_are_rejected(self):
        external = self.work / "outside"
        external.mkdir()
        (self.root / "candidates/v1").symlink_to(external, target_is_directory=True)
        self.reject("ingest", "--version", "v1", "--source", self.image, error="Symlink")
        self.invalid("Symlink")
        self.assertEqual(list(external.iterdir()), [])
        (self.root / "candidates/v1").unlink()
        self.ready()
        (self.root / "export").symlink_to(external, target_is_directory=True)
        self.reject("export", "--version", "v1", "--dpi", 300, error="Symlink")
        self.assertEqual(list(external.iterdir()), [])

    def test_export_validates_all_numeric_options_before_writing(self):
        self.ready()
        for option in ("dpi", "min-dpi", "width-pixels", "height-pixels", "max-bytes"):
            values = ("0", "-1", "nan", "inf") if option in ("dpi", "min-dpi") else ("0", "-1")
            for value in values:
                with self.subTest(option=option, value=value):
                    self.reject("export", "--version", "v1", "--" + option, value, error="positive")
        self.reject("export", "--version", "v1", "--dpi", "1e100", error="Cannot encode export")
        for option, value, error in (("width-pixels", "80", "no resizing"),
                                     ("height-pixels", "40", "no resizing"),
                                     ("max-bytes", "1", "byte limit"),
                                     ("min-dpi", "1000", "below")):
            self.reject("export", "--version", "v1", "--" + option, value, error=error)
        for target in ("../escape.png", str(self.work / "escape.png"), "candidates/v1/image.png", "export/bad.jpg"):
            self.reject("export", "--version", "v1", "--target", target, error="Export" if target.startswith(("candidates", "export")) else "Output")

    def test_export_png_and_tiff_preserve_pixels_and_original(self):
        self.ready()
        original = self.image.read_bytes()
        archived = (self.root / "candidates/v1/image.png").read_bytes()
        for suffix in ("png", "tiff"):
            target = "export/lossless." + suffix
            payload = self.call("export", "--version", "v1", "--target", target, "--dpi", 300,
                                "--min-dpi", 300, "--width-pixels", 40, "--height-pixels", 20)
            self.assertEqual(payload["observed"]["pixels"], [40, 20])
            with Image.open(self.root / target) as output, Image.open(self.image) as source:
                self.assertEqual(output.convert("RGBA").tobytes(), source.convert("RGBA").tobytes())
                self.assertEqual(tuple(round(value) for value in output.info["dpi"]), (300, 300))
            self.assertTrue(self.call("validate", "--require-selection")["ok"])
        self.assertEqual(self.image.read_bytes(), original)
        self.assertEqual((self.root / "candidates/v1/image.png").read_bytes(), archived)
        self.reject("export", "--version", "v1", "--target", "export/lossless.png", "--dpi", 301,
                    error="Output differs")

    def test_transparency_preserved_or_explicitly_composited(self):
        for mode in ("RGBA", "palette"):
            with self.subTest(mode=mode):
                if mode == "RGBA":
                    make_image(self.image, mode="RGBA", dpi=None)
                else:
                    image = Image.new("P", (40, 20), 0)
                    image.putpalette([10, 20, 30] + [0] * 765)
                    image.info["transparency"] = 0
                    image.save(self.image)
                version = "v-" + mode
                self.ingest(version)
                self.review(version)
                self.reject("export", "--version", version, "--mode", "RGB", "--dpi", 300,
                            error="explicit opaque background")
                self.reject("export", "--version", version, "--mode", "RGB", "--dpi", 300,
                            "--background", "#ffffff00", error="opaque")
                for suffix in ("png", "tiff"):
                    target = "export/" + version + "." + suffix
                    self.call("export", "--version", version, "--target", target, "--dpi", 300)
                    with Image.open(self.root / target) as output, Image.open(self.image) as source:
                        self.assertEqual(output.convert("RGBA").tobytes(), source.convert("RGBA").tobytes())
                self.call("export", "--version", version, "--target", "export/" + version + "-rgb.png",
                          "--mode", "RGB", "--background", "white", "--dpi", 300)

    def test_outputs_cannot_overwrite_inputs_or_follow_non_directory_parents(self):
        brief = self.read("brief.json")
        source = self.root / "candidates/v1/candidate.json"
        source.parent.mkdir(parents=True)
        source.write_text("source document")
        brief["sources"] = [{"path": str(source), "sha256": digest(source)}]
        self.write("brief.json", brief)
        self.reject("ingest", "--version", "v1", "--source", self.image,
                    error="Invalid JSON")
        source.unlink()
        source.parent.rmdir()
        source = self.root / "export/source.png"
        source.parent.mkdir()
        make_image(source)
        brief["sources"] = [{"path": str(self.source), "sha256": digest(self.source)},
                            {"path": str(source), "sha256": digest(source)}]
        self.write("brief.json", brief)
        self.ready()
        self.reject("export", "--version", "v1", "--target", "export/source.png", "--dpi", 300,
                    error="Output differs")
        self.reject("export", "--version", "v1", "--target", "export/source.png/child.png", "--dpi", 300,
                    error="not a directory")

    def test_no_resolution_metadata_needs_explicit_dpi(self):
        make_image(self.image, dpi=None)
        self.ingest()
        self.review()
        self.reject("export", "--version", "v1", error="Declare a positive DPI")
        self.call("export", "--version", "v1", "--dpi", 300)

    def test_export_requires_passing_critique_without_selection(self):
        self.ingest()
        self.reject("export", "--version", "v1", "--dpi", 300, error="Missing JSON")
        for verdict in ("revise", "blocked", "unreviewed"):
            self.review(verdict=verdict)
            self.reject("export", "--version", "v1", "--dpi", 300, error="passing")

    def test_export_then_select_other_version_cannot_pass_delivery(self):
        self.ingest()
        self.review()
        self.call("export", "--version", "v1", "--dpi", 300)
        Image.new("RGB", (40, 20), "red").save(self.image)
        self.ingest("v2")
        self.review("v2")
        self.select("v2")
        self.invalid("Export version differs")
        self.call("export", "--version", "v2", "--target", "export/v2.png", "--dpi", 300)
        self.assertTrue(self.call("validate", "--require-selection")["ok"])

    def test_changing_selection_invalidates_previous_export(self):
        self.ready()
        self.call("export", "--version", "v1", "--dpi", 300)
        Image.new("RGB", (40, 20), "red").save(self.image)
        self.ingest("v2")
        self.review("v2")
        self.select("v2")
        self.invalid("Export version differs")

    def test_png_resolution_rounding_does_not_reject_its_own_requested_dpi(self):
        self.ready()
        result = self.call("export", "--version", "v1", "--dpi", 112, "--min-dpi", 112)
        self.assertAlmostEqual(112, result["observed"]["dpi"][0], delta=0.0127)

    def test_critique_history_keeps_findings_and_evidence(self):
        self.ingest()
        findings = self.work / "findings.json"
        findings.write_text('[{"issue":"Arrow defect","blocking":true}]')
        self.review("v1", "--findings", findings, verdict="revise")
        self.evidence.write_text("New inspection resolving the earlier fixture finding\n")
        self.review()
        prior = self.read("candidates/v1/critique.json")["history"][0]
        self.assertEqual("Arrow defect", prior["findings"][0]["issue"])
        self.assertTrue((self.root / prior["inspection"]["evidence"]).is_file())

    def test_archived_run_can_move_when_brief_sources_are_relative(self):
        source = self.root / "source.md"
        source.write_bytes(self.source.read_bytes())
        brief = self.read("brief.json")
        brief["sources"] = [{"path": "source.md", "sha256": digest(source)}]
        self.write("brief.json", brief)
        self.ready("--input-image", self.image, "--receipt", self.receipt)
        self.call("export", "--version", "v1", "--dpi", 300)
        self.image.unlink()
        self.receipt.unlink()
        self.evidence.unlink()
        moved = self.work / "moved-figure"
        self.root.rename(moved)
        self.root = moved
        self.assertTrue(self.call("validate", "--require-selection")["ok"])


if __name__ == "__main__":
    unittest.main()
