"""Safe fresh-path restoration and numerical comparison contracts."""
import hashlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from teacher_error_geometry.public_archive import relative_name, restore, check_tree
from teacher_error_geometry.public_validation import compare_values


class ArchiveContract(unittest.TestCase):
    def test_recorded_metadata_preserves_exact_coefficients_across_roundoff(self):
        path = Path(__file__).resolve().parents[1]/"scripts/frozen_cell_metadata.py"
        spec = importlib.util.spec_from_file_location("_test_frozen_metadata", path)
        helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
        saved = [{"config_id": "C_r0_n0_t1", "group_id": "d1_r0_n0", "rho": -.75,
                  "delta1": .7071067811865477, "delta2": .7071067811865476}]
        current = copy.deepcopy(saved)
        import numpy as np
        current[0]["delta1"] = float(np.nextafter(saved[0]["delta1"], np.inf))
        groups = [{"group_id": "d1_r0_n0", "cells": current}]
        tol = {"atol": 1e-10, "rtol": 1e-9}
        fixed, paired, receipt = helper.validated_frozen_plan(current, groups, saved, tol)
        self.assertEqual(fixed, saved)
        self.assertEqual(paired[0]["cells"], saved)
        self.assertNotEqual(current[0]["delta1"], fixed[0]["delta1"])
        self.assertEqual(receipt["historical_raw_equality_check"], "UNCHANGED_EXACT_EQUALITY")
        bad = copy.deepcopy(saved); bad[0]["delta1"] += 1e-4
        with self.assertRaises(ValueError): helper.validated_frozen_plan(current, groups, bad, tol)
        bad = copy.deepcopy(saved); bad[0]["rho"] = .75
        with self.assertRaises(ValueError): helper.validated_frozen_plan(current, groups, bad, tol)

    def fixture(self, root, kind="regular", duplicate=False):
        project = root/"project"; (project/"evidence").mkdir(parents=True)
        payload = b"immutable\n"
        archive = project/"evidence/records.tar.xz"
        with tarfile.open(archive, "w:xz") as tf:
            for _ in range(2 if duplicate else 1):
                member = tarfile.TarInfo("raw/data.csv"); member.size = len(payload)
                if kind == "symlink":
                    member.type = tarfile.SYMTYPE; member.linkname = "../../outside"; member.size = 0
                tf.addfile(member, io.BytesIO(payload) if kind == "regular" else None)
        entry = {"path": "raw/data.csv", "stored_as": "raw/data.csv", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        alias = dict(entry, path="rerun/raw/data.csv")
        manifest = {"archive": {"path": archive.name, "bytes": archive.stat().st_size, "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}, "entries": [entry, alias]}
        (project/"evidence/manifest.json").write_text(json.dumps(manifest))
        return project

    def test_lossless_alias_and_fresh_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); project = self.fixture(root); destination = root/"restored"
            result = restore(project, destination)
            self.assertEqual(result["files"], 2)
            self.assertEqual((destination/"raw/data.csv").read_bytes(), (destination/"rerun/raw/data.csv").read_bytes())
            with self.assertRaises(FileExistsError): restore(project, destination)

    def test_unsafe_names(self):
        for name in ("", ".", "../a", "/a", "a/../b", "a//b", "a\\b", ".git/config", "C:/x"):
            with self.subTest(name=name), self.assertRaises(ValueError): relative_name(name)

    def test_nonregular_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); project = self.fixture(root, "symlink")
            with self.assertRaises(ValueError): restore(project, root/"out")

    def test_duplicate_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); project = self.fixture(root, duplicate=True)
            with self.assertRaises(ValueError): restore(project, root/"out")

    def test_tamper_and_extra_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); project = self.fixture(root); out = root/"out"
            restore(project, out)
            entries = {e["path"]: e for e in json.loads((project/"evidence/manifest.json").read_text())["entries"]}
            (out/"extra").write_text("unexpected")
            with self.assertRaises(ValueError): check_tree(out, entries)
            archive = project/"evidence/records.tar.xz"
            archive.write_bytes(archive.read_bytes()+b"changed")
            with self.assertRaises(ValueError): restore(project, root/"second")

    def test_summary_tolerance_and_signed_cross(self):
        stats = {"numeric_values_compared": 0, "max_absolute_difference": 0.}
        compare_values({"cross": "+0.00041106984100782904", "z": None}, {"cross": "+0.00041106984100782904", "z": None}, stats)
        with self.assertRaises(ValueError):
            compare_values({"cross": 0.00041106984100782904}, {"cross": 0.0}, stats)


if __name__ == "__main__":
    unittest.main()
