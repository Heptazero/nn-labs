"""仅标准库：验证断点数据不会被不同配置、损坏文件或重跑覆盖。"""
import json
from pathlib import Path
import tempfile
import unittest

from am_bench.storage import BatchStore


class BatchStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = {"config": {"seeds": (0, 1)}, "source": "fixed"}

    def test_resume_and_refuse_overwrite(self):
        first = BatchStore(self.root, self.manifest)
        first.save("batch_0", [{"correct": True, "score": None}])
        resumed = BatchStore(self.root, self.manifest)
        self.assertEqual(resumed.load("batch_0"), [{"correct": True, "score": None}])
        self.assertIsNone(resumed.load("batch_1"))
        with self.assertRaises(FileExistsError):
            resumed.save("batch_0", [])

    def test_manifest_change_is_rejected(self):
        BatchStore(self.root, self.manifest)
        with self.assertRaises(ValueError):
            BatchStore(self.root, {**self.manifest, "source": "changed"})

    def test_corrupted_rows_are_rejected(self):
        store = BatchStore(self.root, self.manifest)
        store.save("batch", [{"correct": True}])
        path = store.path("batch")
        value = json.loads(path.read_text())
        value["rows"][0]["correct"] = False
        path.write_text(json.dumps(value))
        with self.assertRaises(ValueError):
            store.load("batch")

    def test_interrupted_temporary_file_is_not_a_completed_batch(self):
        store = BatchStore(self.root, self.manifest)
        (self.root / "batch.json.interrupted.tmp").write_text("{")
        resumed = BatchStore(self.root, self.manifest)
        self.assertIsNone(resumed.load("batch"))
        resumed.save("batch", [])
        self.assertEqual(resumed.load("batch"), [])

    def test_orphan_directory_and_unsafe_keys_are_rejected(self):
        (self.root / "orphan.json").write_text("{}")
        with self.assertRaises(ValueError):
            BatchStore(self.root, self.manifest)
        (self.root / "orphan.json").unlink()
        store = BatchStore(self.root, self.manifest)
        for key in ("", "../escape", "a/b"):
            with self.assertRaises(ValueError):
                store.load(key)


if __name__ == "__main__":
    unittest.main()
