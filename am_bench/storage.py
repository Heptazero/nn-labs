"""标准库实现的配置锁定、原子批次保存；不依赖 PyTorch。"""
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(canonical_json(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class BatchStore:
    """一个目录一个冻结实验；同一目录不允许不同配置或代码混跑。"""

    def __init__(self, directory, manifest):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest = json.loads(canonical_json(manifest))
        self.signature = fingerprint(self.manifest)
        path = self.directory / "manifest.json"
        if path.exists():
            if json.loads(path.read_text()) != self.manifest:
                raise ValueError("运行目录已有不同配置/代码；请选新目录，不能混合结果。")
        else:
            if any(self.directory.iterdir()):
                raise ValueError("非空目录缺少 manifest；请使用独立实验目录。")
            atomic_json(path, self.manifest)

    def path(self, key):
        if not key or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in key):
            raise ValueError("invalid batch key")
        return self.directory / (key + ".json")

    def load(self, key):
        path = self.path(key)
        if not path.exists():
            return None
        value = json.loads(path.read_text())
        if value["manifest_hash"] != self.signature or value["rows_hash"] != fingerprint(value["rows"]):
            raise ValueError(f"批次校验失败：{key}")
        return value["rows"]

    def save(self, key, rows):
        if self.path(key).exists():
            raise FileExistsError(f"已有批次不会覆盖：{key}")
        atomic_json(self.path(key), {"manifest_hash": self.signature,
                                    "rows_hash": fingerprint(rows), "rows": rows})
