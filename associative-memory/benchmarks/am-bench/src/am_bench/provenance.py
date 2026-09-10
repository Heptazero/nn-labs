"""记录实际载入的仓库与模块内容，而非远端 main 的当前值。"""
import hashlib
from pathlib import Path
import platform
import subprocess


def source_info():
    package = Path(__file__).resolve().parent
    root = package.parent
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unresolved"
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return {"git_commit": commit, "package_sha256": digest.hexdigest(),
            "python_version": platform.python_version()}
