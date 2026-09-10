"""Build Colab-ready copies while keeping the paper authors' source untouched."""

from __future__ import annotations

import json
from pathlib import Path


PAPER_DIR = Path(__file__).resolve().parent
UPSTREAM_DIR = PAPER_DIR / "upstream"
REPRODUCTION_DIR = PAPER_DIR / "reproduction"

NOTEBOOKS = {
    "static_network_colab.ipynb": (
        UPSTREAM_DIR / "projects/static_network/static_network.ipynb",
        "projects/static_network",
        "Static network：观察 MLP 如何扭曲输入圆",
    ),
    "evidence_integration_colab.ipynb": (
        UPSTREAM_DIR
        / "projects/dynamic_networks/evidence_integration/evidence_integration.ipynb",
        "projects/dynamic_networks/evidence_integration",
        "Evidence integration：分析上下文相关的表示扭曲",
    ),
    "wm_colab.ipynb": (
        UPSTREAM_DIR / "projects/dynamic_networks/wm/wm.ipynb",
        "projects/dynamic_networks/wm",
        "Working memory：分析环面、度量与高斯曲率",
    ),
}


def markdown_cell(title: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": "colab-introduction",
        "metadata": {},
        "source": [
            f"# {title}\n",
            "\n",
            "这是论文作者 notebook 的 Colab 运行副本。作者源码保存在 "
            "`../upstream/`，这里仅增加环境初始化并清除旧输出。\n",
            "\n",
            "先运行下面的初始化单元，再选择 **Runtime → Restart session and run all**。",
        ],
    }


def bootstrap_cell(run_directory: str) -> dict:
    source = f'''from pathlib import Path
import os
import subprocess
import sys

REPO_URL = "https://github.com/Heptazero/nn-labs.git"
REPO_DIR = Path("/content/nn-labs")

if not (REPO_DIR / ".git").exists():
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)],
        check=True,
    )

SOURCE_DIR = REPO_DIR / "rnn-warping-2025" / "upstream"
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--quiet",
        "--editable",
        str(SOURCE_DIR),
    ],
    check=True,
)

RUN_DIR = SOURCE_DIR / "{run_directory}"
os.chdir(RUN_DIR)
print(f"Environment ready. Working directory: {{RUN_DIR}}")
'''
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": "colab-bootstrap",
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def build_notebook(source_path: Path, target_path: Path, run_directory: str, title: str) -> None:
    notebook = json.loads(source_path.read_text(encoding="utf-8"))

    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

    notebook["cells"] = [
        markdown_cell(title),
        bootstrap_cell(run_directory),
        *notebook["cells"],
    ]

    target_path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    REPRODUCTION_DIR.mkdir(exist_ok=True)
    for target_name, (source_path, run_directory, title) in NOTEBOOKS.items():
        build_notebook(
            source_path,
            REPRODUCTION_DIR / target_name,
            run_directory,
            title,
        )
        print(f"Built {target_name}")


if __name__ == "__main__":
    main()
