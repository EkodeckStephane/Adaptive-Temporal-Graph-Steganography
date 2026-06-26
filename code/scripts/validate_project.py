from __future__ import annotations

import json
from pathlib import Path


REQUIRED_PATHS = (
    "papers/base",
    "papers/sota",
    "papers/metadata/library.csv",
    "datasets/metadata/manifest.yaml",
    "code/configs/base.yaml",
    "docs/protocols/experimental_protocol.md",
    "docs/scientific_lock/formal_method.md",
    "article/main.tex",
    "thesis/main.tex",
)


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    missing = [path for path in REQUIRED_PATHS if not (root / path).exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise SystemExit(f"Project validation failed. Missing:\n{formatted}")

    pdfs = sorted((root / "papers").rglob("*.pdf"))
    invalid_pdfs = [
        str(path.relative_to(root))
        for path in pdfs
        if path.stat().st_size == 0 or path.read_bytes()[:5] != b"%PDF-"
    ]
    if invalid_pdfs:
        formatted = "\n".join(f"- {path}" for path in invalid_pdfs)
        raise SystemExit(f"Invalid PDF files:\n{formatted}")

    notebook = root / "notebooks/colab/01_environment_and_data.ipynb"
    with notebook.open(encoding="utf-8") as handle:
        json.load(handle)

    print(f"Project structure is valid ({len(pdfs)} PDF files checked).")


if __name__ == "__main__":
    main()
