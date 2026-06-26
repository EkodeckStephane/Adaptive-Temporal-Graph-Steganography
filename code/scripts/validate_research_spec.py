from __future__ import annotations

from pathlib import Path
import sys


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "code/src"))

    from stego.specification import validate_research_specification

    summary = validate_research_specification(root)
    print(
        {
            "level_a_datasets": summary.level_a_datasets,
            "actions": summary.actions,
            "required_ablations": len(summary.ablations),
        }
    )


if __name__ == "__main__":
    main()
