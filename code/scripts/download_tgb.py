from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--root", default="datasets/raw")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    try:
        from tgb.linkproppred.dataset import LinkPropPredDataset
    except ImportError as exc:
        raise SystemExit("Install code/requirements.txt before downloading TGB.") from exc

    for name in args.datasets:
        dataset = LinkPropPredDataset(name=name, root=str(root))
        print(
            {
                "name": name,
                "full_data_keys": sorted(dataset.full_data.keys()),
                "root": str(root.resolve()),
            }
        )


if __name__ == "__main__":
    main()
