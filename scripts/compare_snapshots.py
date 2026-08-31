"""Compares ablation results across two or more frozen-snapshot runs
(COMPETITION_PLAN.md §4 robustness protocol). A horizon only counts as
"beats momentum" once every listed snapshot agrees — one snapshot saying
yes is not evidence, it might just be that day's data revision.

Usage: python scripts/compare_snapshots.py 20260831_1719Z 20260901_1030Z ...
(tags match data/published/ablation_results__<tag>.json, produced by
`backtest/ablation.py --snapshot <tag>`.)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED_DIR = ROOT / "data" / "published"


def _load(tag: str) -> dict:
    path = PUBLISHED_DIR / f"ablation_results__{tag}.json"
    if not path.exists():
        raise FileNotFoundError(f"No ablation results for snapshot '{tag}' at {path}")
    return json.loads(path.read_text())


def main() -> None:
    tags = sys.argv[1:]
    if len(tags) < 2:
        print("Need at least 2 snapshot tags to compare. Usage: compare_snapshots.py <tag1> <tag2> [...]")
        sys.exit(1)

    runs = {tag: _load(tag) for tag in tags}
    all_horizons = sorted({h for r in runs.values() for h in r.get("horizons", {})}, key=int)

    print(f"{'Horizon':<8}" + "".join(f"{tag:<24}" for tag in tags) + "Agreement")
    for horizon in all_horizons:
        verdicts = []
        cells = []
        for tag in tags:
            h = runs[tag].get("horizons", {}).get(horizon)
            if h is None:
                cells.append("no data")
                verdicts.append(None)
                continue
            model = h["ranking_model"]["non_overlapping"]["mean_return"]
            mom = h["momentum"]["non_overlapping"]["mean_return"]
            beats = model > mom
            verdicts.append(beats)
            cells.append(f"{'YES' if beats else 'no'} ({model*100:+.2f}% vs {mom*100:+.2f}%)")

        known = [v for v in verdicts if v is not None]
        if len(known) < 2:
            agreement = "incomplete"
        elif all(known) or not any(known):
            agreement = "STABLE"
        else:
            agreement = "FLIPS — not robust"

        row = f"{horizon + 'D':<8}" + "".join(f"{c:<24}" for c in cells) + agreement
        print(row)

    print(
        "\nOnly report a horizon as 'beats momentum' if every snapshot above says YES "
        "(STABLE + all YES). A FLIPS row means the earlier robustness problem is still "
        "present for that horizon — do not ship it."
    )


if __name__ == "__main__":
    main()
