"""Concatenates data/published/rationale_log/{date}.json entries over a
date range into one readable Markdown timeline — raw material for the
ISTC 2026 Final Stage pitch deck (if the team makes top 7), not the deck
itself. Building slides and presenting them is a human/team task; this
script just saves the scramble of re-reading two months of JSON files on
14 Oct when the top-7 list drops.

Usage:
    python scripts/build_pitch_deck_source.py [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--out PATH]

Defaults to the full available rationale_log range.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
RATIONALE_LOG_DIR = ROOT / "data" / "published" / "rationale_log"
DEFAULT_OUT = ROOT / "data" / "published" / "pitch_deck_source.md"


def _load_log_files(start: str | None, end: str | None) -> list[Path]:
    files = sorted(RATIONALE_LOG_DIR.glob("*.json"))
    if start:
        files = [f for f in files if f.stem >= start]
    if end:
        files = [f for f in files if f.stem <= end]
    return files


def _load_manual_notes(date_str: str) -> str | None:
    path = RATIONALE_LOG_DIR / f"{date_str}_manual_flow_notes.md"
    return path.read_text() if path.exists() else None


def build_markdown(start: str | None = None, end: str | None = None) -> str:
    files = _load_log_files(start, end)
    if not files:
        return "# Pitch Deck Source\n\nNo rationale log entries found for this range.\n"

    lines = ["# Pitch Deck Source — Rationale Log Timeline", ""]
    lines.append(f"Range: {files[0].stem} → {files[-1].stem} ({len(files)} trading days)")
    lines.append("")
    lines.append(
        "Auto-generated from data/published/rationale_log/ — technical/"
        "fundamental/macro/risk notes are pipeline output; money_flow_notes "
        "is the team's own research (see the paired *_manual_flow_notes.md "
        "files). This is source material for slides, not slides."
    )
    lines.append("")

    for f in files:
        date_str = f.stem
        entries = json.loads(f.read_text())
        lines.append(f"## {date_str}")
        lines.append("")

        opens = [e for e in entries if e["action"] == "open"]
        holds = [e for e in entries if e["action"] == "hold"]
        closes = [e for e in entries if e["action"] == "close"]
        lines.append(
            f"_{len(opens)} opened, {len(holds)} held, {len(closes)} closed._"
        )
        lines.append("")

        for e in entries:
            lines.append(f"### {e['ticker']} — {e['action'].upper()}")
            lines.append(f"- **Technical:** {e['technical_notes']}")
            lines.append(f"- **Fundamental:** {e['fundamental_notes']}")
            lines.append(f"- **Macro:** {e['macro_notes']}")
            lines.append(f"- **Risk:** {e['risk_notes']}")
            proxies = e.get("money_flow_proxies")
            if proxies:
                flags = []
                if proxies.get("volume_spike_no_followthrough"):
                    flags.append("volume spike w/o price follow-through")
                if proxies.get("price_up_declining_volume"):
                    flags.append("price up on declining volume")
                proxy_str = ", ".join(flags) if flags else "no notable pattern"
                lines.append(
                    f"- **Money flow proxy (not real flow data):** {proxy_str} "
                    f"(volume z-score {proxies.get('volume_zscore')})"
                )
            money_flow = e.get("money_flow_notes")
            lines.append(f"- **Money flow (team research):** {money_flow or '_not yet filled in_'}")
            lines.append("")

        manual = _load_manual_notes(date_str)
        if manual:
            lines.append("**Manual flow-notes file for this day:**")
            lines.append("")
            lines.append(manual)
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=None, help="YYYY-MM-DD, inclusive")
    parser.add_argument("--end", default=None, help="YYYY-MM-DD, inclusive")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    markdown = build_markdown(args.start, args.end)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(markdown)
    logger.info(f"Pitch deck source written → {out_path} ({len(markdown)} chars)")


if __name__ == "__main__":
    main()
