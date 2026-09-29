"""Package data/dataset/ as a zip for training on Kaggle (private dataset).

    uv run kintts-export                   # all kept clips
    uv run kintts-export --max-hours 6     # best clips only (metadata is sorted best-first)

Writes data/export/kin_host.zip containing train/metadata.csv (file_name,text)
plus the WAVs, the layout Hugging Face `load_dataset(<folder>)` reads directly.
"""

from __future__ import annotations

import argparse
import csv
import zipfile
from pathlib import Path

from kintts.normalize import normalize

ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", type=Path, default=ROOT / "data" / "dataset")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "export" / "kin_host.zip")
    ap.add_argument("--max-hours", type=float, default=None)
    args = ap.parse_args()

    with (args.dataset / "metadata.csv").open() as f:
        rows = list(csv.DictReader(f, delimiter="|"))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = 0.0
    lines = [["file_name", "text"]]
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED) as z:
        for r in rows:
            if args.max_hours and total / 3600 >= args.max_hours:
                break
            z.write(args.dataset / "wavs" / f"{r['id']}.wav", f"train/{r['id']}.wav")
            lines.append([f"{r['id']}.wav", normalize(r["text"]).lower()])
            total += float(r["seconds"])
        tmp = args.out.with_suffix(".csv")
        with tmp.open("w", newline="") as f:
            csv.writer(f).writerows(lines)
        z.write(tmp, "train/metadata.csv")
        tmp.unlink()
    print(f"{len(lines) - 1} clips, {total / 3600:.2f} h -> {args.out}")
    print("Upload it to Kaggle as a PRIVATE dataset named 'kin-host', then run train/kaggle_finetune.ipynb.")


if __name__ == "__main__":
    main()
