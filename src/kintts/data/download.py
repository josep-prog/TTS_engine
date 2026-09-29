"""Download podcast / documentary audio listed in data/sources.txt as 16 kHz mono WAV.

    uv run kintts-download                 # reads data/sources.txt (one URL per line)
    uv run kintts-download URL [URL ...]

Playlists and channels work too. Already-downloaded items are skipped. All
audio stays in data/raw/ (git-ignored) and is for the internal demo only.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("urls", nargs="*")
    ap.add_argument("--sources", type=Path, default=ROOT / "data" / "sources.txt")
    args = ap.parse_args()

    urls = args.urls or [
        ln.strip() for ln in args.sources.read_text().splitlines()
        if ln.strip() and not ln.startswith("#")
    ]
    if not urls:
        raise SystemExit(f"No URLs. Add them to {args.sources} or pass them as arguments.")

    RAW.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        "yt-dlp", "--ffmpeg-location", ffmpeg,
        "-f", "bestaudio/best", "-x", "--audio-format", "wav",
        "--postprocessor-args", "ExtractAudio:-ar 16000 -ac 1",
        "--download-archive", str(RAW / "archive.txt"),
        "--ignore-errors", "--no-overwrites",
        "-o", str(RAW / "%(id)s.%(ext)s"),
        "--write-info-json", "--no-write-playlist-metafiles",
        *urls,
    ]
    subprocess.run(cmd, check=False)
    wavs = sorted(RAW.glob("*.wav"))
    print(f"{len(wavs)} files in {RAW}")


if __name__ == "__main__":
    main()
