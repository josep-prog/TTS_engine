"""Synthesize the evaluation sentences with several TTS systems and build a
blind listening page where native speakers rate each clip 1-5 (MOS).

    uv run kintts-compare                      # MMS baseline, raw vs normalized text
    IJWILAB_API_KEY=ijw_... uv run kintts-compare --systems mms,mms-norm,ijwilab
    uv run kintts-compare --systems mms-norm,hf:your-name/your-kin-vits

Open eval/runs/<run>/index.html in a browser. Ratings are exported as CSV.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

from kintts.normalize import normalize

ROOT = Path(__file__).resolve().parents[2]


def load_sentences(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


class HFVits:
    """Any VITS checkpoint in Hugging Face format (MMS or your own fine-tune)."""

    def __init__(self, model_id: str, use_normalizer: bool):
        import torch
        from transformers import AutoTokenizer, VitsModel

        self.torch = torch
        self.model = VitsModel.from_pretrained(model_id).eval()
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.sr = self.model.config.sampling_rate
        self.use_normalizer = use_normalizer

    def __call__(self, text: str) -> tuple[np.ndarray, int]:
        if self.use_normalizer:
            text = normalize(text)
        inputs = self.tok(text, return_tensors="pt")
        self.torch.manual_seed(0)
        with self.torch.no_grad():
            wav = self.model(**inputs).waveform[0].numpy()
        return wav, self.sr


class IjwiLab:
    URL = "https://ijwilab.com/api/v1/tts.php"

    def __init__(self, speaker_id: int = 0):
        self.key = os.environ.get("IJWILAB_API_KEY")
        if not self.key:
            raise SystemExit("Set IJWILAB_API_KEY to include IjwiLab in the comparison.")
        self.speaker_id = speaker_id

    def __call__(self, text: str) -> tuple[np.ndarray, int]:
        body = json.dumps({"input": text, "language": "rw", "speaker_id": self.speaker_id}).encode()
        req = urllib.request.Request(self.URL, data=body, headers={
            "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            out = json.load(r)
        tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "ijwilab.wav"
        urllib.request.urlretrieve(out["output_url"], tmp)
        wav, sr = sf.read(tmp, dtype="float32")
        return wav, sr


def make_system(name: str):
    if name == "mms":
        return HFVits("facebook/mms-tts-kin", use_normalizer=False)
    if name == "mms-norm":
        return HFVits("facebook/mms-tts-kin", use_normalizer=True)
    if name.startswith("ijwilab"):
        _, _, spk = name.partition(":")
        return IjwiLab(int(spk or 0))
    if name.startswith("hf:"):
        return HFVits(name[3:], use_normalizer=True)
    raise SystemExit(f"Unknown system: {name}")


def build_page(run_dir: Path, sentences: list[str], systems: list[str]) -> None:
    rng = random.Random(42)
    items = []
    for i, text in enumerate(sentences):
        clips = [{"sys": s, "src": f"{s.replace(':', '_').replace('/', '_')}/{i:03d}.wav"} for s in systems]
        rng.shuffle(clips)  # blind: order differs per sentence, names hidden
        items.append({"i": i, "text": text, "clips": clips})
    template = (Path(__file__).parent / "listening_page.html").read_text(encoding="utf-8")
    html = template.replace("__DATA__", json.dumps(items, ensure_ascii=False))
    (run_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", default="mms,mms-norm")
    ap.add_argument("--sentences", type=Path, default=ROOT / "eval" / "sentences.txt")
    ap.add_argument("--run", default="baseline")
    args = ap.parse_args()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    sentences = load_sentences(args.sentences)
    run_dir = ROOT / "eval" / "runs" / args.run
    for name in systems:
        out = run_dir / name.replace(":", "_").replace("/", "_")
        out.mkdir(parents=True, exist_ok=True)
        synth = make_system(name)
        for i, text in enumerate(sentences):
            path = out / f"{i:03d}.wav"
            if path.exists():
                continue
            wav, sr = synth(text)
            sf.write(path, wav, sr)
        print(f"{name}: {len(sentences)} clips -> {out}")
    build_page(run_dir, sentences, systems)
    print(f"Open {run_dir / 'index.html'} to listen and rate.")


if __name__ == "__main__":
    main()
