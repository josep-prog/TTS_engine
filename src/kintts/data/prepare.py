"""Turn long podcast/documentary recordings into a single-speaker TTS dataset.

    uv run kintts-prepare                  # data/raw/*.wav -> data/dataset/
    uv run kintts-prepare --separate       # also strip background music (slow on CPU; use a GPU)

Stages (each is cached in data/work/, so re-runs only process new files):
  1. separate  (optional) Demucs removes music/effects, keeps the voice
  2. segment   Silero VAD cuts speech at pauses into 2-12 s clips
  3. speaker   ECAPA voice embeddings; the largest voice cluster = the main host
  4. asr       Kinyarwanda speech recognition (w2v-BERT 2.0, 1000 h) + confidence
  5. filter    keep host clips with confident transcripts and a normal speaking rate

Output: data/dataset/wavs/*.wav (16 kHz mono) and metadata.csv
        (id|text|seconds|speaker_sim|asr_conf), best clips first.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data"
SR = 16000
ASR_MODEL = "badrex/w2v-bert-2.0-kinyarwanda-asr"


def load_wav(path: Path) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != SR:
        import torchaudio.functional as F
        wav = F.resample(torch.from_numpy(wav), sr, SR).numpy()
    return wav


# 1. separate ---------------------------------------------------------------

def separate(src: Path, out_dir: Path) -> Path:
    out = out_dir / f"{src.stem}.wav"
    if out.exists():
        return out
    from demucs.apply import apply_model
    from demucs.pretrained import get_model
    import torchaudio.functional as F

    model = get_model("htdemucs").eval()
    wav = torch.from_numpy(load_wav(src))
    x = F.resample(wav, SR, model.samplerate).expand(2, -1)[None]
    with torch.no_grad():
        stems = apply_model(model, x, device="cuda" if torch.cuda.is_available() else "cpu", progress=True)[0]
    vocals = stems[model.sources.index("vocals")].mean(0)
    sf.write(out, F.resample(vocals, model.samplerate, SR).numpy(), SR)
    return out


# 2. segment ----------------------------------------------------------------

def segment(wav: np.ndarray, min_s: float, max_s: float) -> list[tuple[int, int]]:
    """Speech regions from VAD, merged across short pauses up to max_s."""
    from silero_vad import get_speech_timestamps, load_silero_vad

    vad = load_silero_vad()
    ts = get_speech_timestamps(torch.from_numpy(wav), vad, sampling_rate=SR,
                               min_silence_duration_ms=300, speech_pad_ms=120)
    segs: list[tuple[int, int]] = []
    for t in ts:
        s, e = t["start"], t["end"]
        if segs and (e - segs[-1][0]) / SR <= max_s and (s - segs[-1][1]) / SR < 0.6:
            segs[-1] = (segs[-1][0], e)  # merge with previous across a short pause
        else:
            segs.append((s, e))
    return [(s, e) for s, e in segs if min_s <= (e - s) / SR <= max_s]


# 3. speaker ----------------------------------------------------------------

def embed(clips: list[np.ndarray]) -> np.ndarray:
    from speechbrain.inference.speaker import EncoderClassifier

    enc = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb",
                                         savedir=str(DATA / "models" / "ecapa"))
    out = []
    for c in clips:
        with torch.no_grad():
            e = enc.encode_batch(torch.from_numpy(c)[None]).squeeze().numpy()
        out.append(e / np.linalg.norm(e))
    return np.stack(out)


def main_speaker_similarity(emb: np.ndarray) -> np.ndarray:
    """Cosine similarity of every clip to the centroid of the biggest voice cluster."""
    from sklearn.cluster import AgglomerativeClustering

    sample = emb if len(emb) <= 3000 else emb[np.random.default_rng(0).choice(len(emb), 3000, replace=False)]
    labels = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                     distance_threshold=0.55).fit_predict(sample)
    top = np.bincount(labels).argmax()
    centroid = sample[labels == top].mean(0)
    centroid /= np.linalg.norm(centroid)
    print(f"  main voice cluster: {np.mean(labels == top):.0%} of sampled clips")
    return emb @ centroid


# 4. asr --------------------------------------------------------------------

class ASR:
    def __init__(self):
        from transformers import AutoModelForCTC, AutoProcessor

        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.proc = AutoProcessor.from_pretrained(ASR_MODEL)
        self.model = AutoModelForCTC.from_pretrained(ASR_MODEL).to(self.dev).eval()

    def __call__(self, wav: np.ndarray) -> tuple[str, float]:
        inputs = self.proc(wav, sampling_rate=SR, return_tensors="pt").to(self.dev)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
        probs = logits.softmax(-1)
        ids = probs.argmax(-1)
        blank = self.proc.tokenizer.pad_token_id
        speech = ids != blank
        conf = float(probs.max(-1).values[speech].mean()) if speech.any() else 0.0
        text = self.proc.decode(ids)
        return " ".join(text.split()), conf


# driver --------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=DATA / "raw")
    ap.add_argument("--out", type=Path, default=DATA / "dataset")
    ap.add_argument("--separate", action="store_true", help="remove background music with Demucs")
    ap.add_argument("--min-sec", type=float, default=2.0)
    ap.add_argument("--max-sec", type=float, default=12.0)
    ap.add_argument("--min-speaker-sim", type=float, default=0.6)
    ap.add_argument("--min-asr-conf", type=float, default=0.90)
    args = ap.parse_args()

    work = DATA / "work"
    (work / "vocals").mkdir(parents=True, exist_ok=True)
    (work / "clips").mkdir(parents=True, exist_ok=True)
    index_path = work / "clips.jsonl"
    done = {json.loads(ln)["id"] for ln in index_path.open()} if index_path.exists() else set()
    sources = sorted(args.raw.glob("*.wav"))
    if not sources:
        raise SystemExit(f"No WAV files in {args.raw}. Run kintts-download first.")

    # Stages 1, 2, 4 per source file; results appended to clips.jsonl
    asr = None
    with index_path.open("a") as index:
        for src in sources:
            if any(d.startswith(src.stem + "_") for d in done):
                continue
            print(f"[{src.name}]")
            path = separate(src, work / "vocals") if args.separate else src
            wav = load_wav(path)
            segs = segment(wav, args.min_sec, args.max_sec)
            print(f"  {len(segs)} segments, {sum(e - s for s, e in segs) / SR / 60:.1f} min")
            asr = asr or ASR()
            for k, (s, e) in enumerate(segs):
                cid = f"{src.stem}_{k:05d}"
                clip = wav[s:e]
                peak = np.abs(clip).max()
                clip = clip / peak * 0.9 if peak > 0 else clip
                sf.write(work / "clips" / f"{cid}.wav", clip, SR)
                text, conf = asr(clip)
                index.write(json.dumps({"id": cid, "text": text, "seconds": (e - s) / SR,
                                        "asr_conf": conf}, ensure_ascii=False) + "\n")
            index.flush()

    # Stage 3 across all clips (needs the whole collection to find the host)
    rows = [json.loads(ln) for ln in index_path.open()]
    print(f"Speaker embeddings for {len(rows)} clips ...")
    emb_path = work / "embeddings.npy"
    ids_path = work / "embeddings.ids.json"
    cached = dict(zip(json.loads(ids_path.read_text()), np.load(emb_path))) if emb_path.exists() else {}
    missing = [r["id"] for r in rows if r["id"] not in cached]
    if missing:
        new = embed([load_wav(work / "clips" / f"{i}.wav") for i in missing])
        cached.update(zip(missing, new))
        ids_path.write_text(json.dumps(list(cached)))
        np.save(emb_path, np.stack(list(cached.values())))
    sims = main_speaker_similarity(np.stack([cached[r["id"]] for r in rows]))

    # Stage 5: filter + write dataset
    kept = []
    for r, sim in zip(rows, sims):
        r["speaker_sim"] = float(sim)
        cps = len(r["text"]) / r["seconds"]
        if (sim >= args.min_speaker_sim and r["asr_conf"] >= args.min_asr_conf
                and 6 <= cps <= 25 and len(r["text"]) > 5):
            kept.append(r)
    kept.sort(key=lambda r: -(r["asr_conf"] + r["speaker_sim"]))

    wav_dir = args.out / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    with (args.out / "metadata.csv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="|")
        w.writerow(["id", "text", "seconds", "speaker_sim", "asr_conf"])
        for r in kept:
            dst = wav_dir / f"{r['id']}.wav"
            if not dst.exists():
                dst.hardlink_to(work / "clips" / f"{r['id']}.wav")
            w.writerow([r["id"], r["text"], f"{r['seconds']:.2f}", f"{r['speaker_sim']:.3f}", f"{r['asr_conf']:.3f}"])

    total = sum(r["seconds"] for r in rows) / 3600
    good = sum(r["seconds"] for r in kept) / 3600
    print(f"Kept {len(kept)}/{len(rows)} clips: {good:.2f} h of {total:.2f} h -> {args.out}")


if __name__ == "__main__":
    main()
