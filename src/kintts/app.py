"""Web testing studio: paste Kinyarwanda text, listen, rate, and say what's wrong.

    uv run kintts-app                       # http://127.0.0.1:7860
    KINTTS_MODELS=checkpoints/host-v1 uv run kintts-app   # add fine-tuned model(s), comma-separated

Every rating is appended to feedback/feedback.csv together with the text, the
model, and the audio file, so each complaint can be traced and fixed.
"""

from __future__ import annotations

import csv
import datetime as dt
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

from kintts.compare import HFVits, IjwiLab
from kintts.normalize import normalize

ROOT = Path(__file__).resolve().parents[2]
FEEDBACK = ROOT / "feedback"
PROBLEMS = ["Tone / intonation wrong", "Word mispronounced", "Rhythm / speed unnatural",
            "Number or date read wrong", "Sounds robotic", "Noise / artifacts",
            "Words skipped or repeated", "English/French word wrong"]


def available_models() -> dict[str, str]:
    models = {"Baseline: MMS (Meta)": "facebook/mms-tts-kin"}
    for path in filter(None, (p.strip() for p in os.environ.get("KINTTS_MODELS", "").split(","))):
        models[f"Ours: {Path(path).name}"] = path
    ckpt_root = ROOT / "checkpoints"
    if ckpt_root.exists():
        for d in sorted(ckpt_root.iterdir()):
            if (d / "config.json").exists():
                models.setdefault(f"Ours: {d.name}", str(d))
    if os.environ.get("IJWILAB_API_KEY"):
        for spk in range(3):
            models[f"IjwiLab speaker {spk}"] = f"ijwilab:{spk}"
    return models


@lru_cache(maxsize=4)
def load(model_ref: str):
    if model_ref.startswith("ijwilab:"):
        return IjwiLab(int(model_ref.split(":")[1]))
    return HFVits(model_ref, use_normalizer=False)


def synthesize(text: str, model_name: str, use_norm: bool, speed: float):
    if not text.strip():
        return None, "", None
    ref = available_models()[model_name]
    spoken = normalize(text) if use_norm else text
    engine = load(ref)
    if isinstance(engine, HFVits):
        engine.model.speaking_rate = speed
    wav, sr = engine(spoken if not ref.startswith("ijwilab") else text)
    FEEDBACK.mkdir(exist_ok=True)
    path = FEEDBACK / "audio" / f"{dt.datetime.now():%Y%m%d-%H%M%S}.wav"
    path.parent.mkdir(exist_ok=True)
    sf.write(path, np.asarray(wav), sr)
    return (sr, np.asarray(wav)), spoken, str(path)


def save_feedback(text, model_name, audio_path, score, problems, comment, tester):
    if not audio_path:
        return "Generate audio first."
    FEEDBACK.mkdir(exist_ok=True)
    f = FEEDBACK / "feedback.csv"
    new = not f.exists()
    with f.open("a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["time", "tester", "model", "score", "problems", "comment", "text", "audio"])
        w.writerow([dt.datetime.now().isoformat(timespec="seconds"), tester, model_name, score,
                    "; ".join(problems), comment, text, Path(audio_path).name])
    return f"Saved. Thank you! ({f.relative_to(ROOT)})"


def build():
    import gradio as gr

    models = list(available_models())
    with gr.Blocks(title="Kinyarwanda TTS Studio") as ui:
        gr.Markdown("## Kinyarwanda TTS Studio\nPaste text, listen, then tell us exactly what sounds wrong.")
        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(label="Kinyarwanda text", lines=5,
                                  value="Muraho neza! Uyu munsi tuzaganira ku mateka y'u Rwanda.")
                with gr.Row():
                    model = gr.Dropdown(models, value=models[-1], label="Model")
                    speed = gr.Slider(0.7, 1.3, value=1.0, step=0.05, label="Speed")
                    use_norm = gr.Checkbox(True, label="Spell out numbers/dates")
                go = gr.Button("Speak", variant="primary")
            with gr.Column(scale=2):
                audio = gr.Audio(label="Result", autoplay=True)
                spoken = gr.Textbox(label="What the model actually read", interactive=False)
        audio_path = gr.State(None)

        gr.Markdown("### Your verdict")
        with gr.Row():
            score = gr.Radio([1, 2, 3, 4, 5], value=3, label="How native does it sound? (5 = can't tell it's a machine)")
            tester = gr.Textbox(label="Tester name", value="Joe")
        problems = gr.CheckboxGroup(PROBLEMS, label="What's wrong?")
        comment = gr.Textbox(label="Details (which word, how a native speaker would say it)", lines=2)
        save = gr.Button("Save feedback")
        status = gr.Markdown()

        go.click(synthesize, [text, model, use_norm, speed], [audio, spoken, audio_path])
        text.submit(synthesize, [text, model, use_norm, speed], [audio, spoken, audio_path])
        save.click(save_feedback, [text, model, audio_path, score, problems, comment, tester], status)
    return ui


def main() -> None:
    build().launch(server_name="127.0.0.1")


if __name__ == "__main__":
    main()
