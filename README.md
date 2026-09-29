# Kinyarwanda TTS

Goal: Kinyarwanda speech that native speakers cannot tell apart from a real person.

**How we measure it:** in blind listening tests, native raters score our clips
as natural as real recordings of the same speaker, within 0.1 MOS (mean opinion
score, 1–5). In an A/B test they pick "the human" about 50% of the time.
Until then, it isn't done.

## Funding demo: 15 October 2026 (internal, non-commercial)

This demo is a proof of concept for internal use only. The voice comes from the
*Mwanafunzi Waruziko* podcast and documentaries. That audio and the models
trained on it stay private: `data/` and `checkpoints/` are git-ignored and are
never pushed to any hub. After funding comes a commercial model trained on 5
contracted speakers.

**Cost:** $0. We fine-tune an existing model instead of training from scratch,
on Kaggle's free GPU (30 h/week).

| Date | Step | Command |
|---|---|---|
| Sep 30 – Oct 2 | Download episodes and build the dataset (let it run overnight) | `uv run kintts-download`, then `uv run kintts-prepare` |
| Oct 2 | Export the best 5–8 h of clips, upload to Kaggle as a **private** dataset | `uv run kintts-export --max-hours 8` |
| Oct 2–5 | Training run 1 on Kaggle (`train/kaggle_finetune.ipynb`) | about 8–12 h of GPU |
| Oct 5–8 | Joe tests in the studio, tags failures, we fix them (normalizer, bad clips, more epochs) | `uv run kintts-app` |
| Oct 8–11 | Training run 2 with the fixes | Kaggle |
| Oct 12–14 | Blind MOS test with 5+ native colleagues: baseline vs ours vs IjwiLab vs the real host | `uv run kintts-compare` |
| Oct 15 | Demo: live studio plus the MOS chart | |

## Quick start

```bash
uv sync --all-extras
uv run pytest                          # text normalizer tests
uv run kintts-app                      # web studio: paste text -> listen -> rate -> feedback/feedback.csv
uv run kintts-normalize "Ni 5,000 Frw ku wa 05/06/2026"
uv run kintts-compare                  # blind MOS page -> eval/runs/baseline/index.html
IJWILAB_API_KEY=ijw_... uv run kintts-compare --systems mms-norm,ijwilab:0,hf:checkpoints/host-v1 --run vs-ijwilab
```

Data pipeline (`data/`, git-ignored):
```bash
echo "https://www.youtube.com/@<channel>/videos" >> data/sources.txt
uv run kintts-download                 # -> data/raw/*.wav (16 kHz mono)
uv run kintts-prepare                  # VAD -> main-host filter -> Kinyarwanda ASR -> data/dataset/
uv run kintts-prepare --separate       # same, but strips background music first (slow on CPU)
uv run kintts-export --max-hours 8     # -> data/export/kin_host.zip for Kaggle
```

## Why this is hard (and where competitors fall short)

1. **Tone and vowel length aren't written.** *inzara* can mean "hunger" or
   "fingernails" depending on tone, and the spelling doesn't show which. The
   model has to infer tone from context. That needs a model with wide context
   and enough data, or a tone-annotated lexicon.
2. **Text normalization.** Numbers agree with the noun class
   (*amafaranga ibihumbi bitanu*, *miliyoni ebyiri*). See `src/kintts/normalize.py`.
3. **Code-switching.** Real Kinyarwanda mixes in English and French
   (*meeting, internet, telefone*).
4. **Data.** Existing open data is mostly Bible readings (formal and monotone)
   or Common Voice (noisy crowd recordings). Neither sounds like everyday speech.

## Roadmap

### Phase 0: Baseline & evaluation (in progress)
- [x] MMS-TTS-kin baseline, raw vs normalized text
- [x] Blind MOS listening page
- [x] Number / money / date / phone normalizer (needs native review)
- [ ] Native speaker reviews `eval/sentences.txt` and the normalizer tests; grow to 100+ sentences
- [ ] Add IjwiLab (API key) and the Digital Umuganda YourTTS model to the comparison
- [ ] Add **real human recordings** of the test sentences as a hidden reference. This is the target to match.

### Phase 1: Data (the most important phase)
- 1 professional voice (radio or news voice actor, with a signed commercial-use consent)
- **15–25 hours**, studio or treated room, 44.1/48 kHz, one mic, same setup every session
- Script: modern, varied text (news, conversation, questions, exclamations,
  numbers, names, code-switched sentences). Cover every syllable
  (e.g. *nshy*, *mbw*, *cy*, *rw*) and both long and short vowels.
- Mark tone and vowel length on a subset (see Phase 3)
- Quality check: automatic loudness, clipping and silence checks, plus
  speech recognition (ASR) that compares each clip to its script line and flags mismatches
- Then a 2nd and 3rd voice (male/female, different ages) for "named speakers"

### Phase 2: Model v1: fast, cheap, deployable
- Fine-tune **VITS / Piper** starting from MMS weights on the Phase 1 data
  (about 1–2 days on one rented A100 or 4090, roughly $20–60)
- Runs in real time on a CPU and exports to ONNX for mobile and edge
- Target MOS ≥ 4.0

### Phase 3: Model v2: native-level
- Fine-tune a modern large model: **F5-TTS** (flow matching) or an
  LLM-style TTS (e.g. XTTS-v2, Orpheus-style), with a Kinyarwanda character vocabulary
- These models carry more context, which helps with tone and expressive "moods"
  (news, conversational, advertising)
- Optional tone input: annotate high tones on part of the data and train a
  small tone predictor from text
- Voice cloning from a short reference clip (IjwiLab only offers this in English)
- Target: within 0.1 MOS of the human recordings

### Phase 4: Product
- HTTP API (REST plus streaming PCM for phone calls and voice agents; `8k μ-law` for telephony)
- Web studio: voices, moods, speed, pronunciation dictionary per customer
- SSML-lite: pauses, emphasis, spell-out, "say-as" number and date
- SDKs, usage metering, on-prem option for government, banks and telcos

## Licensing notes
- `facebook/mms-tts-kin` is **CC-BY-NC**: use it for research and benchmarking only.
  Commercial models must be trained from permissively licensed weights or from scratch.
- Check the licenses of F5-TTS and XTTS checkpoints before selling (XTTS uses the Coqui Public Model License).
- Your own recorded data, with signed consent, is your main asset and your competitive advantage.

## Layout
```
src/kintts/normalize.py        text -> speakable Kinyarwanda
src/kintts/compare.py          multi-system synthesis + blind MOS page
src/kintts/listening_page.html rating UI template
src/kintts/app.py              web testing studio (Gradio)
src/kintts/data/               download / prepare / export pipeline
train/kaggle_finetune.ipynb    free-GPU fine-tuning of MMS-VITS
eval/sentences.txt             test sentences
tests/                         normalizer tests (native-verified cases)
```
