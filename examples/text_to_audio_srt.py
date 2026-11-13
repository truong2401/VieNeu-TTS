"""Text -> one audio file + a timestamped .srt subtitle file that matches it.

Each sentence is synthesised on its own, so its exact duration is known; the
clips are laid back to back (with a short pause) and the SRT cue times are
computed from the real clip lengths, so subtitles line up with the audio.

Usage:
    uv run python examples/text_to_audio_srt.py script.txt
    uv run python examples/text_to_audio_srt.py script.txt -o out/bai1 --voice "Phạm Tuyên"
    uv run python examples/text_to_audio_srt.py script.txt --by-line     # 1 line = 1 cue
    uv run python examples/text_to_audio_srt.py --text "Xin chào. Đây là thử nghiệm."

Outputs: <out>.wav and <out>.srt (default: next to the input file).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

# Split after . ! ? … when followed by whitespace (keeps the punctuation).
_SENT = re.compile(r"(?<=[.!?…])\s+")


def split_text(text: str, by_line: bool) -> list[str]:
    parts: list[str] = []
    for line in text.replace("\r\n", "\n").split("\n"):
        line = line.strip()
        if not line:
            continue
        if by_line:
            parts.append(line)
        else:
            parts.extend(s.strip() for s in _SENT.split(line) if s.strip())
    return parts


def fmt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?", help="UTF-8 text file (omit if using --text)")
    ap.add_argument("--text", help="Text given directly on the command line")
    ap.add_argument("-o", "--output", help="Output path without extension (default: next to input, or ./output)")
    ap.add_argument("--voice", default=None, help="Preset voice name, e.g. 'Phạm Tuyên' (default: model default)")
    ap.add_argument("--ref-audio", default=None, help="Reference clip (3-8 s) for voice cloning")
    ap.add_argument("--by-line", action="store_true", help="One line of the input = one subtitle cue")
    ap.add_argument("--pause", type=float, default=0.3, help="Silence between cues, seconds (default 0.3)")
    args = ap.parse_args()

    if args.text:
        text = args.text
        out_base = Path(args.output or "output")
    elif args.input:
        src = Path(args.input)
        text = src.read_text(encoding="utf-8-sig")
        out_base = Path(args.output) if args.output else src.with_suffix("")
    else:
        ap.error("give an input file or --text")

    cues_text = split_text(text, args.by_line)
    if not cues_text:
        ap.error("no text to read")

    from vieneu import Vieneu

    tts = Vieneu()
    sr = tts.sample_rate
    kwargs = {}
    if args.ref_audio:
        kwargs["ref_audio"] = args.ref_audio
    elif args.voice:
        kwargs["voice"] = args.voice

    pause = np.zeros(int(sr * args.pause), dtype=np.float32)
    chunks: list[np.ndarray] = []
    srt_blocks: list[str] = []
    cursor = 0.0  # seconds

    for i, sentence in enumerate(cues_text, 1):
        audio = np.asarray(tts.infer(sentence, **kwargs), dtype=np.float32).reshape(-1)
        dur = len(audio) / sr
        start, end = cursor, cursor + dur
        srt_blocks.append(f"{i}\n{fmt_time(start)} --> {fmt_time(end)}\n{sentence}\n")
        chunks.append(audio)
        if i < len(cues_text):
            chunks.append(pause)
        cursor = end + args.pause
        print(f"[{i}/{len(cues_text)}] {fmt_time(start)} --> {fmt_time(end)}  {sentence[:60]}")

    out_base.parent.mkdir(parents=True, exist_ok=True)
    wav_path = out_base.with_suffix(".wav")
    srt_path = out_base.with_suffix(".srt")
    tts.save(np.concatenate(chunks), wav_path)
    srt_path.write_text("\n".join(srt_blocks), encoding="utf-8")
    print(f"\nSaved audio: {wav_path}\nSaved subtitles: {srt_path}")


if __name__ == "__main__":
    main()
