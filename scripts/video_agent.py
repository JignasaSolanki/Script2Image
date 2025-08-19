# scripts/video_agent.py
# CPU-only, free models. Creates a slideshow video with sliding transitions,
# captions, and a background audio track made from per-sentence TTS.
#
# Inputs:
#   - CLI arg text OR env INPUT_SCRIPT (from GitHub Actions).
#
# Outputs:
#   - generated_audio/final_audio.mp3
#   - generated_video/final_video.mp4
#
# Notes:
#   - Attempts to call your existing scripts/image_gen.py with the sentence text
#     and an --out path. If that script doesn't support --out, it should still
#     save to generated_images/. If no image is produced, we fall back to
#     rendering a simple caption card via Pillow so the pipeline never breaks.

import os
import re
import sys
import math
import subprocess
from pathlib import Path
from datetime import datetime

from PIL import Image, ImageDraw, ImageFont

from moviepy.editor import (
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    concatenate_videoclips,
    concatenate_audioclips
)

from TTS.api import TTS


# ----------------- Config -----------------
VIDEO_W, VIDEO_H = 1280, 720
MARGIN = 48
CAPTION_BOX_ALPHA = 180        # 0-255 black box opacity
CAPTION_FONT_SIZE = 44
CAPTION_MAX_WIDTH = VIDEO_W - 2*MARGIN
CAPTION_LINE_SPACING = 10
TRANSITION_SECONDS = 0.6       # slide transition length between sentences
BG_COLOR = (18, 18, 18)        # background when we need to letterbox / fill

AUDIO_DIR = Path("generated_audio")
IMAGE_DIR = Path("generated_images")
FRAME_DIR = Path("generated_frames")       # per-sentence framed images with captions
VIDEO_DIR = Path("generated_video")

MODEL = "tts_models/en/ljspeech/tacotron2-DDC"  # Free Coqui model (CPU)

# If you have a persistent font file you like, put it in repo and point here.
# Otherwise we use a default PIL font (not pretty but portable).
FONT_PATH = None  # e.g., "assets/Inter-SemiBold.ttf"

# ------------------------------------------


def ensure_dirs():
    for d in [AUDIO_DIR, IMAGE_DIR, FRAME_DIR, VIDEO_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def split_into_sentences(text: str):
    """
    Simple sentence splitter (avoids heavy NLTK downloads in CI).
    Splits on . ? ! followed by whitespace/newline.
    """
    text = text.strip()
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # Split on . ? ! followed by space/cap
    parts = re.split(r"(?<=[\.!?])\s+", text)
    # Clean up empties
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences


def wrap_caption(draw: ImageDraw.Draw, text: str, font, max_width: int):
    """
    Word-wrap text to fit within max_width in pixels.
    Returns list of lines.
    """
    words = text.split()
    lines, cur = [], []
    for w in words:
        test = " ".join(cur + [w])
        w_px, _ = draw.textsize(test, font=font)
        if w_px <= max_width:
            cur.append(w)
        else:
            if cur:
                lines.append(" ".join(cur))
            cur = [w]
    if cur:
        lines.append(" ".join(cur))
    return lines


def render_captioned_frame(base_img_path: Path, sentence: str, out_path: Path):
    """
    Take an image, resize/letterbox to VIDEO_WxVIDEO_H, and draw a semi-transparent
    caption box with wrapped text near bottom.
    If base_img_path doesn't exist, create a plain background with caption only.
    """
    # Base canvas
    canvas = Image.new("RGB", (VIDEO_W, VIDEO_H), BG_COLOR)

    # Load or make placeholder
    if base_img_path.exists():
        img = Image.open(base_img_path).convert("RGB")
    else:
        img = Image.new("RGB", (VIDEO_W, VIDEO_H), (30, 30, 30))

    # Fit image to canvas (contain)
    img_ratio = img.width / img.height
    can_ratio = VIDEO_W / VIDEO_H
    if img_ratio >= can_ratio:
        # width bound
        new_w = VIDEO_W
        new_h = int(new_w / img_ratio)
    else:
        # height bound
        new_h = VIDEO_H
        new_w = int(new_h * img_ratio)

    img = img.resize((new_w, new_h), Image.LANCZOS)
    x = (VIDEO_W - new_w) // 2
    y = (VIDEO_H - new_h) // 2
    canvas.paste(img, (x, y))

    # Caption box + text
    draw = ImageDraw.Draw(canvas, "RGBA")
    try:
        if FONT_PATH and Path(FONT_PATH).exists():
            font = ImageFont.truetype(FONT_PATH, CAPTION_FONT_SIZE)
        else:
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    lines = wrap_caption(draw, sentence, font, CAPTION_MAX_WIDTH)
    line_heights = [draw.textsize(line, font=font)[1] for line in lines]
    text_h = sum(line_heights) + CAPTION_LINE_SPACING*(len(lines)-1)
    box_pad_x, box_pad_y = 20, 14

    box_w = CAPTION_MAX_WIDTH + 2*box_pad_x
    box_h = text_h + 2*box_pad_y
    box_x = (VIDEO_W - box_w)//2
    box_y = VIDEO_H - box_h - MARGIN

    # Draw semi-transparent black rect
    draw.rectangle(
        [box_x, box_y, box_x + box_w, box_y + box_h],
        fill=(0, 0, 0, CAPTION_BOX_ALPHA)
    )

    # Draw text centered
    cur_y = box_y + box_pad_y
    for line in lines:
        tw, th = draw.textsize(line, font=font)
        tx = (VIDEO_W - tw)//2
        draw.text((tx, cur_y), line, font=font, fill=(255, 255, 255))
        cur_y += th + CAPTION_LINE_SPACING

    canvas.save(out_path)


def tts_sentence_to_mp3(sentence: str, idx: int) -> Path:
    """
    Generate per-sentence MP3 using Coqui TTS on CPU.
    """
    tts = TTS(MODEL, progress_bar=False, gpu=False)
    AUDIO_DIR.mkdir(exist_ok=True)
    wav_path = AUDIO_DIR / f"sent_{idx:03d}.wav"
    mp3_path = AUDIO_DIR / f"sent_{idx:03d}.mp3"

    # create WAV
    tts.tts_to_file(text=sentence, file_path=str(wav_path))

    # convert WAV -> MP3 with ffmpeg
    subprocess.run([
        "ffmpeg", "-y", "-i", str(wav_path),
        "-codec:a", "libmp3lame", "-qscale:a", "2",
        str(mp3_path)
    ], check=True)

    # cleanup wav
    try:
        wav_path.unlink()
    except Exception:
        pass

    return mp3_path


def try_generate_image(sentence: str, idx: int) -> Path:
    """
    Call your existing image_gen.py to produce ONE image for this sentence.
    If nothing appears, we fallback to a blank.
    Expected: scripts/image_gen.py "<sentence>" --out generated_images/sent_XXX.png
    Adjust the command below if your script uses a different CLI.
    """
    IMAGE_DIR.mkdir(exist_ok=True)
    out_path = IMAGE_DIR / f"sent_{idx:03d}.png"

    # Try invoke user's generator
    try:
        cmd = ["python", "scripts/generate_images_hfapi.py", sentence, "--out", str(out_path)]
        subprocess.run(cmd, check=True)
    except Exception:
        # If generator fails, we'll fallback.
        pass

    # If still missing, fallback to blank; the caption will still appear
    if not out_path.exists():
        img = Image.new("RGB", (VIDEO_W, VIDEO_H), (40, 40, 40))
        img.save(out_path)

    return out_path


def make_slide_transition_clip(prev_img_path: Path, next_img_path: Path, duration=TRANSITION_SECONDS):
    """
    Creates a left-to-right sliding transition between two images.
    prev slides left out; next slides in from right.
    """
    prev_clip = ImageClip(str(prev_img_path)).resize((VIDEO_W, VIDEO_H))
    next_clip = ImageClip(str(next_img_path)).resize((VIDEO_W, VIDEO_H))

    def prev_pos(t):
        # t: 0..duration -> x moves from 0 to -VIDEO_W
        x = - (t / duration) * VIDEO_W
        return (x, 0)

    def next_pos(t):
        # t: 0..duration -> x moves from VIDEO_W to 0
        x = VIDEO_W - (t / duration) * VIDEO_W
        return (x, 0)

    comp = CompositeVideoClip(
        [
            prev_clip.set_start(0).set_position(prev_pos),
            next_clip.set_start(0).set_position(next_pos)
        ],
        size=(VIDEO_W, VIDEO_H)
    ).set_duration(duration)

    return comp


def main():
    ensure_dirs()

    # 1) Read text
    text = " ".join(sys.argv[1:]).strip() or os.environ.get("INPUT_SCRIPT", "").strip()
    if not text:
        raise SystemExit("No input text provided. Pass text as CLI arg or set INPUT_SCRIPT env.")

    sentences = split_into_sentences(text)
    if not sentences:
        raise SystemExit("Could not split text into sentences.")

    # 2) For each sentence: TTS and image (and captioned frame)
    sentence_mp3s = []
    frame_paths = []

    for i, sent in enumerate(sentences, start=1):
        # TTS per sentence (to get precise durations)
        mp3_path = tts_sentence_to_mp3(sent, i)
        sentence_mp3s.append(mp3_path)

        # Image for the sentence
        img_path = try_generate_image(sent, i)

        # Captioned frame (draws caption on image)
        framed_path = FRAME_DIR / f"frame_{i:03d}.png"
        render_captioned_frame(img_path, sent, framed_path)
        frame_paths.append(framed_path)

    # 3) Build audio track by concatenating all sentence mp3s
    audio_clips = [AudioFileClip(str(p)) for p in sentence_mp3s]
    full_audio = concatenate_audioclips(audio_clips)
    final_audio_path = AUDIO_DIR / "final_audio.mp3"
    full_audio.write_audiofile(str(final_audio_path), codec="libmp3lame")

    # Also collect per-sentence durations (after write, clips have durations)
    durations = [AudioFileClip(str(p)).duration for p in sentence_mp3s]

    # 4) Build video clips per sentence (each still frame lasts exactly its sentence duration)
    body_clips = []
    for framed_path, dur in zip(frame_paths, durations):
        # slight in-clip drift (gentle pan) for visual life
        img_clip = ImageClip(str(framed_path)).set_duration(dur).resize((VIDEO_W, VIDEO_H))
        body_clips.append(img_clip)

    # 5) Insert slide transitions between clips
    # We shorten each body clip by half the transition on both sides (except first/last)
    # then interleave with slide transition clips.
    if len(body_clips) == 1:
        final_video = body_clips[0]
    else:
        adjusted = []
        for idx, clip in enumerate(body_clips):
            if idx == 0 or idx == len(body_clips) - 1:
                adjusted.append(clip)
            else:
                new_dur = max(0.1, clip.duration - TRANSITION_SECONDS)
                adjusted.append(clip.set_duration(new_dur))

        assembled = [adjusted[0]]
        for i in range(len(adjusted) - 1):
            trans = make_slide_transition_clip(frame_paths[i], frame_paths[i+1], TRANSITION_SECONDS)
            assembled.append(trans)
            assembled.append(adjusted[i+1])

        final_video = concatenate_videoclips(assembled, method="compose")

    # 6) Set audio & write video
    final_video = final_video.set_audio(AudioFileClip(str(final_audio_path)))
    final_video_fpath = VIDEO_DIR / "final_video.mp4"

    # H.264 + AAC, CPU-friendly preset
    final_video.write_videofile(
        str(final_video_fpath),
        fps=30,
        codec="libx264",
        audio_codec="aac",
        bitrate="3000k",
        preset="medium"
    )

    print(f"\n✅ Done!\nAudio: {final_audio_path}\nVideo: {final_video_fpath}")


if __name__ == "__main__":
    main()
