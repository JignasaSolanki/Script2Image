# scripts/tts_audio.py
from pathlib import Path
from datetime import datetime
import os, subprocess, sys
from TTS.api import TTS

AUDIO_DIR = Path("generated_audio")
AUDIO_DIR.mkdir(exist_ok=True)

# Free English model (Coqui TTS, runs offline on CPU)
MODEL = "tts_models/en/ljspeech/tacotron2-DDC"

def make_audio(text: str):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    wav_path = AUDIO_DIR / f"voice_{ts}.wav"
    mp3_path = AUDIO_DIR / f"voice_{ts}.mp3"

    # Generate wav with Coqui
    tts = TTS(MODEL, progress_bar=False, gpu=False)
    tts.tts_to_file(text=text, file_path=str(wav_path))

    # Convert wav → mp3 (ffmpeg)
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(wav_path),
         "-codec:a", "libmp3lame", "-qscale:a", "2", str(mp3_path)],
        check=True
    )
    wav_path.unlink(missing_ok=True)
    print(f"✅ Audio saved: {mp3_path}")

if __name__ == "__main__":
    # Take text from command arg OR GitHub Actions env
    text = " ".join(sys.argv[1:]) or os.environ.get("INPUT_SCRIPT", "")
    if not text.strip():
        raise SystemExit("❌ No text input provided")
    make_audio(text)
