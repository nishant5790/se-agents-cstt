import subprocess
import os
import sys
import re
import json
import base64
import math
import shutil
import numpy as np
from PIL import Image
from openai import AzureOpenAI
from dataclasses import dataclass, field, asdict
from typing import Optional, List
from pathlib import Path
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
	sys.path.insert(0, str(_ROOT))

from agent_team.llm import azure_openai_client
_PKG = Path(__file__).resolve().parent
load_dotenv(_PKG / ".env")  # package-local .env, regardless of cwd
load_dotenv()                # also honour a cwd/parent .env if present


# ──────────────────────────────────────────────
# 1. AZURE OPENAI CLIENT
# ──────────────────────────────────────────────

client, GPT4O_DEPLOYMENT = azure_openai_client()


# ──────────────────────────────────────────────
# 2. DATA MODEL
# ──────────────────────────────────────────────
@dataclass
class Block:
    index: int
    id: str
    source: str
    modality: str
    title: str
    timestamp: float
    image_ref: Optional[str]
    metadata: dict = field(default_factory=dict)
    text: str=""
    visual_description: Optional[str]=None

    def __str__(self):
        out = (
            f"[{self.index}] id={self.id}\n"
            f"source={self.source}\n"
            f"modality={self.modality}\n"
            f"title={self.title}\n"
            f"timestamp={self.timestamp}\n"
            f"image_ref={self.image_ref}\n"
            f"metadata={self.metadata}\n"
            f"text:\n{self.text}\n"
        )
        if self.visual_description:
            out += f"visual_description:\n{self.visual_description}\n"
        return out



# ──────────────────────────────────────────────
#  HELPERS for video analyser gpt based experimentation
# ──────────────────────────────────────────────

def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    return text.strip('-')

def encode_image_base64(image_path: str) -> str:
    try:
        with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return None


def encode_audio_base64(audio_path: str) -> str:
    try:
        with open(audio_path, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode("utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return None


def _resolve_media_tool(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found

    try:
        import imageio_ffmpeg

        ffmpeg_path = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if name == "ffmpeg":
            return str(ffmpeg_path)
        probe_name = "ffprobe.exe" if os.name == "nt" else "ffprobe"
        ffprobe_path = ffmpeg_path.with_name(probe_name)
        if ffprobe_path.exists():
            return str(ffprobe_path)
    except Exception:
        pass

    raise FileNotFoundError(
        f"{name} not found. Install ffmpeg or add it to PATH."
    )


def get_video_duration(video_path: str) -> float:
    try:
        cmd = [
            _resolve_media_tool("ffprobe"), "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception:
        # Fallback for environments that have ffmpeg but no ffprobe.
        ffmpeg_cmd = [_resolve_media_tool("ffmpeg"), "-i", video_path]
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        output = (result.stderr or "") + "\n" + (result.stdout or "")
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", output)
        if not match:
            raise RuntimeError("Unable to determine video duration from ffmpeg output")
        hours = int(match.group(1))
        minutes = int(match.group(2))
        seconds = float(match.group(3))
        return hours * 3600 + minutes * 60 + seconds



# ──────────────────────────────────────────────
# 4. FFMPEG — Extract audio chunk
# ──────────────────────────────────────────────


def extract_audio_chunk(video_path: str, start: float, duration: float, output_path: str):
    """Extract a chunk of audio as WAV."""
    command = [
        _resolve_media_tool("ffmpeg"), "-y",
        "-ss", str(start),
        "-i", video_path,
        "-t", str(duration),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        output_path
    ]
    subprocess.run(command, check=True, capture_output=True)


# ──────────────────────────────────────────────
# 5. FFMPEG — Extract frame at timestamp
# ──────────────────────────────────────────────

def extract_frame(video_path: str, timestamp: float, output_path: str) -> str:
    command = [
        _resolve_media_tool("ffmpeg"), "-y",
        "-ss", str(timestamp),
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        output_path
    ]
    subprocess.run(command, check=True, capture_output=True)
    return output_path


# ──────────────────────────────────────────────
# 6. SCENE CHANGE DETECTION
# ──────────────────────────────────────────────
def frames_are_similar(frame1: str, frame2: str, threshold: float = 0.95) -> bool:
    try:
        img1 = np.array(Image.open(frame1).resize((320, 180)).convert("RGB"), dtype=np.float32)
        img2 = np.array(Image.open(frame2).resize((320, 180)).convert("RGB"), dtype=np.float32)
        diff = np.abs(img1 - img2).mean()
        similarity = 1.0 - (diff / 255.0)
        return similarity >= threshold
    except Exception:
        return False


# ──────────────────────────────────────────────
# 7. GPT-4o — AUDIO ONLY (transcription)
# ──────────────────────────────────────────────
def transcribe_with_gpt4o(audio_path:str)-> str:

    """Send audio chunk to GPT-4o for transcription."""
    audio_b64 = encode_audio_base64(audio_path)

    response = client.chat.completions.create(
        model=GPT4O_DEPLOYMENT,
        modalities=["text"],
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a transcription assistant. "
                    "Transcribe the audio accurately. Return ONLY the transcript text. "
                    "If there is no speech, return '[silence]'."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_b64,
                            "format": "wav"
                        }
                    }
                ]
            }
        ],
        max_tokens=500,
        temperature=0.1
    )

    return response.choices[0].message.content.strip()

# ──────────────────────────────────────────────
# 7.1. Whisper — AUDIO ONLY (transcription)
# ──────────────────────────────────────────────
_WHISPER_MODEL = None


def _load_whisper_model():
    """Load (and cache) the local Whisper model so it isn't reloaded per chunk."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is not None:
        return _WHISPER_MODEL
    try:
        import whisper  # openai-whisper
    except ImportError as exc:  # pragma: no cover - guidance only
        raise RuntimeError(
            "openai-whisper not installed. Run: pip install openai-whisper"
        ) from exc

    # Honour corporate proxy certs for the model download (Azure CDN).
    try:
        from agent_team.llm import _ensure_tls_trust
        _ensure_tls_trust()
    except Exception:
        pass

    model_size = os.getenv("LOCAL_WHISPER_MODEL", "base")
    _WHISPER_MODEL = whisper.load_model(model_size)
    return _WHISPER_MODEL


def transcribe_with_whisper(audio_path: str) -> str:
    """Local Whisper transcription. Reads the 16 kHz mono WAV into a numpy array
    so Whisper does not need ffmpeg on PATH. Returns the joined transcript text."""
    import wave

    model = _load_whisper_model()

    # The chunk is already 16 kHz mono PCM s16le, decode it directly.
    with wave.open(audio_path, "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    audio = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0

    result = model.transcribe(audio)
    parts = []
    for s in result.get("segments", []):
        text = (s.get("text") or "").strip()
        if text:
            parts.append(text)
    ouptut_transcript = " ".join(parts).strip() or "[silence]"
    # print(f"  Whisper transcript: \n {ouptut_transcript}{'...' if len(ouptut_transcript) > 60 else ''} \n")
    print(f"  Whisper transcript: \n {ouptut_transcript}\n")
    return ouptut_transcript



# ──────────────────────────────────────────────
# 8. GPT-4o — VISION ONLY (frame description)
# ─────────────────────────────────────────────

def describe_frame_gpt4o(image_path: str, transcript_context: str = "") -> str:
    """Send frame to GPT-4o Vision for description."""
    img_b64 = encode_image_base64(image_path)

    response = client.chat.completions.create(
        model=GPT4O_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a visual analyzer for meeting/training videos. "
                    "Describe what is shown on screen concisely: slides, dashboards, "
                    "UI, diagrams, text on screen, people. Keep it under 2-3 sentences."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Describe this frame. Speaker context: '{transcript_context}'"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}",
                            "detail": "low"
                        }
                    }
                ]
            }
        ],
        max_tokens=200,
        temperature=0.3
    )

    return response.choices[0].message.content.strip()



# ──────────────────────────────────────────────
# 9. GPT-4o — MULTIMODAL (audio + image together)
#    Best mode: sends BOTH in one call
# ──────────────────────────────────────────────

def analyze_chunk_multimodal(audio_path: str, image_path: str) -> dict:
    """
    Send audio + frame to GPT-4o in a SINGLE call.
    Returns: {"transcript": "...", "visual_description": "..."}
    """
    audio_b64 = encode_audio_base64(audio_path)
    img_b64 = encode_image_base64(image_path)

    response = client.chat.completions.create(
        model=GPT4O_DEPLOYMENT,
        modalities=["text"],
        messages=[
            {
                "role": "system",
                "content": (
                    "You analyze meeting/training video chunks. "
                    "You receive an audio clip and a video frame from the same moment. "
                    "Return a JSON object with exactly these keys:\n"
                    '{"transcript": "exact transcription of the audio", '
                    '"visual_description": "concise description of what is shown on screen"}\n'
                    "If no speech, set transcript to '[silence]'. "
                    "Keep visual_description under 2 sentences."
                )
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_b64,
                            "format": "wav"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}",
                            "detail": "low"
                        }
                    }
                ]
            }
        ],
        max_tokens=500,
        temperature=0.2
    )

    raw = response.choices[0].message.content.strip()

    # Parse JSON from response
    try:
        # Handle markdown code blocks if present
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"transcript": raw, "visual_description": ""}


# ──────────────────────────────────────────────
# 10. MAIN PIPELINE
# ──────────────────────────────────────────────
def process_video(
        video_path: str,
        output_dir: str = "output",
        chunk_duration: float = 30.0,    # seconds per chun
        similarity_threshold: float = 0.95,
        mode: str = "multimodal", # "multimodal" | "separate" | "transcript_only"
        start_index: int = 0

    )-> List[Block]:

    """
    Full GPT-4o pipeline: Video → Audio chunks + Frames → GPT-4o → Blocks

    Args:
        video_path:             Path to input video
        output_dir:             Output directory
        chunk_duration:         Duration of each audio chunk (seconds)
        similarity_threshold:   Frame dedup threshold
        mode:
            "multimodal"      → audio + frame sent together in 1 GPT-4o call (BEST)
            "separate"        → audio and frame analyzed separately
            "transcript_only" → audio transcription only, no frame analysis
        start_index:            Starting block index
    """
    video_path = os.path.abspath(video_path)
    video_name = Path(video_path).stem
    video_filename = Path(video_path).stem
    slug = slugify(video_name)

    assets_dir = os.path.join(output_dir, "assets")
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(assets_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    # get video duration
    total_duration = get_video_duration(video_path)
    num_chunks = math.ceil(total_duration / chunk_duration)
    print(f"Video : {video_filename}")
    print(f"Duration: {total_duration:.1f}s → {num_chunks} chunks ({chunk_duration}s each)")

    # process each chunk
    blocks: list[Block] = []
    prev_frame_path = None

    for chunk_idx in range(num_chunks):
        start_time = chunk_idx * chunk_duration
        actual_duration = min(chunk_duration, total_duration - start_time)
        print(f"\n processing chunk {chunk_idx+1}/{num_chunks}"
              f"[{start_time:.1f}s - {start_time + actual_duration:.1f}s]")
        # extract audio chunk

        audio_chunk_path = os.path.join(temp_dir, f"chunk_{chunk_idx}.wav")
        extract_audio_chunk(video_path, start_time, actual_duration, audio_chunk_path)

        # extract frame( middle of chunk)
        frame_timestamp = start_time + (actual_duration/2)
        temp_frame = os.path.join(temp_dir, f"frame_{chunk_idx}.jpg")
        extract_frame(video_path, frame_timestamp, temp_frame)

        # scene change detection
        is_new_scene = ( prev_frame_path is None or
                         not frames_are_similar(prev_frame_path, temp_frame,similarity_threshold)
                         )
        # gpt 4o analysis
        transcript = ""
        visual_desc = None

        if mode == "multimodal":
            print("  GPT-4o multimodal analysis...")
            result = analyze_chunk_multimodal(audio_chunk_path, temp_frame)
            transcript = result["transcript"]
            if is_new_scene:
                visual_desc = result["visual_description"]


        elif mode == "separate":
            # Audio → transcript
            # print(f"  GPT-4o transcription...")
            # transcript = transcribe_with_gpt4o(audio_chunk_path)
            print(f"  transcription using whisper...")
            transcript = transcribe_with_whisper(audio_chunk_path)
            # Frame → description (only if new scene)
            if is_new_scene:
                print(f"  GPT-4o frame analysis...")
                visual_desc = describe_frame_gpt4o(temp_frame, transcript)

        elif mode == "transcript_only":
            # print(f"  GPT-4o transcription...")
            # transcript = transcribe_with_gpt4o(audio_chunk_path)

            print(f"  transcription using whisper ...")
            transcript = transcribe_with_whisper(audio_chunk_path)


        # ── Save frame if new scene ──
        timestamp = round(start_time, 1)
        timestamp_id = int(timestamp * 100)
        block_id = f"{slug}__t__{timestamp_id}"

        image_ref = None

        if is_new_scene:
            final_frame = os.path.join(assets_dir, f"frame_{block_id}.jpg")
            os.rename(temp_frame, final_frame)
            image_ref = os.path.abspath(final_frame)
            prev_frame_path = final_frame
        else:
            os.remove(temp_frame)

        # clean audio chunk
        os.remove(audio_chunk_path)

        # build block
        block = Block(
            index=start_index + chunk_idx,
            id = block_id,
            source=f"Input --{video_filename}",
            modality="transcript",
            title=f"{video_name} @ {int(timestamp)}s",
            timestamp=timestamp,
            image_ref=image_ref,
            metadata={"chunk_duration": actual_duration, "mode": mode},
            text=transcript,
            visual_description=visual_desc

        )
        print('--- BLOCK ------\n')
        print(block)
        print("\n" + "==" * 60)
        blocks.append(block)

    # cleanup temp dir
    os.rmdir(temp_dir)

    print(f" \n Pipeline Completed : {len(blocks)} blocks generated")
    return blocks


# ──────────────────────────────────────────────
# 11. SAVE BLOCKS
# ──────────────────────────────────────────────
def save_blocks(blocks: List[Block], output_dir: str = "outputs"):
    txt_path = os.path.join(output_dir, "blocks.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for block in blocks:
            f.write(str(block))
            f.write("\n\n")
    print(f"Saved: {txt_path}")

    json_path = os.path.join(output_dir, "blocks.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([asdict(b) for b in blocks], f, indent=2, ensure_ascii=False)
    print(f"Saved: {json_path}")



# ──────────────────────────────────────────────
# 12. ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":
    VIDEO_PATH = r"C:\Users\KNishant\Documents\rok_project\SE Agent\agent_team\inputs\INPUT--UAT Kickoff & Demo-20250331_103150-Meeting Recording.mp4"
    OUTPUT_DIR = "outputs1"

    blocks = process_video(
        video_path=VIDEO_PATH,
        output_dir=OUTPUT_DIR,
        chunk_duration=30,
        similarity_threshold=0.95,
        mode="separate",
        start_index=0
    )

    save_blocks(blocks, OUTPUT_DIR)

    # preview

    print("\n" + "=="*60)
    for block in blocks[:3]:
        print(block)
        print()

