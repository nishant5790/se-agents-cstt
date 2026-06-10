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
load_dotenv()


# ──────────────────────────────────────────────
# 1. SETUP
# ──────────────────────────────────────────────

WHISPER_MODEL_SIZE = os.getenv("LOCAL_WHISPER_MODEL", "base")
whisper_model = None

client, VISION_MODEL = azure_openai_client()

# Transcription backend: 'azure' uses Azure OpenAI Whisper deployment,
# anything else (e.g. 'local') falls back to the local openai-whisper model.
_AZURE_WHISPER_DEPLOYMENT = os.getenv("AZURE_OPENAI_WHISPER_DEPLOYMENT", "whisper")
_BACKEND_PREF = os.getenv("MEDIA_TRANSCRIBE_BACKEND", "azure").strip().lower()
_AZURE_AVAILABLE = bool(
    os.getenv("AZURE_OPENAI_ENDPOINT")
    and os.getenv("AZURE_OPENAI_API_KEY")
    and _AZURE_WHISPER_DEPLOYMENT
)
TRANSCRIBE_BACKEND = "azure" if (_BACKEND_PREF == "azure" and _AZURE_AVAILABLE) else "local"


# ──────────────────────────────────────────────
# 2. DATA MODEL
# ──────────────────────────────────────────────
@dataclass
class FrameInfo:
    """Single frame within a block."""
    timestamp: float
    image_path: str
    visual_description: Optional[str] = None

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "image_path": self.image_path,
            "visual_description": self.visual_description
        }


@dataclass
class Block:
    index: int
    id: str
    source: str
    modality: str
    title: str
    timestamp: float
    image_ref: Optional[str]
    frames: List[FrameInfo] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    text: str = ""

    def __str__(self):
        out = (
            f"[{self.index}] id={self.id}\n"
            f"source={self.source}\n"
            f"modality={self.modality}\n"
            f"title={self.title}\n"
            f"timestamp={self.timestamp}\n"
            f"image_ref={self.image_ref}\n"
            f"frames_count={len(self.frames)}\n"
        )
        for i, frame in enumerate(self.frames):
            out += f"  frame[{i}]: {frame.image_path} @ {frame.timestamp}s\n"
            if frame.visual_description:
                out += f"    visual: {frame.visual_description}\n"
        out += (
            f"metadata={self.metadata}\n"
            f"text:\n{self.text}\n"
        )
        return out

    def to_dict(self):
        return {
            "index": self.index,
            "id": self.id,
            "source": self.source,
            "modality": self.modality,
            "title": self.title,
            "timestamp": self.timestamp,
            "image_ref": self.image_ref,
            "frames": [f.to_dict() for f in self.frames],
            "metadata": self.metadata,
            "text": self.text
        }


# ──────────────────────────────────────────────
# 3. HELPERS
# ──────────────────────────────────────────────
def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9]+', '-', text)
    return text.strip('-')


def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

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
# 4. FFMPEG — Audio & Frame Extraction
# ──────────────────────────────────────────────
def extract_audio_chunk(video_path: str, start: float, duration: float, output_path: str):
    """Extract audio chunk as WAV for Whisper."""
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


def extract_frame(video_path: str, timestamp: float, output_path: str) -> str:
    """Extract single frame at timestamp."""
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
# 5. SCENE CHANGE DETECTION
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
# 6. PER-CHUNK: TRANSCRIBE (Whisper local)
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



def _transcribe_local(audio_path: str) -> str:
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
    return " ".join(parts).strip() or "[silence]"


def _transcribe_azure(audio_path: str) -> str:
    """Azure OpenAI Whisper transcription. Sends the WAV chunk to the deployed
    Whisper model and returns the joined transcript text."""
    if client is None:
        raise RuntimeError("Azure OpenAI client unavailable for Whisper transcription")
    with open(audio_path, "rb") as fh:
        resp = client.audio.transcriptions.create(
            model=_AZURE_WHISPER_DEPLOYMENT,
            file=fh,
            response_format="verbose_json",
        )
    segments = getattr(resp, "segments", None)
    if segments is None and isinstance(resp, dict):
        segments = resp.get("segments")
    parts: list[str] = []
    if segments:
        for seg in segments:
            text = (seg.get("text") if isinstance(seg, dict) else getattr(seg, "text", "")) or ""
            text = text.strip()
            if text:
                parts.append(text)
    if not parts:
        full = getattr(resp, "text", "") or (resp.get("text", "") if isinstance(resp, dict) else "")
        if full.strip():
            parts.append(full.strip())
    return " ".join(parts).strip() or "[silence]"


def transcribe_with_whisper(audio_path: str) -> str:
    """Dispatch to Azure Whisper if configured (MEDIA_TRANSCRIBE_BACKEND=azure and
    Azure creds present), otherwise fall back to local openai-whisper."""
    if TRANSCRIBE_BACKEND == "azure":
        try:
            output_transcript = _transcribe_azure(audio_path)
        except Exception as exc:
            print(f"  ⚠️  Azure Whisper failed ({exc}); falling back to local Whisper.")
            output_transcript = _transcribe_local(audio_path)
    else:
        output_transcript = _transcribe_local(audio_path)
    print(f"  Whisper transcript: \n {output_transcript}\n")
    return output_transcript



# ──────────────────────────────────────────────
# 7. PER-CHUNK: EXTRACT & FILTER FRAMES
# ──────────────────────────────────────────────
def extract_and_filter_frames(
    video_path: str,
    chunk_start: float,
    chunk_end: float,
    temp_dir: str,
    prev_frame_path: Optional[str],
    frame_interval: float = 5.0,
    similarity_threshold: float = 0.95
) -> tuple:
    """
    Extract frames within chunk, keep only unique ones.

    Returns:
        (unique_frames: [(timestamp, path)], last_frame_path)
    """
    # Step A: Extract candidate frames
    candidates = []
    ts = chunk_start
    while ts < chunk_end:
        frame_path = os.path.join(temp_dir, f"temp_{ts:.1f}.jpg")
        extract_frame(video_path, ts, frame_path)
        candidates.append((ts, frame_path))
        ts += frame_interval

    # Step B: Filter — keep only visually unique frames
    unique = []
    last_kept = prev_frame_path

    for ts, path in candidates:
        if last_kept is None or not frames_are_similar(last_kept, path, similarity_threshold):
            unique.append((ts, path))
            last_kept = path
        else:
            os.remove(path)

    return unique, last_kept


# ──────────────────────────────────────────────
# 8. PER-CHUNK: DESCRIBE FRAMES (GPT-4o Vision)
# ──────────────────────────────────────────────
def describe_frames_gpt4o(
    frames: List,
    transcript_context: str
) -> List[FrameInfo]:
    """
    Send each unique frame to GPT-4o for visual description.
    Returns list of FrameInfo objects.
    """
    frame_infos = []

    for ts, path in frames:
        visual_desc = None
        try:
            img_b64 = encode_image_base64(path)

            response = client.chat.completions.create(
                model=VISION_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a visual analyzer for meeting/training videos. "
                            "Describe what is shown on screen concisely: slides, dashboards, "
                            "UI screens, diagrams, text on screen, people, etc. "
                            "Keep it under 2-3 sentences. Focus on information content."
                        )
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": f"Describe this video frame. Speaker is saying: '{transcript_context}'"
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
            visual_desc = response.choices[0].message.content.strip()
            print(f"      👁️  Described frame @ {ts:.1f}s")

        except Exception as e:
            print(f"      ⚠️  Vision error @ {ts:.1f}s: {e}")

        frame_infos.append(FrameInfo(
            timestamp=ts,
            image_path=os.path.abspath(path),
            visual_description=visual_desc
        ))

    # print(f" the frame info of the chunks : \n {frame_infos}.....\n")

    return frame_infos


# ──────────────────────────────────────────────
# 9. PROCESS SINGLE CHUNK (all 3 steps together)
# ──────────────────────────────────────────────
def process_single_chunk(
    video_path: str,
    chunk_idx: int,
    chunk_start: float,
    chunk_end: float,
    slug: str,
    video_filename: str,
    video_name: str,
    assets_dir: str,
    temp_dir: str,
    prev_frame_path: Optional[str],
    frame_interval: float,
    similarity_threshold: float,
    analyze_frames: bool,
    start_index: int
) -> tuple:
    """
    Process ONE chunk completely:
      Step 1: Extract audio → Whisper → transcript
      Step 2: Extract frames → scene detect → keep unique
      Step 3: GPT-4o describe unique frames
      Step 4: Build block

    Returns:
        (block, last_frame_path, vision_calls)
    """
    actual_duration = chunk_end - chunk_start
    timestamp = round(chunk_start, 1)
    timestamp_id = int(timestamp * 100)
    block_id = f"{slug}__t__{timestamp_id}"

    print(f"\n{'─' * 60}")
    print(f"📦 CHUNK {chunk_idx + 1} [{chunk_start:.1f}s → {chunk_end:.1f}s] ({actual_duration:.1f}s)")
    print(f"{'─' * 60}")

    # ── STEP 1: Transcribe ──
    print(f"  📝 Step 1: Extracting audio & transcribing...")
    audio_path = os.path.join(temp_dir, f"audio_chunk_{chunk_idx}.wav")
    extract_audio_chunk(video_path, chunk_start, actual_duration, audio_path)
    transcript = transcribe_with_whisper(audio_path)
    os.remove(audio_path)
    print(f"     ✅ Transcript: \"{transcript[:80]}...\"" if len(transcript) > 80
          else f"     ✅ Transcript: \"{transcript}\"")

    # ── STEP 2: Extract & filter frames ──
    print(f"  🖼️  Step 2: Extracting frames (every {frame_interval}s)...")
    unique_frames, last_frame = extract_and_filter_frames(
        video_path, chunk_start, chunk_end,
        temp_dir, prev_frame_path,
        frame_interval, similarity_threshold
    )
    num_candidates = max(1, int(actual_duration / frame_interval))
    print(f"     📸 Extracted: {num_candidates} → Unique: {len(unique_frames)}")

    # ── STEP 3: GPT-4o describe frames ──
    vision_calls = 0
    frame_infos = []

    if unique_frames and analyze_frames:
        print(f"  👁️  Step 3: GPT-4o analyzing {len(unique_frames)} unique frames...")
        frame_infos = describe_frames_gpt4o(unique_frames, transcript)
        vision_calls = len(unique_frames)
    elif unique_frames:
        # No GPT-4o, just save frames
        frame_infos = [
            FrameInfo(timestamp=ts, image_path=os.path.abspath(path), visual_description=None)
            for ts, path in unique_frames
        ]
    else:
        print(f"  ⏭️  Step 3: No new frames (same scene as previous chunk)")

    # ── Move frames to assets dir with proper names ──
    for i, fi in enumerate(frame_infos):
        frame_ts_id = int(fi.timestamp * 100)
        frame_filename = f"{block_id}__f__{frame_ts_id}.jpg"
        final_path = os.path.join(assets_dir, frame_filename)

        if os.path.exists(fi.image_path) and fi.image_path != final_path:
            os.rename(fi.image_path, final_path)
            fi.image_path = os.path.abspath(final_path)

    # ── STEP 4: Build block ──
    block = Block(
        index=start_index + chunk_idx,
        id=block_id,
        source=f"INPUT--{video_filename}",
        modality="transcript",
        title=f"{video_name} @ {int(timestamp)}s",
        timestamp=timestamp,
        image_ref=frame_infos[0].image_path if frame_infos else None,
        frames=frame_infos,
        metadata={
            "chunk_duration": actual_duration,
            "frames_extracted": num_candidates,
            "frames_kept": len(unique_frames)
        },
        text=transcript
    )
    
    print(f"  ✅ Block [{block.index}] built — {len(frame_infos)} frames, {vision_calls} API calls")
    return block, last_frame, vision_calls


# ──────────────────────────────────────────────
# 10. MAIN PIPELINE
# ──────────────────────────────────────────────
def process_video(
    video_path: str,
    output_dir: str = "outputs2",
    chunk_duration: float = 30.0,
    frame_interval: float = 5.0,
    similarity_threshold: float = 0.95,
    analyze_frames: bool = True,
    start_index: int = 0
) -> List:
    """
    Chunk-by-chunk pipeline:
      For each chunk → transcribe → extract frames → describe → build block

    Args:
        video_path:             Path to input video
        output_dir:             Output directory
        chunk_duration:         Duration of each chunk in seconds
        frame_interval:         Extract 1 frame every N seconds within chunk
        similarity_threshold:   Frame dedup threshold (0.0–1.0)
        analyze_frames:         If True, use GPT-4o for frame descriptions
        start_index:            Starting block index
    """
    video_path = os.path.abspath(video_path)
    video_name = Path(video_path).stem
    video_filename = Path(video_path).name
    slug = slugify(video_name)

    assets_dir = os.path.join(output_dir, "assets")
    temp_dir = os.path.join(output_dir, "temp")
    os.makedirs(assets_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    total_duration = get_video_duration(video_path)
    num_chunks = math.ceil(total_duration / chunk_duration)

    print(f"{'=' * 60}")
    print(f"📹 Video:          {video_filename}")
    print(f"⏱️  Duration:       {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"📦 Chunks:         {num_chunks} × {chunk_duration}s")
    print(f"🖼️  Frame interval: Every {frame_interval}s")
    if TRANSCRIBE_BACKEND == "azure":
        print(f"🎤 Transcription:  Azure Whisper ({_AZURE_WHISPER_DEPLOYMENT})")
    else:
        print(f"🎤 Transcription:  Local Whisper ({WHISPER_MODEL_SIZE})")
    print(f"👁️  Vision:         Azure GPT-4o ({VISION_MODEL})")
    print(f"{'=' * 60}")

    # ── Process chunk by chunk ──
    blocks: List[Block] = []
    prev_frame_path = None
    total_vision_calls = 0
    total_frames_saved = 0

    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * chunk_duration
        chunk_end = min(chunk_start + chunk_duration, total_duration)

        block, prev_frame_path, vision_calls = process_single_chunk(
            video_path=video_path,
            chunk_idx=chunk_idx,
            chunk_start=chunk_start,
            chunk_end=chunk_end,
            slug=slug,
            video_filename=video_filename,
            video_name=video_name,
            assets_dir=assets_dir,
            temp_dir=temp_dir,
            prev_frame_path=prev_frame_path,
            frame_interval=frame_interval,
            similarity_threshold=similarity_threshold,
            analyze_frames=analyze_frames,
            start_index=start_index
        )

        blocks.append(block)
        total_vision_calls += vision_calls
        total_frames_saved += len(block.frames)

    # ── Cleanup temp ──
    for f in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, f))
    os.rmdir(temp_dir)

    # ── Summary ──
    print(f"\n{'=' * 60}")
    print(f"🎉 PIPELINE COMPLETE")
    print(f"   📝 Total blocks:         {len(blocks)}")
    print(f"   🖼️  Total frames saved:   {total_frames_saved}")
    print(f"   👁️  Total GPT-4o calls:   {total_vision_calls}")
    print(f"   💰 Est. vision cost:     ~${total_vision_calls * 0.01:.2f}")
    print(f"{'=' * 60}")

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
    print(f"📄 Saved: {txt_path}")

    json_path = os.path.join(output_dir, "blocks.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([b.to_dict() for b in blocks], f, indent=2, ensure_ascii=False)
    print(f"📄 Saved: {json_path}")


# ──────────────────────────────────────────────
# 12. ENTRY POINT
# ──────────────────────────────────────────────
if __name__ == "__main__":

    VIDEO_PATH = r"C:\Users\KNishant\Documents\rok_project\SE Agent\agent_team\inputs\INPUT--UAT Kickoff & Demo-20250331_103150-Meeting Recording.mp4"
    OUTPUT_DIR = "outputs6"

    blocks = process_video(
        video_path=VIDEO_PATH,
        output_dir=OUTPUT_DIR,
        chunk_duration=60.0,            # 90 second chunks
        frame_interval=5.0,             # 1 frame every 5 seconds
        similarity_threshold=0.95,      # scene change sensitivity
        analyze_frames=True,            # GPT-4o vision ON
        start_index=0
    )

    save_blocks(blocks, OUTPUT_DIR)

    # Preview first 3 blocks
    print("\n" + "=" * 60)
    print("PREVIEW")
    print("=" * 60)
    for block in blocks[:3]:
        print(block)
        print()