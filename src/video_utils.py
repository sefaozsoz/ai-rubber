"""Video frame extraction and writing helpers."""

import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MAX_PROCESS_WIDTH = 960  # keep VRAM usage sane on 6 GB GPUs


def write_image(path: Path, image: np.ndarray, quality: int | None = None) -> None:
    """Unicode-safe imwrite: cv2.imwrite fails silently on non-ASCII Windows paths."""
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if quality else []
    ok, buffer = cv2.imencode(path.suffix, image, params)
    if not ok:
        raise IOError(f"Kare kodlanamadi: {path}")
    path.write_bytes(buffer.tobytes())


def read_image(path: Path) -> np.ndarray:
    """Unicode-safe imread."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise IOError(f"Kare okunamadi: {path}")
    return image


@dataclass(frozen=True)
class VideoInfo:
    frames_dir: Path
    fps: float
    width: int
    height: int
    frame_count: int


def extract_frames(video_path: str | Path, frames_dir: str | Path) -> VideoInfo:
    """Extract video frames as JPEGs (resized to MAX_PROCESS_WIDTH if wider)."""
    frames_dir = Path(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Video acilamadi: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    scale = min(1.0, MAX_PROCESS_WIDTH / src_w) if src_w > 0 else 1.0
    out_w = _even(int(src_w * scale))
    out_h = _even(int(src_h * scale))

    count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if scale < 1.0:
            frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        write_image(frames_dir / f"{count:05d}.jpg", frame, quality=95)
        count += 1
    cap.release()

    if count == 0:
        raise ValueError(f"Videodan hic kare okunamadi: {video_path}")
    return VideoInfo(frames_dir=frames_dir, fps=fps, width=out_w, height=out_h, frame_count=count)


def read_first_frame_rgb(info: VideoInfo):
    """Return the first extracted frame as an RGB numpy array."""
    frame = read_image(info.frames_dir / "00000.jpg")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def ffmpeg_exe() -> str:
    """Return a usable ffmpeg binary (bundled with imageio-ffmpeg)."""
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def remux_with_audio(processed_video: Path, original_video: Path, output: Path) -> Path:
    """Copy the original audio track onto the processed video if one exists."""
    cmd = [
        ffmpeg_exe(), "-y",
        "-i", str(processed_video),
        "-i", str(original_video),
        "-map", "0:v:0", "-map", "1:a:0?",
        "-c:v", "copy", "-c:a", "aac", "-shortest",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        # No audio track or remux failure: fall back to the processed video as-is.
        return processed_video
    return output


def _even(value: int) -> int:
    return value if value % 2 == 0 else value - 1
