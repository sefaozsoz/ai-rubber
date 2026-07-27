"""End-to-end pipeline: frames + masks -> ProPainter inpainting -> final video."""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROPAINTER_DIR = PROJECT_ROOT / "third_party" / "ProPainter"

# 6 GB VRAM defaults: fp16 + short sub-videos keep memory in check.
SUBVIDEO_LENGTH = 50
MASK_DILATION = 8


def run_propainter(
    frames_dir: Path,
    masks_dir: Path,
    output_dir: Path,
    fps: float,
    log_path: Path | None = None,
) -> Path:
    """Run ProPainter inference over extracted frames and masks.

    Returns the path of the inpainted video. ProPainter downloads its own
    weights (~500 MB) into third_party/ProPainter/weights on first run.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "inference_propainter.py",
        "--video", str(frames_dir),
        "--mask", str(masks_dir),
        "--output", str(output_dir),
        "--fp16",
        "--subvideo_length", str(SUBVIDEO_LENGTH),
        "--mask_dilation", str(MASK_DILATION),
        "--save_fps", str(max(1, round(fps))),
    ]
    result = subprocess.run(
        cmd, cwd=str(PROPAINTER_DIR), capture_output=True, text=True, encoding="utf-8"
    )
    if log_path is not None:
        log_path.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        tail = "\n".join((result.stderr or result.stdout).splitlines()[-15:])
        raise RuntimeError(f"ProPainter hata verdi:\n{tail}")

    video_name = frames_dir.name
    inpainted = output_dir / video_name / "inpaint_out.mp4"
    if not inpainted.exists():
        candidates = list(output_dir.rglob("inpaint_out.mp4"))
        if not candidates:
            raise RuntimeError(f"ProPainter cikti videosu bulunamadi: {output_dir}")
        inpainted = candidates[0]
    return inpainted
