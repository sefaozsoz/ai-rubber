"""End-to-end test: synthetic video -> SAM2 propagation -> ProPainter removal."""

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import run_propainter
from src.sam2_wrapper import Sam2Session
from src.video_utils import extract_frames


def make_test_video(path: Path, frames: int = 24, size: tuple[int, int] = (320, 240)) -> None:
    """Moving red square over a noisy background."""
    rng = np.random.default_rng(0)
    background = rng.integers(60, 200, (size[1], size[0], 3), dtype=np.uint8)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 12, size)
    for i in range(frames):
        frame = background.copy()
        x = 40 + i * 6
        cv2.rectangle(frame, (x, 90), (x + 50, 150), (0, 0, 255), -1)
        writer.write(frame)
    writer.release()


def main() -> None:
    work = Path(tempfile.mkdtemp(prefix="airubber_e2e_"))
    video = work / "test.mp4"
    make_test_video(video)

    info = extract_frames(video, work / "frames")
    print(f"frames: {info.frame_count} @ {info.width}x{info.height}")

    sam2 = Sam2Session(info.frames_dir)
    mask = sam2.add_point(65, 120, positive=True)  # ilk karedeki kare objenin ortasi
    print(f"click mask px: {int(mask.sum())}")
    assert mask.sum() > 500, "SAM2 tiklanan objeyi maskeleyemedi"

    saved = sam2.propagate_and_save_masks(work / "masks")
    sam2.close()
    print(f"propagated masks: {saved}")
    assert saved == info.frame_count

    result = run_propainter(
        frames_dir=info.frames_dir,
        masks_dir=work / "masks",
        output_dir=work / "result",
        fps=info.fps,
        log_path=work / "propainter.log",
    )
    print(f"inpainted video: {result} ({result.stat().st_size // 1024} KB)")
    print("E2E OK")


if __name__ == "__main__":
    main()
