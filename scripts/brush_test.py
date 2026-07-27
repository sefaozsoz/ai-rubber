"""Brush prompt test: rough painted blob -> SAM2 refines onto the object."""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.e2e_test import make_test_video
from src.sam2_wrapper import Sam2Session
from src.video_utils import extract_frames

work = Path(tempfile.mkdtemp(prefix="airubber_brush_"))
video = work / "test.mp4"
make_test_video(video)
info = extract_frames(video, work / "frames")

sam2 = Sam2Session(info.frames_dir)
brush = np.zeros((info.height, info.width), dtype=bool)
brush[100:140, 50:100] = True  # kirmizi karenin ustune kaba bir boyama
refined = sam2.add_mask(brush)
print(f"brush px: {int(brush.sum())} -> refined px: {int(refined.sum())}")
assert refined.sum() > 500

saved = sam2.propagate_and_save_masks(work / "masks")
sam2.close()
assert saved == info.frame_count
print("BRUSH OK")
