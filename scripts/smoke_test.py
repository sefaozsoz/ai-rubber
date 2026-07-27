"""GPU smoke test: SAM2 loads on CUDA and the Gradio app imports."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sam2.build_sam import build_sam2_video_predictor

from src.sam2_wrapper import SAM2_CHECKPOINT, SAM2_CONFIG

predictor = build_sam2_video_predictor(SAM2_CONFIG, str(SAM2_CHECKPOINT), device="cuda")
print("SAM2 loaded on", next(predictor.parameters()).device)

import app  # noqa: E402,F401

print("app import ok")
