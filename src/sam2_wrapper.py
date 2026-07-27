"""SAM 2 wrapper: interactive point-based selection + full-video mask propagation."""

import os
from pathlib import Path

import numpy as np
import torch

from src.video_utils import write_image

# AIRUBBER_SAM2_MODEL: tiny | small | base_plus | large (VRAM'e gore sec)
_SAM2_VARIANTS = {
    "tiny": ("sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
    "small": ("sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
    "base_plus": ("sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "large": ("sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
}
_variant = os.environ.get("AIRUBBER_SAM2_MODEL", "small")
_ckpt_name, SAM2_CONFIG = _SAM2_VARIANTS[_variant]
SAM2_CHECKPOINT = Path(__file__).resolve().parent.parent / "weights" / _ckpt_name
OBJ_ID = 1


class Sam2Session:
    """Holds SAM2 video predictor state for one uploaded video."""

    def __init__(self, frames_dir: Path):
        from sam2.build_sam import build_sam2_video_predictor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # T4 gibi eski GPU'lar bfloat16 desteklemez -> float16'ya dus
        self.autocast_dtype = (
            torch.bfloat16
            if self.device == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        self.predictor = build_sam2_video_predictor(
            SAM2_CONFIG, str(SAM2_CHECKPOINT), device=self.device
        )
        self.state = self.predictor.init_state(
            video_path=str(frames_dir),
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )
        self.points: list[tuple[int, int]] = []
        self.labels: list[int] = []
        self.has_prompt = False

    def add_point(self, x: int, y: int, positive: bool) -> np.ndarray:
        """Add a click on frame 0 and return the current mask (H, W) bool."""
        self.points.append((x, y))
        self.labels.append(1 if positive else 0)

        with torch.inference_mode(), torch.autocast(self.device, dtype=self.autocast_dtype):
            _, _, mask_logits = self.predictor.add_new_points_or_box(
                inference_state=self.state,
                frame_idx=0,
                obj_id=OBJ_ID,
                points=np.array(self.points, dtype=np.float32),
                labels=np.array(self.labels, dtype=np.int32),
            )
        self.has_prompt = True
        return (mask_logits[0, 0] > 0.0).cpu().numpy()

    def add_mask(self, brush_mask: np.ndarray) -> np.ndarray:
        """Use a painted (brush) mask as the selection prompt on frame 0.

        Replaces any previous click points; SAM2 snaps the rough brush area
        onto the underlying object and returns the refined mask.
        """
        self.clear_points()
        with torch.inference_mode(), torch.autocast(self.device, dtype=self.autocast_dtype):
            _, _, mask_logits = self.predictor.add_new_mask(
                inference_state=self.state,
                frame_idx=0,
                obj_id=OBJ_ID,
                mask=brush_mask.astype(bool),
            )
        self.has_prompt = True
        return (mask_logits[0, 0] > 0.0).cpu().numpy()

    def clear_points(self) -> None:
        self.points.clear()
        self.labels.clear()
        self.has_prompt = False
        self.predictor.reset_state(self.state)

    def propagate_and_save_masks(self, masks_dir: Path) -> int:
        """Propagate the selection through the video, writing one PNG mask per frame."""
        if not self.has_prompt:
            raise ValueError("Once objeyi tiklayarak veya fircayla secmelisin.")

        masks_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        with torch.inference_mode(), torch.autocast(self.device, dtype=self.autocast_dtype):
            for frame_idx, _, mask_logits in self.predictor.propagate_in_video(self.state):
                mask = (mask_logits[0, 0] > 0.0).cpu().numpy().astype(np.uint8) * 255
                write_image(masks_dir / f"{frame_idx:05d}.png", mask)
                count += 1
        return count

    def close(self) -> None:
        self.predictor.reset_state(self.state)
        del self.predictor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def overlay_mask(frame_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Blend a red highlight of the mask onto the frame for UI preview."""
    overlay = frame_rgb.copy()
    overlay[mask] = (overlay[mask] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(np.uint8)
    return overlay
