"""SAM 2 wrapper: interactive point-based selection + full-video mask propagation."""

from pathlib import Path

import numpy as np
import torch

from src.video_utils import write_image

SAM2_CHECKPOINT = Path(__file__).resolve().parent.parent / "weights" / "sam2.1_hiera_small.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
OBJ_ID = 1


class Sam2Session:
    """Holds SAM2 video predictor state for one uploaded video."""

    def __init__(self, frames_dir: Path):
        from sam2.build_sam import build_sam2_video_predictor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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

    def add_point(self, x: int, y: int, positive: bool) -> np.ndarray:
        """Add a click on frame 0 and return the current mask (H, W) bool."""
        self.points.append((x, y))
        self.labels.append(1 if positive else 0)

        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            _, _, mask_logits = self.predictor.add_new_points_or_box(
                inference_state=self.state,
                frame_idx=0,
                obj_id=OBJ_ID,
                points=np.array(self.points, dtype=np.float32),
                labels=np.array(self.labels, dtype=np.int32),
            )
        return (mask_logits[0, 0] > 0.0).cpu().numpy()

    def clear_points(self) -> None:
        self.points.clear()
        self.labels.clear()
        self.predictor.reset_state(self.state)

    def propagate_and_save_masks(self, masks_dir: Path) -> int:
        """Propagate the selection through the video, writing one PNG mask per frame."""
        if not self.points:
            raise ValueError("Once objeye en az bir kez tiklamalisin.")

        masks_dir.mkdir(parents=True, exist_ok=True)
        count = 0
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
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
