"""AI Rubber — videodan obje silme araci.

Akis: video yukle -> ilk karede objeye tikla (SAM2 maskeler) ->
"Objeyi Sil" -> SAM2 maskeyi tum videoya yayar -> ProPainter objeyi siler.
"""

import os
import shutil
import sys
import uuid
from pathlib import Path

import cv2
import gradio as gr
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import run_propainter
from src.sam2_wrapper import Sam2Session, overlay_mask
from src.video_utils import extract_frames, read_first_frame_rgb, remux_with_audio

# ASCII-safe konum: OneDrive/"Masaüstü" altindaki yollar OpenCV ve ucuncu parti
# kodda sorun cikariyor, oturum dosyalari AppData'da tutulur.
SESSIONS_DIR = Path(os.environ.get("LOCALAPPDATA", PROJECT_ROOT)) / "ai-rubber" / "sessions"


def on_video_upload(video_path: str | None, session: dict | None):
    _cleanup_session(session)
    if not video_path:
        return None, None, None, gr.update(value="")

    session_dir = SESSIONS_DIR / uuid.uuid4().hex[:8]
    frames_dir = session_dir / "frames"
    info = extract_frames(video_path, frames_dir)
    first_frame = read_first_frame_rgb(info)

    sam2 = Sam2Session(frames_dir)
    new_session = {
        "dir": session_dir,
        "info": info,
        "sam2": sam2,
        "source": video_path,
        "first_frame": first_frame,
    }
    editor_value = {"background": first_frame, "layers": [], "composite": None}
    msg = f"{info.frame_count} kare @ {info.fps:.0f} fps ({info.width}x{info.height}). Objeye tikla veya fircayla boya."
    return new_session, first_frame, editor_value, gr.update(value=msg)


def on_image_click(session: dict | None, point_mode: str, evt: gr.SelectData):
    if not session or session.get("sam2") is None:
        raise gr.Error("Once bir video yukle.")
    x, y = evt.index
    positive = point_mode.startswith("Ekle")
    mask = session["sam2"].add_point(x, y, positive)
    return overlay_mask(session["first_frame"], mask)


def on_brush_apply(session: dict | None, editor_value: dict | None):
    if not session or session.get("sam2") is None:
        raise gr.Error("Once bir video yukle.")
    brush_mask = _brush_mask_from_editor(editor_value, session["first_frame"].shape[:2])
    if brush_mask is None or not brush_mask.any():
        raise gr.Error("Once fircayla objenin uzerini boya.")
    mask = session["sam2"].add_mask(brush_mask)
    return overlay_mask(session["first_frame"], mask)


def _brush_mask_from_editor(editor_value: dict | None, shape: tuple[int, int]) -> np.ndarray | None:
    """Collect painted pixels (alpha > 0) from all editor layers."""
    if not editor_value or not editor_value.get("layers"):
        return None
    mask = np.zeros(shape, dtype=bool)
    for layer in editor_value["layers"]:
        if layer is None:
            continue
        alpha = layer[..., 3] if layer.ndim == 3 and layer.shape[2] == 4 else layer.max(axis=2)
        if alpha.shape != shape:
            alpha = cv2.resize(alpha, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        mask |= alpha > 0
    return mask


def on_clear_points(session: dict | None):
    if not session or session.get("sam2") is None:
        return None, None
    session["sam2"].clear_points()
    editor_value = {"background": session["first_frame"], "layers": [], "composite": None}
    return session["first_frame"], editor_value


def on_remove_object(session: dict | None, progress=gr.Progress()):
    if not session or session.get("sam2") is None:
        raise gr.Error("Once bir video yukleyip objeye tiklamalisin.")

    session_dir: Path = session["dir"]
    masks_dir = session_dir / "masks"
    info = session["info"]

    progress(0.1, desc="Maske tum videoya yayiliyor (SAM2)...")
    session["sam2"].propagate_and_save_masks(masks_dir)
    session["sam2"].close()  # ProPainter icin VRAM bosalt
    session["sam2"] = None

    progress(0.5, desc="Obje siliniyor (ProPainter)... ilk calistirmada agirliklar iner")
    inpainted = run_propainter(
        frames_dir=info.frames_dir,
        masks_dir=masks_dir,
        output_dir=session_dir / "result",
        fps=info.fps,
        log_path=session_dir / "propainter.log",
    )

    progress(0.9, desc="Ses ekleniyor...")
    final = remux_with_audio(inpainted, Path(session["source"]), session_dir / "final.mp4")
    return str(final), "Bitti! Yeni bir duzenleme icin videoyu tekrar yukle."


def _cleanup_session(session: dict | None) -> None:
    if not session:
        return
    if session.get("sam2") is not None:
        session["sam2"].close()
    old_dir = session.get("dir")
    if old_dir and Path(old_dir).exists():
        shutil.rmtree(old_dir, ignore_errors=True)


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="AI Rubber — Videodan Obje Sil") as demo:
        gr.Markdown("# 🧽 AI Rubber\nVideo yukle → objeye tikla → **Objeyi Sil**. Tamamen lokal ve ucretsiz (SAM2 + ProPainter).")
        session = gr.State(None)

        with gr.Row():
            with gr.Column():
                video_in = gr.Video(label="Video yukle", sources=["upload"])
                status = gr.Textbox(label="Durum", interactive=False)
                point_mode = gr.Radio(
                    ["Ekle (obje)", "Cikar (arka plan)"],
                    value="Ekle (obje)",
                    label="Tiklama modu (Tikla sekmesi icin)",
                )
                with gr.Row():
                    clear_btn = gr.Button("Secimi Temizle")
                    remove_btn = gr.Button("🧽 Objeyi Sil", variant="primary")
            with gr.Column():
                with gr.Tab("🖱️ Tikla"):
                    frame_view = gr.Image(
                        label="Objeye tikla — kirmizi maske cikar", interactive=False
                    )
                with gr.Tab("🖌️ Firca"):
                    brush_editor = gr.ImageEditor(
                        label="Objenin uzerini boya, sonra 'Fircayi Uygula'",
                        type="numpy",
                        sources=(),
                        transforms=(),
                        brush=gr.Brush(colors=["#ff0000"], default_size=25),
                        layers=False,
                    )
                    brush_btn = gr.Button("🖌️ Fircayi Uygula", variant="secondary")
                    brush_preview = gr.Image(label="Firca secim onizlemesi", interactive=False)
                video_out = gr.Video(label="Sonuc")

        video_in.change(
            on_video_upload, [video_in, session], [session, frame_view, brush_editor, status]
        )
        frame_view.select(on_image_click, [session, point_mode], [frame_view])
        brush_btn.click(on_brush_apply, [session, brush_editor], [brush_preview])
        clear_btn.click(on_clear_points, [session], [frame_view, brush_editor])
        remove_btn.click(on_remove_object, [session], [video_out, status])
    return demo


if __name__ == "__main__":
    build_ui().launch(inbrowser=True)
