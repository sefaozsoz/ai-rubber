# 🧽 AI Rubber — Videodan Obje Silme

Videodan istenmeyen objeleri silen, **tamamen lokal ve ücretsiz** çalışan araç.
Video yükle → ilk karede objeye tıkla → "Objeyi Sil" → obje tüm videodan kaybolur.

## Nasıl çalışır?

| Aşama | Model | Lisans |
|---|---|---|
| Obje seçimi + maske takibi | [SAM 2.1](https://github.com/facebookresearch/sam2) (Meta) | Apache-2.0 |
| Video inpainting (silme) | [ProPainter](https://github.com/sczhou/ProPainter) (ICCV 2023) | S-Lab (ticari olmayan) |

1. **SAM 2.1** — ilk karede tıkladığın noktalardan objenin maskesini çıkarır ve
   maskeyi videonun tüm karelerine yayar (video object segmentation + tracking).
2. **ProPainter** — maskelenen bölgeyi optical flow + transformer ile komşu
   karelerden doldurur; obje silinmiş gibi görünür.

## Gereksinimler

- Windows + NVIDIA GPU (6 GB VRAM yeterli — fp16 ve 960px yeniden boyutlandırma varsayılan)
- Python 3.10+
- ~8 GB disk (PyTorch CUDA + model ağırlıkları)

## Kurulum

```powershell
.\scripts\setup.ps1
```

Script sırasıyla: venv kurar → ProPainter & SAM2 klonlar → bağımlılıkları ve
CUDA'lı PyTorch'u kurar → SAM2 ağırlığını (~180 MB) indirir.
ProPainter kendi ağırlıklarını (~500 MB) ilk çalıştırmada otomatik indirir.

## Çalıştırma

```powershell
& "$env:USERPROFILE\venvs\ai-rubber\Scripts\python.exe" app.py
```

Tarayıcıda Gradio arayüzü açılır:

1. Video yükle (kısa klipler önerilir; kareler işlem için en fazla 960px genişliğe küçültülür).
2. **Tıklama modu** "Ekle (obje)" iken silmek istediğin objeye tıkla — kırmızı maske önizlemesi görünür.
   Maske taşarsa "Cikar (arka plan)" moduyla fazla bölgelere tıklayarak düzelt.
3. **🧽 Objeyi Sil** — SAM2 maskeyi yayar, ProPainter siler, ses orijinalden geri eklenir.
4. Sonucu indir.

## Notlar

- 6 GB VRAM için ayarlar: `src/pipeline.py` içinde `--fp16`, `subvideo_length=50`.
  Bellek hatası alırsan `MAX_PROCESS_WIDTH`'i (src/video_utils.py) 640'a düşür.
- ProPainter lisansı ticari kullanım için uygun değil; ticari proje gerekirse
  [DiffuEraser](https://github.com/lixiaowen-xw/DiffuEraser) (Apache-2.0) alternatifine geçilebilir.
