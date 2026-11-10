# LatentMovement 🎬

**AI Infinite Motion & Single Image Studio**

LatentMovement directs fluid camera movements (pan right, pan left, pan up, pan down, zoom in, zoom out) through the latent space of Stable Diffusion, producing infinite motion sequences and standalone single image generations.

Crafted by **FlaskArchitect** & **Esperanza**.

---

## ✨ Features

- **Latent Camera Motion**:
  - Pan Right & Pan Left (infinite panorama shifting)
  - Pan Up & Pan Down
  - Zoom Out & Zoom In (recursive spatial expansion / contraction)
- **Single Image Generation**:
  - 1-click text-to-image or img2img creation.
  - Generates images into `novel_images_singles/` with companion JSON metadata and embedded PNG text chunks.
  - Generated single images can be immediately set as the **Base Frame** to kick off camera movement!
- **Dynamic Checkpoint & LoRA Detection**:
  - Auto-detects checkpoints in `./models/checkpoints/` (or symlinked external drive).
  - Empty-state UI with 1-click folder selection and live rescan.
  - Supports LoRA models with dynamic strength scaling.
  - Supports external VAEs (e.g. `vae-ft-mse-840000-ema-pruned.safetensors`).
- **Prompt Traveling & Keyframing**:
  - Linear prompt traveling across frame intervals (e.g. `0: alien planet \n 50: crystal jungle`).
  - Weighted prompt parsing `(word:1.2)` support.
- **Automatic Video Compilation**:
  - Direct FFmpeg rendering of frame sequences into `output.mp4` with Lanczos scaling and unsharp filtering.
- **Standalone Tauri v2 AppImage**:
  - Ultra-lightweight native Linux desktop wrapper with auto-server supervision.

---

## 🚀 Quick Start

### Running the AppImage
```bash
chmod +x LatentMovement-x86_64.AppImage
./LatentMovement-x86_64.AppImage
```

### Running from Source
```bash
# Start backend server
python latent_movement_server.py --port 5002

# Or launch with Tauri desktop wrapper
npm run tauri dev
```

---

## 📁 Directory Structure

```
├── models/                     # Symlink or directory for AI weights
│   ├── checkpoints/            # Stable Diffusion .safetensors models
│   ├── loras/                  # LoRA adapter weights
│   └── vae/                    # External VAE models
├── novel_images_singles/       # Saved single image generations
├── novel_images_{UID}/         # Generated motion sequences & output.mp4
├── templates/                  # Web studio interface
├── static/                     # Icons and static web assets
├── src-tauri/                  # Tauri v2 native desktop source
└── latent_movement_server.py   # Core Python inference server
```

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for details.
