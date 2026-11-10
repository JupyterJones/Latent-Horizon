# 📋 Master Project Specification & System Reference Template

> **Instructions**: Use this template whenever you want to start a new script, video pipeline, or web app. Fill in the bracketed `[ ... ]` blanks in Part 1. Part 2 contains all verified system paths, hardware limits, model lists, and avatar tools for instant AI reference.

---

# PART 1: FILL-IN-THE-BLANK PROJECT SPECIFICATION

## 1. Project Concept & Goal
* **Project Name**: `[e.g., Cyberpunk Detective SLERP Video Generator]`
* **Primary Objective**: `[e.g., Generate a 3-act continuous morph video with synced Kokoro-TTS narration]`
* **Delivery Format**: `[Standalone Python CLI Script / Flask Web App / Batch Overnight Runner / Live Streamer]`
* **Target Aspect Ratio & Resolution**: `[512x768 (Portrait 9:16) / 512x768 (Landscape 16:9) / 512x512 (Square)]`

---

## 2. Diffusion, Model & Embedding Settings

* **Checkpoint, vae & Lora Path**: `[/media/jack/9930-14E1/models/]`
* **Checkpoint Model**: `[comiccraftLCM_beta4.safetensors / dreamshaper_8LCM.safetensors / cyberrealisticLCM_cyberrealistic42.safetensors / other]`
* **Custom VAE**: `[vae-ft-mse-840000-ema-pruned.safetensors / None]`
* **LoRA Adapter**: `[more_details.safetensors (Weight: 0.70) / None / other]`
* **SLERP Step Size ($\Delta t$)**: `[0.04 (26 frames) / 0.02 (51 frames) / 0.01 (101 frames)]`
* **LCM Sampling Steps**: `[8 steps (Standard LCM) / 4 steps (Fast preview)]`
* **CFG Scale (Guidance)**: `[2.0 (Standard LCM) / 1.5 to 2.2]`
* **Fixed Seed**: `[437437 / Random]`
* **Prompt Chunking**: `[Yes - Enable 75-token sub-block chunking for verbose prompts]`

---

## 3. Prompts & Narration Script
* **Prompt A (Start State)**:
  ```text
  [Enter initial scene description, subject anchors, lighting, and style keywords]
  ```
* **Prompt B (Destination State)**:
  ```text
  [Enter final transformed scene description, preserving identical subject anchors]
  ```
* **Voice Narration (Verbose Script)**:
  ```text
  [Enter the complete narrative script to be spoken by Kokoro-TTS]
  ```
* **Voice ID & Pacing**:
  * **Voice Name**: `[af_sarah (Female) / am_adam (Male) / af_nicole / am_michael / af_bella / am_fenrir]`
  * **Speed Factor**: `[1.0 (Normal) / 0.95 (Cinematic slow)]`

---

## 4. End-to-End Execution Pipeline
*Specify the execution flow:*
1. **Step 1 - Prompt Encoding**: `[Encode Prompt A and B into chunked (1, 77, 768) embeddings]`
2. **Step 2 - SLERP Interpolation**: `[Render N frames along the trajectory using token-wise SLERP]`
3. **Step 3 - High-Res Upscale**: `[Upscale frames 2x using RealESRGAN_x4 / PIL Lanczos]`
4. **Step 4 - Audio Synthesis**: `[Generate speech narration WAV using Kokoro-TTS]`
5. **Step 5 - Video Stitching**: `[Compile MP4 with FFmpeg, syncing frame durations to narration audio]`
6. **Step 6 - Memory Release**: `[Purge pipeline memory and invoke gc.collect()]`

---

## 5. UI / Frontend Preferences (If Web / GUI based)
* **Interface Layout**: `[Sleek Dark Mode, Glassmorphic cards, Modern typography / Headless]`
* **Controls & Features**:
  * `[Live image preview card for Prompt A & B]`
  * `[Interactive JSON preset manager (Load / Save / Edit)]`
  * `[Real-time frame generation progress bar]`
  * `[Integrated audio player and video viewer]`
  * `[NEVER use white input backgrounds with light letters, it is impossible for humans to read]`

---

## 6. Avatar / Lip-Sync / Animation (Optional)
* **Avatar Engine**: `[None / avatar_maker / LivePortrait / Wav2Lip-master]`
* **Source Face / Image**: `[e.g., /home/jack/Desktop/avatar_maker/use.jpg]`
* **Target Video / Audio**: `[e.g., Generated Kokoro narration WAV file]`

---

# PART 2: SYSTEM HARDWARE & WORKSPACE REFERENCE GUIDE

## 💻 Hardware & System Constraints
* **Operating System**: Linux x86_64
* **CPU**: 8 Cores (CPU-Only inference)
* **RAM**: 13.59 GB Total (~11.9 GB available)
* **Swap**: 12.00 GB Total (~7.2 GB free)
* **Throttling Guard**: `os.nice(15)` + `torch.set_num_threads(2)` (Keeps Linux desktop UI 100% smooth)
* **Instant Exit**: `SIGINT` / `SIGTERM` handler calling `os._exit(0)`

---

## 📁 Model & Asset Directories

### 1. Stable Diffusion Checkpoints (`/home/jack/Desktop/Comfy-UI/models/checkpoints/`)
* `comiccraftLCM_beta4.safetensors` *(Graphic novel & comic illustration)*
* `dreamshaper_8LCM.safetensors` *(General purpose art & realism)*
* `cyberrealisticLCM_cyberrealistic42.safetensors` *(High-fidelity photorealism)*
* `epicphotogasmLCM_ultimatefidelity.safetensors` *(Photographic portraits)*
* `photonLCM_v10.safetensors` *(Realistic cinematic sci-fi)*
* `delusionsLCM_v20.safetensors` *(Surreal & fantasy art)*
* `frtMixmodelLCM_realisticV10.safetensors`
* `possibleLoveliesLCM_v10.safetensors`
* `silversRealmixLCM_v10.safetensors`
* `xenoxtcLCMEditionArtToon_v10.safetensors`

### 2. VAE Models (`/home/jack/Desktop/Comfy-UI/models/vae/`)
* `vae-ft-mse-840000-ema-pruned.safetensors` *(Standard clean SD 1.5 VAE)*
* `vae-ft-mse-840000-ema-pruned.ckpt`

### 3. LoRA Adapters (`/home/jack/Desktop/Comfy-UI/models/loras/`)
* `more_details.safetensors` *(Weight: 0.6–0.8, linked locally in `/home/jack/Desktop/overnight/`)*
* `Detail_Slider_V2_By_Stable .safetensors`
* `PAseer-SD15-LCM Quick.safetensors`
* `SW_PoeticV5preview3_sd.safetensors`
* `Ghibli_v6.safetensors`
* `Low_Poly_Art.safetensors`
* `DarkPunk_Style.safetensors`
* `Terror Tales.safetensors`
* `easynegative.safetensors`

### 4. Neural Upscalers (`/home/jack/Desktop/Comfy-UI/models/upscale_models/`)
* `RealESRGAN_x4.pth` *(Loaded via Spandrel on CPU with Lanczos fallback)*

---

## 🎙️ Kokoro-TTS Configuration (`/home/jack/Desktop/Kokoro-TTS-Pause/`)
* **Binary Executable**: `/home/jack/Desktop/Kokoro-TTS-Pause/venv/bin/kokoro-tts`
* **ONNX Model**: `/home/jack/Desktop/Kokoro-TTS-Pause/kokoro-v1.0.onnx`
* **Voices File**: `/home/jack/Desktop/Kokoro-TTS-Pause/voices-v1.0.bin`
* **Available Voices**:
  * **Female**: `af_sarah`, `af_nicole`, `af_sky`, `af_bella`, `bf_emma`
  * **Male**: `am_adam`, `am_michael`, `am_fenrir`, `bm_george`

---

## 🤖 Local Ollama Models (`http://localhost:11434`)
* **Custom AI Directors**:
  * `deepseek-slerp:1` *(SLERP trajectory & prompt director)*
  * `comic-director:1` *(Graphic novel & comic screenplay director)*
  * `noir-detective:1` *(1940s noir narrative director)*
  * `scifi-writer:1` *(Futuristic sci-fi worldbuilder)*
  * `fantasy-scribe:1` *(Epic fantasy narration)*
  * `horror-vault:1` *(Horror storytelling)*
  * `starcor-news:1` *(News broadcast writer)*
* **General & Coding Models**:
  * `granite4.1:3b`, `llama3.2:3b`, `qwen2.5-coder:7b`, `deepseek-r1:1.5b`
  * Vision Models: `moondream:latest`, `LlaVa:latest`
  * Embeddings: `nomic-embed-text:latest`

---

## 🎭 Avatar, Talking Head & Video Studios
* **Avatar Maker**: `/home/jack/Desktop/avatar_maker/` (`mariana_avatar_gen.py`, `Stable-Face-Maker.py`, `avatar.py`)
* **LivePortrait**: `/home/jack/Desktop/LivePortrait/` (`app.py`, `app_animals.py`, `speed.py` - high-res reenactment)
* **Wav2Lip**: `/home/jack/Desktop/Wav2Lip-master/` (`audio.py`, `blink.py`, `hq_wav2lip_train.py` - lip sync)
* **ONE_STEP SLERP Studio**: `/home/jack/Desktop/ONE_STEP-Diffusion-SLERP/`
* **Comic Video Maker**: `/home/jack/Desktop/Comic_Book_Video_Maker/`

---

## 📐 Mathematical SLERP Formula Reference
```python
# Token-wise numerically stable SLERP (Preserves token magnitude & avoids zero-collapse)
@torch.no_grad()
def slerp(t: float, v0: torch.Tensor, v1: torch.Tensor, dot_threshold: float = 0.9995) -> torch.Tensor:
    t_val = float(max(0.0, min(1.0, t)))
    if t_val == 0.0: return v0.clone()
    if t_val == 1.0: return v1.clone()

    norm_v0, norm_v1 = torch.norm(v0, dim=-1, keepdim=True), torch.norm(v1, dim=-1, keepdim=True)
    target_norm = (1.0 - t_val) * norm_v0 + t_val * norm_v1

    v0_u, v1_u = v0 / (norm_v0 + 1e-8), v1 / (norm_v1 + 1e-8)
    dot = torch.clamp(torch.sum(v0_u * v1_u, dim=-1, keepdim=True), -1.0, 1.0)
    close_mask = (dot > dot_threshold)

    theta_0 = torch.acos(dot)
    safe_sin = torch.where(close_mask, torch.ones_like(theta_0), torch.sin(theta_0))
    s0 = torch.sin(theta_0 * (1.0 - t_val)) / safe_sin
    s1 = torch.sin(theta_0 * t_val) / safe_sin

    res_unit = torch.where(close_mask, (1.0 - t_val) * v0_u + t_val * v1_u, s0 * v0_u + s1 * v1_u)
    return (res_unit / (torch.norm(res_unit, dim=-1, keepdim=True) + 1e-8)) * target_norm
```
