import os
import sys
import argparse

# Limit thread usage for OpenMP, BLAS, and PyTorch runtime (v2 update)
MAX_CPU_THREADS = int(os.environ.get("TORCH_NUM_THREADS") or os.environ.get("OMP_NUM_THREADS") or 4)
os.environ["OMP_NUM_THREADS"] = str(MAX_CPU_THREADS)
os.environ["OPENBLAS_NUM_THREADS"] = str(MAX_CPU_THREADS)
os.environ["MKL_NUM_THREADS"] = str(MAX_CPU_THREADS)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(MAX_CPU_THREADS)
os.environ["NUMEXPR_NUM_THREADS"] = str(MAX_CPU_THREADS)
os.environ["TORCH_NUM_THREADS"] = str(MAX_CPU_THREADS)

# Restrict Linux CPU affinity to 4 cores if available
if hasattr(os, "sched_setaffinity") and hasattr(os, "cpu_count"):
    try:
        available_cores = os.cpu_count() or 4
        target_cores = set(range(min(MAX_CPU_THREADS, available_cores)))
        os.sched_setaffinity(0, target_cores)
        print(f"[*] Process CPU affinity set to cores: {sorted(list(target_cores))}")
    except Exception as e:
        print(f"[*] Note: Could not set CPU affinity: {e}")

import json
import time
import uuid
import shutil
import threading
import traceback
import subprocess
import signal
import re
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory, send_file
from PIL import Image, ImageFilter
from PIL.PngImagePlugin import PngInfo
import torch

# Patch PyTorch Parameter instantiation to prevent incompatibility issues with HuggingFace PEFT/Transformers
try:
    _orig_param_new = torch.nn.Parameter.__new__
    def _patched_param_new(cls, *args, **kwargs):
        kwargs.pop('_is_hf_initialized', None)
        return _orig_param_new(cls, *args, **kwargs)
    torch.nn.Parameter.__new__ = staticmethod(_patched_param_new)
except Exception as e:
    print(f"[*] Warning: Could not apply PyTorch Parameter monkey patch: {e}")

# Limit PyTorch to leave CPU cores free to prevent system freezing during CPU inference
try:
    torch.set_num_threads(MAX_CPU_THREADS)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(min(MAX_CPU_THREADS, 4))
        except Exception:
            pass
    print(f"[*] Set PyTorch thread limit to {torch.get_num_threads()} threads (max limit: {MAX_CPU_THREADS}).")
except Exception as e:
    print(f"[*] Warning: Could not configure PyTorch thread limit: {e}")

LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS

# Determine root application directory
if getattr(sys, 'frozen', False):
    # PyInstaller runtime
    APP_DIR = os.path.dirname(sys.executable)
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))

# Current working directory (where user launched the app/AppImage)
EXEC_DIR = os.environ.get("LATENT_MOVEMENT_CWD") or os.getcwd()

def get_config_path():
    host_cfg = os.path.join(EXEC_DIR, "config.json")
    if os.path.exists(host_cfg):
        return host_cfg
    app_cfg = os.path.join(APP_DIR, "config.json")
    if os.path.exists(app_cfg):
        return app_cfg
    return host_cfg

def get_draft_path():
    host_draft = os.path.join(EXEC_DIR, "draft.json")
    if os.path.exists(host_draft):
        return host_draft
    app_draft = os.path.join(APP_DIR, "draft.json")
    if os.path.exists(app_draft):
        return app_draft
    return host_draft

CONFIG_PATH = get_config_path()
DRAFT_PATH = get_draft_path()

def load_app_config():
    cfg = {
        "models_dir": "",
        "output_dir": ""
    }
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                saved = json.load(f)
                cfg.update(saved)
        except Exception as e:
            print(f"[!] Warning reading config.json: {e}")
    return cfg

def save_app_config(cfg):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=4)
        return True
    except Exception as e:
        print(f"[!] Error saving config.json: {e}")
        return False

# Smart Resolution of Models Directory
def resolve_models_dir():
    cfg = load_app_config()
    candidate_paths = [
        os.environ.get("LATENT_MOVEMENT_MODELS"),
        cfg.get("models_dir"),
        os.path.join(EXEC_DIR, "models"),
        os.path.join(APP_DIR, "models"),
        os.path.expanduser("~/models"),
        os.path.expanduser("~/ComfyUI/models"),
        os.path.expanduser("~/stable-diffusion-webui/models")
    ]
    for p in candidate_paths:
        if p and os.path.exists(p):
            checkpoints_sub = os.path.join(p, "checkpoints")
            if os.path.exists(checkpoints_sub) or os.path.exists(p):
                return os.path.abspath(p)
                
    # Fallback default: ./models in current working directory
    default_p = os.path.abspath(os.path.join(EXEC_DIR, "models"))
    return default_p

MODELS_BASE = resolve_models_dir()
CHECKPOINTS_DIR = os.path.join(MODELS_BASE, "checkpoints")
LORAS_DIR = os.path.join(MODELS_BASE, "loras")
VAE_DIR = os.path.join(MODELS_BASE, "vae")

# Ensure models subdirectories exist
for d in [CHECKPOINTS_DIR, LORAS_DIR, VAE_DIR]:
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass

# Resolve Output & Upload Directories
def resolve_output_dir():
    cfg = load_app_config()
    out = os.environ.get("LATENT_MOVEMENT_OUTPUT") or cfg.get("output_dir")
    if out and os.path.exists(out):
        return os.path.abspath(out)
    return EXEC_DIR

OUTPUT_BASE = resolve_output_dir()
SINGLES_DIR = os.path.join(OUTPUT_BASE, "novel_images_singles")
UPLOAD_FOLDER = os.path.join(OUTPUT_BASE, "uploads")

os.makedirs(SINGLES_DIR, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def resolve_template_folder():
    host_tmpl = os.path.join(EXEC_DIR, "templates")
    if os.path.isdir(host_tmpl) and os.path.exists(os.path.join(host_tmpl, "index0.html")):
        return host_tmpl
    return os.path.join(APP_DIR, "templates")

app = Flask(__name__, 
            template_folder=resolve_template_folder(),
            static_folder=os.path.join(APP_DIR, "static"))

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global runtime state
active_runs = {}
pipeline_cache = {
    'model_path': None,
    'vae_path': None,
    'device': None,
    'pipe': None,
    'txt2img_pipe': None
}
pipeline_lock = threading.Lock()

def detect_default_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def scan_checkpoints():
    models = []
    if os.path.exists(CHECKPOINTS_DIR):
        for f in os.listdir(CHECKPOINTS_DIR):
            if f.endswith(('.safetensors', '.ckpt')):
                models.append(f)
    models.sort()
    return models

def scan_loras():
    loras = []
    if os.path.exists(LORAS_DIR):
        for f in os.listdir(LORAS_DIR):
            if f.endswith(('.safetensors', '.ckpt', '.pt')):
                loras.append(f)
    loras.sort()
    return loras

def scan_vaes():
    vaes = []
    if os.path.exists(VAE_DIR):
        for f in os.listdir(VAE_DIR):
            if f.endswith(('.safetensors', '.ckpt', '.pt')):
                vaes.append(f)
    vaes.sort()
    return vaes

def get_default_checkpoint():
    models = scan_checkpoints()
    if not models:
        return ""
    # Prefer dreamshaper_8LCM if available
    for m in models:
        if "dreamshaper_8LCM" in m:
            return os.path.join(CHECKPOINTS_DIR, m)
    return os.path.join(CHECKPOINTS_DIR, models[0])

def create_feathered_mask(canvas_width, canvas_height, direction, start_blend, end_blend, blur_radius=8):
    if direction in ("right", "left"):
        gradient = Image.new("L", (canvas_width, 1))
        if start_blend >= end_blend:
            for x in range(canvas_width):
                val = 255 if x < start_blend else 0
                gradient.putpixel((x, 0), val)
        else:
            for x in range(canvas_width):
                if x < start_blend:
                    val = 255 if direction == "right" else 0
                elif x > end_blend:
                    val = 0 if direction == "right" else 255
                else:
                    t = (x - start_blend) / (end_blend - start_blend)
                    val = int(255 * (1.0 - t if direction == "right" else t))
                gradient.putpixel((x, 0), val)
        mask = gradient.resize((canvas_width, canvas_height))
        
    elif direction in ("down", "up"):
        gradient = Image.new("L", (1, canvas_height))
        if start_blend >= end_blend:
            for y in range(canvas_height):
                val = 255 if y < start_blend else 0
                gradient.putpixel((0, y), val)
        else:
            for y in range(canvas_height):
                if y < start_blend:
                    val = 255 if direction == "down" else 0
                elif y > end_blend:
                    val = 0 if direction == "down" else 255
                else:
                    t = (y - start_blend) / (end_blend - start_blend)
                    val = int(255 * (1.0 - t if direction == "down" else t))
                gradient.putpixel((0, y), val)
        mask = gradient.resize((canvas_width, canvas_height))
        
    elif direction in ("out", "in"):
        from PIL import ImageDraw
        mask = Image.new("L", (canvas_width, canvas_height), 0)
        draw = ImageDraw.Draw(mask)
        x_offset = start_blend
        y_offset = end_blend
        draw.rectangle(
            [x_offset, y_offset, canvas_width - x_offset - 1, canvas_height - y_offset - 1],
            fill=255
        )
        mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))
        
    else:
        gradient = Image.new("L", (canvas_width, 1))
        for x in range(canvas_width):
            val = 255 if x < start_blend else 0
            gradient.putpixel((x, 0), val)
        mask = gradient.resize((canvas_width, canvas_height))
        
    return mask

def correct_colors(target_img, reference_img):
    import numpy as np
    target = np.array(target_img).astype(np.float32)
    reference = np.array(reference_img).astype(np.float32)
    
    for i in range(3):
        ref_mean = np.mean(reference[:, :, i])
        ref_std = np.std(reference[:, :, i])
        tgt_mean = np.mean(target[:, :, i])
        tgt_std = np.std(target[:, :, i])
        
        if tgt_std > 1e-5:
            target[:, :, i] = (target[:, :, i] - tgt_mean) * (ref_std / tgt_std) + ref_mean
        else:
            target[:, :, i] = target[:, :, i] - tgt_mean + ref_mean
            
    return Image.fromarray(np.clip(target, 0, 255).astype(np.uint8))

def parse_prompt_weights(prompt_text):
    pattern = r"\(([^)]+)\)"
    parts = []
    last_idx = 0
    
    for match in re.finditer(pattern, prompt_text):
        start, end = match.span()
        if start > last_idx:
            parts.append((prompt_text[last_idx:start], 1.0))
            
        match_content = match.group(1)
        if ":" in match_content:
            sub_text, weight_str = match_content.rsplit(":", 1)
            try:
                weight = float(weight_str)
            except ValueError:
                sub_text = match_content
                weight = 1.1
        else:
            sub_text = match_content
            weight = 1.1
            
        parts.append((sub_text, weight))
        last_idx = end
        
    if last_idx < len(prompt_text):
        parts.append((prompt_text[last_idx:], 1.0))
        
    if not parts:
        parts.append((prompt_text, 1.0))
        
    return parts

def get_chunked_embeds(pipe, prompt_text, device, lora_scale_val=None, target_chunks=None):
    """
    Tokenizes and encodes prompt text into chunks of 77 tokens.
    Supports arbitrarily long prompts (> 77 tokens) without truncation.
    Applies (word:weight) weighted parsing across chunks.
    """
    if hasattr(pipe, "adjust_lora_scale_for_text_encoder") and lora_scale_val is not None:
        try:
            pipe.adjust_lora_scale_for_text_encoder(pipe.text_encoder, lora_scale_val)
        except Exception:
            pass

    bos_token_id = pipe.tokenizer.bos_token_id or 49406
    eos_token_id = pipe.tokenizer.eos_token_id or 49407
    max_length = pipe.tokenizer.model_max_length or 77
    chunk_capacity = max_length - 2  # 75 usable content tokens per chunk

    parts = parse_prompt_weights(prompt_text or "")
    raw_token_ids = []
    raw_weights = []

    for text, weight in parts:
        ids = pipe.tokenizer(text, add_special_tokens=False).input_ids
        raw_token_ids.extend(ids)
        raw_weights.extend([weight] * len(ids))

    # Split into chunks of 75 content tokens
    id_chunks = []
    weight_chunks = []
    if len(raw_token_ids) == 0:
        id_chunks.append([])
        weight_chunks.append([])
    else:
        for i in range(0, len(raw_token_ids), chunk_capacity):
            id_chunks.append(raw_token_ids[i:i + chunk_capacity])
            weight_chunks.append(raw_weights[i:i + chunk_capacity])

    # Pad with empty chunks if a target_chunks count is requested
    if target_chunks is not None and len(id_chunks) < target_chunks:
        pad_needed = target_chunks - len(id_chunks)
        for _ in range(pad_needed):
            id_chunks.append([])
            weight_chunks.append([])

    # Encode each chunk
    all_embeds = []
    for chunk_ids, chunk_w in zip(id_chunks, weight_chunks):
        full_ids = [bos_token_id] + chunk_ids + [eos_token_id]
        full_w = [1.0] + chunk_w + [1.0]

        pad_len = max_length - len(full_ids)
        if pad_len > 0:
            full_ids.extend([eos_token_id] * pad_len)
            full_w.extend([1.0] * pad_len)
        elif pad_len < 0:
            full_ids = full_ids[:max_length]
            full_w = full_w[:max_length]

        input_tensor = torch.tensor([full_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = pipe.text_encoder(input_tensor)[0]
        w_tensor = torch.tensor([full_w], dtype=out.dtype, device=device).unsqueeze(-1)
        all_embeds.append(out * w_tensor)

    return torch.cat(all_embeds, dim=1)

def get_chunked_embed_pair(pipe, prompt_text, negative_text, device, lora_scale_val=None, min_chunks=1):
    """
    Encodes both prompt and negative prompt, ensuring both have identical sequence lengths
    by matching the number of 77-token chunks.
    """
    chunk_capacity = (pipe.tokenizer.model_max_length or 77) - 2

    # Calculate required chunks for positive prompt
    p_parts = parse_prompt_weights(prompt_text or "")
    p_len = sum(len(pipe.tokenizer(t, add_special_tokens=False).input_ids) for t, _ in p_parts)
    p_chunks = max(1, (p_len + chunk_capacity - 1) // chunk_capacity) if p_len > 0 else 1

    # Calculate required chunks for negative prompt
    n_parts = parse_prompt_weights(negative_text or "")
    n_len = sum(len(pipe.tokenizer(t, add_special_tokens=False).input_ids) for t, _ in n_parts)
    n_chunks = max(1, (n_len + chunk_capacity - 1) // chunk_capacity) if n_len > 0 else 1

    total_chunks = max(p_chunks, n_chunks, min_chunks)

    p_embeds = get_chunked_embeds(pipe, prompt_text, device, lora_scale_val, target_chunks=total_chunks)
    n_embeds = get_chunked_embeds(pipe, negative_text, device, lora_scale_val, target_chunks=total_chunks)

    return p_embeds, n_embeds

def get_weighted_embeds(pipe, prompt_text, device, lora_scale_val=None):
    """Backward compatibility wrapper using chunked encoding."""
    return get_chunked_embeds(pipe, prompt_text, device, lora_scale_val)

def compile_video(output_dir, fps=30, scale=1.0, sharpen=0.0):
    print("\n" + "="*60)
    print("   AUTOMATIC VIDEO COMPILATION // FFMPEG ENGINE")
    print("="*60)
    
    if not shutil.which("ffmpeg"):
        print("[!] WARNING: 'ffmpeg' executable was not found in system path.")
        return False
        
    output_video_path = os.path.join(output_dir, "output.mp4")
    
    if os.path.exists(output_video_path):
        counter = 1
        while True:
            backup_path = os.path.join(output_dir, f"output_{counter}.mp4")
            if not os.path.exists(backup_path):
                break
            counter += 1
        try:
            os.rename(output_video_path, backup_path)
            print(f"[*] Archived previous video to: {os.path.basename(backup_path)}")
        except Exception as e:
            print(f"[!] Warning: Failed to archive previous video: {e}")
    
    video_filters = []
    if scale != 1.0:
        video_filters.append(f"scale=trunc(iw*{scale}/2)*2:trunc(ih*{scale}/2)*2:flags=lanczos")
    if sharpen > 0.0:
        video_filters.append(f"unsharp=5:5:{sharpen}:5:5:0.0")
        
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", os.path.join(output_dir, "frame%04d.png")
    ]
    
    if video_filters:
        cmd.extend(["-vf", ",".join(video_filters)])
        
    cmd.extend([
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        output_video_path
    ])
    
    try:
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        print(f"[SUCCESS] Video compiled successfully: {output_video_path}")
        return True
    except Exception as e:
        print(f"[!] ERROR during FFmpeg video compilation: {e}")
        return False

def parse_prompt_travel(prompt_str):
    keyframes = []
    normalized = (prompt_str or "").strip().replace('\\n', '\n')
    lines = normalized.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(':', 1)
        if len(parts) == 2:
            try:
                frame_idx = int(parts[0].strip())
                prompt_text = parts[1].strip()
                keyframes.append({'frame': frame_idx, 'prompt': prompt_text})
                continue
            except ValueError:
                pass
    
    if not keyframes:
        keyframes.append({'frame': 0, 'prompt': prompt_str.strip()})
        
    keyframes.sort(key=lambda x: x['frame'])
    return keyframes

def get_pipeline(model_path, vae_name, device, run_id=None):
    global pipeline_cache
    with pipeline_lock:
        vae_path = None
        if vae_name and vae_name != "None":
            vae_path = os.path.join(VAE_DIR, vae_name)
            if not os.path.exists(vae_path):
                vae_path = None

        if (pipeline_cache['model_path'] == model_path and 
            pipeline_cache['vae_path'] == vae_path and 
            pipeline_cache['device'] == device and 
            pipeline_cache['pipe'] is not None):
            return pipeline_cache['pipe']
        
        def log_msg(msg):
            print(msg)
            if run_id:
                log_to_run(run_id, msg)

        # Free memory of previous pipeline
        if pipeline_cache['pipe'] is not None:
            log_msg("[*] Freeing previous pipeline cache memory...")
            del pipeline_cache['pipe']
            pipeline_cache['pipe'] = None
            pipeline_cache['txt2img_pipe'] = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        log_msg(f"[*] Loading Stable Diffusion model: {model_path} on {device}...")
        from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline
        
        # Load Img2Img pipeline
        pipe = StableDiffusionImg2ImgPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            safety_checker=None,
            local_files_only=False
        )
        
        if vae_path:
            log_msg(f"[*] Loading external VAE weights: {vae_path}...")
            try:
                from diffusers import AutoencoderKL
                vae_dtype = torch.float16 if "cuda" in device else torch.float32
                pipe.vae = AutoencoderKL.from_single_file(
                    vae_path,
                    torch_dtype=vae_dtype,
                    local_files_only=False
                )
                log_msg("[*] External VAE loaded successfully.")
            except Exception as e:
                log_msg(f"[!] Warning: Failed to load external VAE '{vae_name}': {e}. Falling back to built-in VAE.")
        else:
            log_msg("[*] Using built-in VAE.")
        
        if "LCM" in os.path.basename(model_path).upper() or "LCM" in model_path.upper():
            log_msg("[*] Detected LCM checkpoint. Swapping to LCMScheduler...")
            from diffusers import LCMScheduler
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            
        pipe = pipe.to(device)
        
        # Enable memory optimizations
        try:
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing("max")
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            if hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()
            log_msg("[*] Pipeline memory optimizations enabled.")
        except Exception as opt_err:
            log_msg(f"[*] Warning: Could not enable pipeline memory optimizations: {opt_err}")

        # Instantiate shared txt2img pipeline sharing exact same components
        try:
            txt2img_pipe = StableDiffusionPipeline(**pipe.components)
        except Exception:
            txt2img_pipe = None

        pipeline_cache['model_path'] = model_path
        pipeline_cache['vae_path'] = vae_path
        pipeline_cache['device'] = device
        pipeline_cache['pipe'] = pipe
        pipeline_cache['txt2img_pipe'] = txt2img_pipe
        return pipe

def get_txt2img_pipeline(model_path, vae_name, device):
    pipe = get_pipeline(model_path, vae_name, device)
    if pipeline_cache.get('txt2img_pipe') is not None:
        return pipeline_cache['txt2img_pipe']
    from diffusers import StableDiffusionPipeline
    txt2img_pipe = StableDiffusionPipeline(**pipe.components)
    pipeline_cache['txt2img_pipe'] = txt2img_pipe
    return txt2img_pipe

def save_run_state(run_output_dir, params, current_frame, status="running"):
    state_path = os.path.join(run_output_dir, "state.json")
    generated_images = []
    folder_name = os.path.basename(run_output_dir)
    
    for f in range(current_frame + 1):
        frame_name = f"frame{f:04d}.png"
        frame_path = os.path.join(run_output_dir, frame_name)
        if os.path.exists(frame_path):
            generated_images.append({
                "frame": f,
                "path": f"{folder_name}/{frame_name}"
            })
            
    state_data = {
        "settings": params,
        "state": {
            "last_completed_frame": current_frame,
            "generated_images": generated_images,
            "status": status
        }
    }
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state_data, f, indent=4)

def load_all_runs_from_disk():
    global active_runs
    if not os.path.exists(OUTPUT_BASE):
        return
        
    for entry in os.listdir(OUTPUT_BASE):
        run_dir = os.path.join(OUTPUT_BASE, entry)
        if not os.path.isdir(run_dir) or entry == "novel_images_singles" or entry == "uploads":
            continue
            
        state_path = os.path.join(run_dir, "state.json")
        if os.path.exists(state_path):
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    state_data = json.load(f)
                    
                settings = state_data.get("settings", {})
                state_info = state_data.get("state", {})
                last_frame = state_info.get("last_completed_frame", 0)
                status = state_info.get("status", "completed")
                
                # Verify frames on disk
                for i in range(last_frame + 1):
                    if not os.path.exists(os.path.join(run_dir, f"frame{i:04d}.png")):
                        last_frame = max(0, i - 1)
                        if status == "success":
                            status = "incomplete"
                        break
                        
                last_image = None
                if os.path.exists(os.path.join(run_dir, f"frame{last_frame:04d}.png")):
                    last_image = f"/{entry}/frame{last_frame:04d}.png"
                    
                video_path = None
                if os.path.exists(os.path.join(run_dir, "output.mp4")):
                    video_path = f"/{entry}/output.mp4"
                    
                logs = []
                log_path = os.path.join(run_dir, "log.txt")
                if os.path.exists(log_path):
                    with open(log_path, "r", encoding="utf-8", errors="ignore") as lf:
                        logs = [line.strip() for line in lf.readlines()]
                        
                run_id = entry.replace("novel_images_", "")
                active_runs[run_id] = {
                    'status': status,
                    'current_frame': last_frame,
                    'total_frames': settings.get('frames', last_frame + 1),
                    'current_step': 0,
                    'total_steps': settings.get('steps', 5),
                    'logs': logs,
                    'last_image': last_image,
                    'video_path': video_path,
                    'error': None,
                    'params': settings,
                    'timestamp': settings.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                    'output_dir': run_dir,
                    'folder_name': entry
                }
            except Exception as e:
                print(f"[!] Error loading state for {entry}: {e}")

def log_to_run(run_id, message):
    timestamp = datetime.now().strftime('%H:%M:%S')
    log_line = f"[{timestamp}] {message}"
    print(f"[{run_id}] {log_line}")
    if run_id in active_runs:
        active_runs[run_id]['logs'].append(log_line)
        run_output_dir = active_runs[run_id].get('output_dir')
        if run_output_dir and os.path.exists(run_output_dir):
            log_path = os.path.join(run_output_dir, "log.txt")
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(log_line + "\n")
            except Exception:
                pass

def run_generation_task(run_id, params):
    run_output_dir = active_runs[run_id]['output_dir']
    os.makedirs(run_output_dir, exist_ok=True)
    active_runs[run_id]['status'] = 'running'
    
    try:
        model = params.get('model', '')
        if not model:
            model = get_default_checkpoint()
        if not model or not os.path.exists(model):
            # Check relative to checkpoints dir
            alt = os.path.join(CHECKPOINTS_DIR, os.path.basename(model))
            if os.path.exists(alt):
                model = alt
            else:
                raise ValueError(f"Model file '{model}' not found on disk.")

        lora_name = params.get('lora_name', 'None')
        lora_scale = float(params.get('lora_scale', 0.8))
        vae_name = params.get('vae_name', 'None')
        device = params.get('device', detect_default_device())
        prompt = params['prompt']
        negative = params.get('negative', '')
        width = int(params['width'])
        height = int(params['height'])
        shift = int(params['shift'])
        blur_width = int(params.get('blur_width', 8))
        blur_radius = int(params.get('blur_radius', 8))
        frames = int(params['frames'])
        steps = int(params['steps'])
        strength = float(params['strength'])
        guidance = float(params['guidance'])
        seed = int(params['seed'])
        keep_original = params.get('keep_original', True)
        color_correct = params.get('color_correct', False)
        direction = params.get('direction', 'right')
        fps = int(params.get('fps', 30))
        scale = float(params.get('scale', 1.0))
        sharpen = float(params.get('sharpen', 0.0))
        input_image_path = params.get('input_image') or params.get('input_image_path')
        mask_image_path = params.get('mask_image') or params.get('mask_image_path')

        # Check existing frame state for resuming
        start_frame_idx = 0
        state_path = os.path.join(run_output_dir, "state.json")
        if os.path.exists(state_path):
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    state_data = json.load(f)
                last_frame = state_data.get("state", {}).get("last_completed_frame", -1)
                last_frame_path = os.path.join(run_output_dir, f"frame{last_frame:04d}.png")
                if last_frame >= 0 and os.path.exists(last_frame_path):
                    log_to_run(run_id, f"Resuming from frame {last_frame:04d} (frame {last_frame + 1} of {frames - 1})...")
                    start_frame_idx = last_frame + 1
                    current = Image.open(last_frame_path).convert("RGB")
                    current = current.resize((width, height), resample=LANCZOS)
            except Exception as e:
                log_to_run(run_id, f"Notice resuming state: {e}")

        log_to_run(run_id, "Initializing Stable Diffusion pipeline...")
        pipe = get_pipeline(model, vae_name, device, run_id)

        # LoRA loading
        cross_kwargs = None
        if hasattr(pipe, "unload_lora_weights"):
            try:
                pipe.unload_lora_weights()
            except Exception:
                pass

        if lora_name and lora_name != "None":
            lora_path = os.path.join(LORAS_DIR, lora_name)
            if not os.path.exists(lora_path) and os.path.exists(lora_name):
                lora_path = lora_name
            if os.path.exists(lora_path):
                log_to_run(run_id, f"Loading LoRA weights: {lora_name}...")
                pipe.load_lora_weights(lora_path)
                cross_kwargs = {"scale": lora_scale}
                log_to_run(run_id, f"LoRA applied (scale: {lora_scale})")
            else:
                log_to_run(run_id, f"Warning: LoRA file not found: {lora_path}")

        generator = torch.Generator(device=device).manual_seed(seed)

        embeds_cache = {}
        # Precompute keyframe chunks to find maximum chunk length needed for consistent blending
        initial_keyframes = parse_prompt_travel(prompt)
        chunk_cap = (pipe.tokenizer.model_max_length or 77) - 2
        max_prompt_chunks = 1
        for kf in initial_keyframes:
            p_text = kf['prompt']
            p_parts = parse_prompt_weights(p_text or "")
            p_len = sum(len(pipe.tokenizer(t, add_special_tokens=False).input_ids) for t, _ in p_parts)
            p_chunks = max(1, (p_len + chunk_cap - 1) // chunk_cap) if p_len > 0 else 1
            if p_chunks > max_prompt_chunks:
                max_prompt_chunks = p_chunks

        n_parts = parse_prompt_weights(negative or "")
        n_len = sum(len(pipe.tokenizer(t, add_special_tokens=False).input_ids) for t, _ in n_parts)
        n_chunks = max(1, (n_len + chunk_cap - 1) // chunk_cap) if n_len > 0 else 1
        run_max_chunks = max(max_prompt_chunks, n_chunks)

        def get_embeds(prompt_text, lora_scale_val=None):
            run_params = active_runs[run_id].get('params', params)
            neg_prompt = run_params.get('negative', negative)
            cache_key = (prompt_text, neg_prompt, lora_scale_val, run_max_chunks)
            if cache_key not in embeds_cache:
                total_tokens = run_max_chunks * 77
                log_to_run(run_id, f"Encoding chunked text conditioning ({total_tokens} tokens, {run_max_chunks} chunks): '{prompt_text[:40]}...'")
                p_embeds, n_embeds = get_chunked_embed_pair(pipe, prompt_text, neg_prompt, device, lora_scale_val, min_chunks=run_max_chunks)
                embeds_cache[cache_key] = (p_embeds, n_embeds)
            return embeds_cache[cache_key]

        def step_callback(step: int, timestep: int, latents: torch.FloatTensor):
            if run_id in active_runs:
                active_runs[run_id]['current_step'] = step
                total_steps = int(active_runs[run_id].get('params', params).get('steps', steps))
                active_runs[run_id]['total_steps'] = total_steps
                
                if (step % 4 == 0 or step == total_steps - 1) and hasattr(pipe, 'vae') and pipe.vae is not None:
                    try:
                        with torch.no_grad():
                            latents_scaled = 1.0 / pipe.vae.config.scaling_factor * latents
                            sample_tensor = pipe.vae.decode(latents_scaled.to(pipe.vae.dtype)).sample
                            sample_tensor = (sample_tensor / 2 + 0.5).clamp(0, 1)
                            image_numpy = sample_tensor.cpu().permute(0, 2, 3, 1).float().numpy()
                            preview_img = pipe.numpy_to_pil(image_numpy)[0]
                            preview_path = os.path.join(run_output_dir, "preview.png")
                            preview_img.save(preview_path)
                            active_runs[run_id]['last_image'] = f"/{active_runs[run_id]['folder_name']}/preview.png"
                    except Exception:
                        pass

        with torch.no_grad():
            # Build Frame 0
            if start_frame_idx == 0:
                frame0_path = os.path.join(run_output_dir, "frame0000.png")
                
                if input_image_path and os.path.exists(input_image_path):
                    log_to_run(run_id, f"Loading input image: {input_image_path}")
                    source = Image.open(input_image_path).convert("RGB")
                    source = source.resize((width, height), resample=LANCZOS)
                    
                    if keep_original and not mask_image_path:
                        log_to_run(run_id, "Using input image directly as Frame 0000...")
                        current = source
                    else:
                        log_to_run(run_id, "Synthesizing Frame 0000 from input image...")
                        keyframes = parse_prompt_travel(prompt)
                        p_emb, n_emb = get_embeds(keyframes[0]['prompt'], lora_scale)
                        run_strength = strength if mask_image_path else 0.35
                        
                        generated_img = pipe(
                            prompt_embeds=p_emb,
                            negative_prompt_embeds=n_emb,
                            image=source,
                            strength=run_strength,
                            guidance_scale=guidance,
                            num_inference_steps=steps,
                            generator=generator,
                            cross_attention_kwargs=cross_kwargs,
                            callback=step_callback,
                            callback_steps=1
                        ).images[0]
                        generated_img = generated_img.resize((width, height), resample=LANCZOS)
                        
                        if mask_image_path and os.path.exists(mask_image_path):
                            mask = Image.open(mask_image_path).convert("L")
                            mask = mask.resize((width, height), resample=LANCZOS)
                            current = Image.composite(generated_img, source, mask)
                        else:
                            current = generated_img
                else:
                    # Pure Text-to-Image for Frame 0000
                    log_to_run(run_id, "No input image provided. Generating Frame 0000 from prompt...")
                    txt2img_p = get_txt2img_pipeline(model, vae_name, device)
                    keyframes = parse_prompt_travel(prompt)
                    p_emb, n_emb = get_embeds(keyframes[0]['prompt'], lora_scale)
                    
                    generated_img = txt2img_p(
                        prompt_embeds=p_emb,
                        negative_prompt_embeds=n_emb,
                        width=width,
                        height=height,
                        guidance_scale=guidance,
                        num_inference_steps=steps,
                        generator=generator,
                        cross_attention_kwargs=cross_kwargs,
                        callback=step_callback,
                        callback_steps=1
                    ).images[0]
                    current = generated_img.resize((width, height), resample=LANCZOS)

                current.save(frame0_path)
                active_runs[run_id]['current_frame'] = 0
                active_runs[run_id]['last_image'] = f"/{active_runs[run_id]['folder_name']}/frame0000.png"
                save_run_state(run_output_dir, active_runs[run_id]['params'], 0, "running")
                log_to_run(run_id, "Frame 0000 saved.")
                start_frame_idx = 1

            # Main Latent Movement Loop
            for frame in range(start_frame_idx, frames):
                if active_runs[run_id].get('cancelled', False):
                    log_to_run(run_id, "Movement sequence cancelled by user.")
                    active_runs[run_id]['status'] = 'cancelled'
                    save_run_state(run_output_dir, active_runs[run_id]['params'], frame - 1, "cancelled")
                    return

                log_to_run(run_id, f"Processing Frame {frame:04d} of {frames - 1}...")
                run_params = active_runs[run_id].get('params', params)
                current_prompt = run_params.get('prompt', prompt)
                current_steps = int(run_params.get('steps', steps))
                current_strength = float(run_params.get('strength', strength))
                current_guidance = float(run_params.get('guidance', guidance))
                current_lora_scale = float(run_params.get('lora_scale', lora_scale))
                direction = run_params.get('direction', direction)

                current_cross_kwargs = {"scale": current_lora_scale} if lora_name and lora_name != "None" else None

                # Prompt interpolation
                keyframes = parse_prompt_travel(current_prompt)
                inject_active = run_params.get('inject_prompt', False)
                injected_text = run_params.get('injected_prompt', '')
                if inject_active and injected_text:
                    for kf in keyframes:
                        kf['prompt'] = kf['prompt'] + ", " + injected_text

                if len(keyframes) == 1:
                    current_prompt_embeds, current_negative_prompt_embeds = get_embeds(keyframes[0]['prompt'], current_lora_scale)
                else:
                    if frame <= keyframes[0]['frame']:
                        current_prompt_embeds, current_negative_prompt_embeds = get_embeds(keyframes[0]['prompt'], current_lora_scale)
                    elif frame >= keyframes[-1]['frame']:
                        current_prompt_embeds, current_negative_prompt_embeds = get_embeds(keyframes[-1]['prompt'], current_lora_scale)
                    else:
                        start_kf = keyframes[0]
                        end_kf = keyframes[-1]
                        for i in range(len(keyframes) - 1):
                            if keyframes[i]['frame'] <= frame <= keyframes[i+1]['frame']:
                                start_kf = keyframes[i]
                                end_kf = keyframes[i+1]
                                break
                        w = (frame - start_kf['frame']) / max(1, (end_kf['frame'] - start_kf['frame']))
                        p_start, n_start = get_embeds(start_kf['prompt'], current_lora_scale)
                        p_end, n_end = get_embeds(end_kf['prompt'], current_lora_scale)
                        current_prompt_embeds = (1.0 - w) * p_start + w * p_end
                        current_negative_prompt_embeds = (1.0 - w) * n_start + w * n_end

                # Canvas motion synthesis
                if direction == 'right':
                    canvas_width = width + shift
                    canvas_height = height
                    canvas = Image.new("RGB", (canvas_width, canvas_height))
                    canvas.paste(current, (-shift, 0))
                    last_col = current.crop((width - 1, 0, width, height))
                    gap_padding = last_col.resize((canvas_width - (width - shift), canvas_height), resample=LANCZOS)
                    canvas.paste(gap_padding, (width - shift, 0))
                    blurred_canvas = canvas.copy()
                    edge = blurred_canvas.crop((canvas_width - blur_width, 0, canvas_width, canvas_height))
                    edge = edge.filter(ImageFilter.GaussianBlur(blur_radius))
                    blurred_canvas.paste(edge, (canvas_width - blur_width, 0))
                    start_blend = canvas_width - blur_width
                    end_blend = width - shift
                    mask = create_feathered_mask(canvas_width, canvas_height, direction, start_blend, end_blend)
                    blended_canvas = Image.composite(canvas, blurred_canvas, mask)

                elif direction == 'left':
                    canvas_width = width + shift
                    canvas_height = height
                    canvas = Image.new("RGB", (canvas_width, canvas_height))
                    canvas.paste(current, (shift, 0))
                    first_col = current.crop((0, 0, 1, height))
                    gap_padding = first_col.resize((2 * shift, canvas_height), resample=LANCZOS)
                    canvas.paste(gap_padding, (0, 0))
                    blurred_canvas = canvas.copy()
                    edge = blurred_canvas.crop((0, 0, blur_width, canvas_height))
                    edge = edge.filter(ImageFilter.GaussianBlur(blur_radius))
                    blurred_canvas.paste(edge, (0, 0))
                    start_blend = shift
                    end_blend = blur_width
                    mask = create_feathered_mask(canvas_width, canvas_height, direction, start_blend, end_blend)
                    blended_canvas = Image.composite(canvas, blurred_canvas, mask)

                elif direction == 'down':
                    canvas_width = width
                    canvas_height = height + shift
                    canvas = Image.new("RGB", (canvas_width, canvas_height))
                    canvas.paste(current, (0, -shift))
                    last_row = current.crop((0, height - 1, width, height))
                    gap_padding = last_row.resize((canvas_width, 2 * shift), resample=LANCZOS)
                    canvas.paste(gap_padding, (0, height - shift))
                    blurred_canvas = canvas.copy()
                    edge = blurred_canvas.crop((0, canvas_height - blur_width, canvas_width, canvas_height))
                    edge = edge.filter(ImageFilter.GaussianBlur(blur_radius))
                    blurred_canvas.paste(edge, (0, canvas_height - blur_width))
                    start_blend = canvas_height - blur_width
                    end_blend = height - shift
                    mask = create_feathered_mask(canvas_width, canvas_height, direction, start_blend, end_blend)
                    blended_canvas = Image.composite(canvas, blurred_canvas, mask)

                elif direction == 'up':
                    canvas_width = width
                    canvas_height = height + shift
                    canvas = Image.new("RGB", (canvas_width, canvas_height))
                    canvas.paste(current, (0, shift))
                    first_row = current.crop((0, 0, width, 1))
                    gap_padding = first_row.resize((canvas_width, 2 * shift), resample=LANCZOS)
                    canvas.paste(gap_padding, (0, 0))
                    blurred_canvas = canvas.copy()
                    edge = blurred_canvas.crop((0, 0, canvas_width, blur_width))
                    edge = edge.filter(ImageFilter.GaussianBlur(blur_radius))
                    blurred_canvas.paste(edge, (0, 0))
                    start_blend = shift
                    end_blend = blur_width
                    mask = create_feathered_mask(canvas_width, canvas_height, direction, start_blend, end_blend)
                    blended_canvas = Image.composite(canvas, blurred_canvas, mask)

                elif direction == 'out':
                    # Zoom Out
                    canvas_width = width
                    canvas_height = height
                    canvas = Image.new("RGB", (canvas_width, canvas_height))
                    shrunk_width = max(8, ((width - 2 * shift) // 2) * 2)
                    shrunk_height = max(8, int(round(shrunk_width * height / width) // 2) * 2)
                    shrunk = current.resize((shrunk_width, shrunk_height), resample=LANCZOS)
                    x_offset = (width - shrunk_width) // 2
                    y_offset = (height - shrunk_height) // 2
                    canvas.paste(shrunk, (x_offset, y_offset))
                    
                    # Pad border edges
                    top_row = shrunk.crop((0, 0, shrunk_width, 1))
                    canvas.paste(top_row.resize((shrunk_width, y_offset), resample=LANCZOS), (x_offset, 0))
                    bottom_row = shrunk.crop((0, shrunk_height - 1, shrunk_width, shrunk_height))
                    canvas.paste(bottom_row.resize((shrunk_width, height - (y_offset + shrunk_height)), resample=LANCZOS), (x_offset, y_offset + shrunk_height))
                    left_col = canvas.crop((x_offset, 0, x_offset + 1, height))
                    canvas.paste(left_col.resize((x_offset, height), resample=LANCZOS), (0, 0))
                    right_col = canvas.crop((x_offset + shrunk_width - 1, 0, x_offset + shrunk_width, height))
                    canvas.paste(right_col.resize((width - (x_offset + shrunk_width), height), resample=LANCZOS), (x_offset + shrunk_width, 0))
                    
                    blurred_canvas = canvas.filter(ImageFilter.GaussianBlur(blur_radius))
                    mask = create_feathered_mask(canvas_width, canvas_height, direction, x_offset, y_offset, blur_radius)
                    blended_canvas = Image.composite(canvas, blurred_canvas, mask)

                elif direction == 'in':
                    # Zoom In
                    canvas_width = width - (width % 2)
                    canvas_height = height - (height % 2)
                    crop_width = max(8, ((width - 2 * shift) // 2) * 2)
                    crop_height = max(8, int(round(crop_width * height / width) // 2) * 2)
                    x_offset = (width - crop_width) // 2
                    y_offset = (height - crop_height) // 2
                    cropped = current.crop((x_offset, y_offset, x_offset + crop_width, y_offset + crop_height))
                    canvas = cropped.resize((width, height), resample=LANCZOS)
                    blurred_canvas = canvas.filter(ImageFilter.GaussianBlur(blur_radius))
                    mask = create_feathered_mask(canvas_width, canvas_height, "out", x_offset, y_offset, blur_radius)
                    blended_canvas = Image.composite(canvas, blurred_canvas, mask)

                else:
                    blended_canvas = current

                # Safety check for minimal effective steps
                eff_steps = int(current_steps * current_strength)
                if eff_steps < 2:
                    current_strength = min(1.0, max(current_strength, 2.0 / current_steps))

                # Diffusion Step
                expanded = pipe(
                    prompt_embeds=current_prompt_embeds,
                    negative_prompt_embeds=current_negative_prompt_embeds,
                    image=blended_canvas,
                    strength=current_strength,
                    guidance_scale=current_guidance,
                    num_inference_steps=current_steps,
                    generator=generator,
                    cross_attention_kwargs=current_cross_kwargs,
                    callback=step_callback,
                    callback_steps=1
                ).images[0]

                # Crop canvas back to target resolution
                if direction == 'right':
                    current = expanded.crop((shift, 0, shift + width, height))
                elif direction == 'left':
                    current = expanded.crop((0, 0, width, height))
                elif direction == 'down':
                    current = expanded.crop((0, shift, width, shift + height))
                elif direction == 'up':
                    current = expanded.crop((0, 0, width, height))
                else:
                    current = expanded

                if color_correct and frame > 0:
                    frame0_path = os.path.join(run_output_dir, "frame0000.png")
                    if os.path.exists(frame0_path):
                        try:
                            current = correct_colors(current, Image.open(frame0_path).convert("RGB"))
                        except Exception:
                            pass

                frame_path = os.path.join(run_output_dir, f"frame{frame:04d}.png")
                current.save(frame_path)

                active_runs[run_id]['current_frame'] = frame
                active_runs[run_id]['last_image'] = f"/{active_runs[run_id]['folder_name']}/frame{frame:04d}.png"
                save_run_state(run_output_dir, active_runs[run_id]['params'], frame, "running")
                log_to_run(run_id, f"Frame {frame:04d} saved.")

                # Cleanup per frame
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Final Video Assembly
        log_to_run(run_id, "Compiling video using FFmpeg...")
        run_params = active_runs[run_id].get('params', params)
        current_fps = int(run_params.get('fps', fps))
        current_scale = float(run_params.get('scale', scale))
        current_sharpen = float(run_params.get('sharpen', sharpen))

        video_success = compile_video(run_output_dir, current_fps, current_scale, current_sharpen)
        
        if video_success or os.path.exists(os.path.join(run_output_dir, "output.mp4")):
            active_runs[run_id]['status'] = 'success'
            active_runs[run_id]['video_path'] = f"/{active_runs[run_id]['folder_name']}/output.mp4"
            save_run_state(run_output_dir, active_runs[run_id]['params'], frames - 1, "success")
            log_to_run(run_id, "SUCCESS: Latent Movement sequence completed!")
        else:
            active_runs[run_id]['status'] = 'failed'
            active_runs[run_id]['error'] = "Video compilation failed."
            save_run_state(run_output_dir, active_runs[run_id]['params'], frames - 1, "failed")
            log_to_run(run_id, "ERROR: Video compilation failed.")

    except Exception as e:
        error_msg = traceback.format_exc()
        log_to_run(run_id, f"ERROR: {str(e)}")
        log_to_run(run_id, error_msg)
        active_runs[run_id]['status'] = 'failed'
        active_runs[run_id]['error'] = str(e)
        try:
            curr = active_runs[run_id].get('current_frame', -1)
            save_run_state(run_output_dir, active_runs[run_id]['params'], curr, "failed")
        except Exception:
            pass

# Routes
@app.route('/')
def index():
    default_dev = detect_default_device()
    default_model = get_default_checkpoint()
    return render_template('index0.html', default_device=default_dev, default_model=default_model)

@app.route('/api/models')
def get_models():
    models = scan_checkpoints()
    return jsonify({
        'success': True, 
        'models': models, 
        'models_dir': CHECKPOINTS_DIR,
        'count': len(models)
    })

@app.route('/api/loras')
def get_loras():
    loras = scan_loras()
    return jsonify({'success': True, 'loras': loras})

@app.route('/api/vaes')
def get_vaes():
    vaes = scan_vaes()
    return jsonify({'success': True, 'vaes': vaes})

@app.route('/api/models/config', methods=['GET', 'POST'])
def handle_models_config():
    global MODELS_BASE, CHECKPOINTS_DIR, LORAS_DIR, VAE_DIR
    if request.method == 'POST':
        data = request.get_json() or {}
        new_dir = data.get('models_dir')
        if not new_dir or not os.path.exists(new_dir):
            return jsonify({'success': False, 'error': 'Directory does not exist.'}), 400
        
        cfg = load_app_config()
        cfg['models_dir'] = os.path.abspath(new_dir)
        save_app_config(cfg)
        
        MODELS_BASE = os.path.abspath(new_dir)
        CHECKPOINTS_DIR = os.path.join(MODELS_BASE, "checkpoints") if os.path.exists(os.path.join(MODELS_BASE, "checkpoints")) else MODELS_BASE
        LORAS_DIR = os.path.join(MODELS_BASE, "loras")
        VAE_DIR = os.path.join(MODELS_BASE, "vae")
        
        return jsonify({
            'success': True,
            'models_dir': MODELS_BASE,
            'checkpoints': scan_checkpoints(),
            'loras': scan_loras(),
            'vaes': scan_vaes()
        })
    else:
        return jsonify({
            'success': True,
            'models_dir': MODELS_BASE,
            'checkpoints': scan_checkpoints(),
            'loras': scan_loras(),
            'vaes': scan_vaes()
        })

@app.route('/api/models/rescan', methods=['GET', 'POST'])
def rescan_models():
    return jsonify({
        'success': True,
        'models_dir': MODELS_BASE,
        'checkpoints': scan_checkpoints(),
        'loras': scan_loras(),
        'vaes': scan_vaes(),
        'count': len(scan_checkpoints())
    })

# Single Image Generation Endpoint
@app.route('/api/generate_single', methods=['POST'])
def generate_single():
    try:
        prompt = request.form.get('prompt', '')
        negative = request.form.get('negative', '')
        model = request.form.get('model', '')
        if not model:
            model = get_default_checkpoint()
        if not model or not os.path.exists(model):
            alt = os.path.join(CHECKPOINTS_DIR, os.path.basename(model))
            if os.path.exists(alt):
                model = alt
            else:
                return jsonify({'success': False, 'error': f"Model checkpoint not found: '{model}'"}), 400

        lora_name = request.form.get('lora_name', 'None')
        lora_scale = float(request.form.get('lora_scale', 0.8))
        vae_name = request.form.get('vae_name', 'None')
        device = request.form.get('device', detect_default_device())
        width = int(request.form.get('width', 768))
        height = int(request.form.get('height', 512))
        steps = int(request.form.get('steps', 25))
        guidance = float(request.form.get('guidance', 7.0))
        strength = float(request.form.get('strength', 0.65))
        
        seed_str = request.form.get('seed', '')
        seed = int(seed_str) if seed_str.strip().isdigit() else int(time.time() * 1000) % 2147483647
        generator = torch.Generator(device=device).manual_seed(seed)

        # Check for input image (txt2img vs img2img)
        image_file = request.files.get('input_image_file')
        input_image_path = None
        if image_file and image_file.filename != '':
            fname = f"{str(uuid.uuid4())[:8]}_{image_file.filename}"
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            image_file.save(save_path)
            input_image_path = save_path
        elif request.form.get('input_image_path'):
            input_image_path = request.form.get('input_image_path')

        print(f"[*] Generating Single Image (seed={seed}, model={os.path.basename(model)})...")
        pipe = get_pipeline(model, vae_name, device)

        # Apply LoRA if requested
        cross_kwargs = None
        if hasattr(pipe, "unload_lora_weights"):
            try:
                pipe.unload_lora_weights()
            except Exception:
                pass

        if lora_name and lora_name != "None":
            lora_path = os.path.join(LORAS_DIR, lora_name)
            if not os.path.exists(lora_path) and os.path.exists(lora_name):
                lora_path = lora_name
            if os.path.exists(lora_path):
                pipe.load_lora_weights(lora_path)
                cross_kwargs = {"scale": lora_scale}

        # Text Embeddings with Chunking support (>77 tokens)
        p_emb, n_emb = get_chunked_embed_pair(pipe, prompt, negative, device, lora_scale)

        with torch.no_grad():
            if input_image_path and os.path.exists(input_image_path):
                source = Image.open(input_image_path).convert("RGB")
                source = source.resize((width, height), resample=LANCZOS)
                output_image = pipe(
                    prompt_embeds=p_emb,
                    negative_prompt_embeds=n_emb,
                    image=source,
                    strength=strength,
                    guidance_scale=guidance,
                    num_inference_steps=steps,
                    generator=generator,
                    cross_attention_kwargs=cross_kwargs
                ).images[0]
            else:
                txt2img_p = get_txt2img_pipeline(model, vae_name, device)
                output_image = txt2img_p(
                    prompt_embeds=p_emb,
                    negative_prompt_embeds=n_emb,
                    width=width,
                    height=height,
                    guidance_scale=guidance,
                    num_inference_steps=steps,
                    generator=generator,
                    cross_attention_kwargs=cross_kwargs
                ).images[0]

        # Save single image to novel_images_singles
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"single_{timestamp_str}_{seed}.png"
        filepath = os.path.join(SINGLES_DIR, filename)

        # Attach PNG metadata
        pnginfo = PngInfo()
        pnginfo.add_text("prompt", prompt)
        pnginfo.add_text("negative", negative)
        pnginfo.add_text("model", os.path.basename(model))
        pnginfo.add_text("seed", str(seed))
        pnginfo.add_text("steps", str(steps))
        pnginfo.add_text("guidance", str(guidance))
        output_image.save(filepath, pnginfo=pnginfo)

        meta = {
            "filename": filename,
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "prompt": prompt,
            "negative": negative,
            "model": os.path.basename(model),
            "lora": lora_name,
            "lora_scale": lora_scale,
            "vae": vae_name,
            "seed": seed,
            "width": width,
            "height": height,
            "steps": steps,
            "guidance": guidance,
            "device": device
        }
        meta_filepath = os.path.join(SINGLES_DIR, f"single_{timestamp_str}_{seed}.json")
        with open(meta_filepath, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4)

        return jsonify({
            'success': True,
            'filename': filename,
            'url': f"/singles/{filename}",
            'filepath': filepath,
            'seed': seed,
            'metadata': meta
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

# Start Latent Movement Sequence
@app.route('/api/start', methods=['POST'])
def start_generation():
    def get_form_int(key, default):
        val = request.form.get(key)
        try: return int(val)
        except: return default

    def get_form_float(key, default):
        val = request.form.get(key)
        try: return float(val)
        except: return default

    image_file = request.files.get('input_image_file')
    input_image_path = None
    
    if image_file and image_file.filename != '':
        fname = f"{str(uuid.uuid4())[:8]}_{image_file.filename}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        image_file.save(save_path)
        input_image_path = save_path
    else:
        input_image_path = request.form.get('input_image_path')

    mask_file = request.files.get('mask_image_file')
    mask_image_path = None
    if mask_file and mask_file.filename != '':
        mask_fname = f"mask_{str(uuid.uuid4())[:8]}.png"
        mask_save_path = os.path.join(app.config['UPLOAD_FOLDER'], mask_fname)
        mask_file.save(mask_save_path)
        mask_image_path = mask_save_path

    model_req = request.form.get('model', '')
    if not model_req:
        model_req = get_default_checkpoint()

    params = {
        'input_image': input_image_path,
        'mask_image': mask_image_path,
        'prompt': request.form.get('prompt', ''),
        'negative': request.form.get('negative', ''),
        'width': get_form_int('width', 768),
        'height': get_form_int('height', 512),
        'shift': get_form_int('shift', 4),
        'blur_width': get_form_int('blur_width', 8),
        'blur_radius': get_form_int('blur_radius', 8),
        'frames': get_form_int('frames', 90),
        'steps': get_form_int('steps', 25),
        'strength': get_form_float('strength', 0.38),
        'guidance': get_form_float('guidance', 7.0),
        'device': request.form.get('device', detect_default_device()),
        'model': model_req,
        'seed': get_form_int('seed', int(time.time())),
        'keep_original': request.form.get('keep_original') == 'true',
        'color_correct': request.form.get('color_correct') == 'true',
        'fps': get_form_int('fps', 30),
        'scale': get_form_float('scale', 1.0),
        'sharpen': get_form_float('sharpen', 0.0),
        'lora_name': request.form.get('lora_name', 'None'),
        'lora_scale': get_form_float('lora_scale', 0.8),
        'vae_name': request.form.get('vae_name', 'None'),
        'inject_prompt': request.form.get('inject_prompt') == 'true',
        'injected_prompt': request.form.get('injected_prompt', ''),
        'direction': request.form.get('direction', 'right')
    }

    run_id = str(uuid.uuid4())[:8]
    folder_name = f"novel_images_{run_id}"
    run_output_dir = os.path.join(OUTPUT_BASE, folder_name)

    active_runs[run_id] = {
        'status': 'pending',
        'current_frame': 0,
        'total_frames': params['frames'],
        'current_step': 0,
        'total_steps': 0,
        'logs': [f"[System] Initializing Latent Movement session in {folder_name}"],
        'last_image': None,
        'video_path': None,
        'error': None,
        'params': params,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'output_dir': run_output_dir,
        'folder_name': folder_name
    }

    thread = threading.Thread(target=run_generation_task, args=(run_id, params))
    thread.daemon = True
    thread.start()

    return jsonify({'success': True, 'run_id': run_id, 'folder_name': folder_name})

@app.route('/api/resume/<run_id>', methods=['POST'])
def resume_generation(run_id):
    if run_id not in active_runs:
        load_all_runs_from_disk()
        if run_id not in active_runs:
            return jsonify({'success': False, 'error': 'Run not found.'}), 404

    run = active_runs[run_id]
    params = run['params']
    
    active_runs[run_id]['status'] = 'pending'
    active_runs[run_id]['cancelled'] = False
    active_runs[run_id]['error'] = None
    active_runs[run_id]['current_step'] = 0
    active_runs[run_id]['total_steps'] = 0
    active_runs[run_id]['logs'].append("[System] Resuming Latent Movement from last completed frame...")

    thread = threading.Thread(target=run_generation_task, args=(run_id, params))
    thread.daemon = True
    thread.start()

    return jsonify({'success': True, 'run_id': run_id})

@app.route('/api/delete/<run_id>', methods=['POST'])
def delete_run(run_id):
    if run_id in active_runs:
        target_dir = active_runs[run_id].get('output_dir')
        del active_runs[run_id]
    else:
        target_dir = os.path.join(OUTPUT_BASE, f"novel_images_{run_id}")
        
    if target_dir and os.path.exists(target_dir):
        try:
            shutil.rmtree(target_dir)
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    return jsonify({'success': True})

@app.route('/api/status/<run_id>')
def get_status(run_id):
    if run_id not in active_runs:
        load_all_runs_from_disk()
        if run_id not in active_runs:
            return jsonify({'success': False, 'error': 'Run not found'}), 404

    run = active_runs[run_id]
    folder_name = run.get('folder_name', f"novel_images_{run_id}")
    run_output_dir = run.get('output_dir', os.path.join(OUTPUT_BASE, folder_name))
    
    generated_images = []
    if os.path.exists(run_output_dir):
        for f in range(run['current_frame'] + 1):
            frame_name = f"frame{f:04d}.png"
            if os.path.exists(os.path.join(run_output_dir, frame_name)):
                generated_images.append({
                    "frame": f,
                    "path": f"{folder_name}/{frame_name}"
                })

    return jsonify({
        'success': True,
        'status': run['status'],
        'current_frame': run['current_frame'],
        'total_frames': run['total_frames'],
        'current_step': run['current_step'],
        'total_steps': run['total_steps'],
        'last_image': run['last_image'],
        'video_path': run['video_path'],
        'error': run['error'],
        'params': run['params'],
        'generated_images': generated_images
    })

@app.route('/api/logs/<run_id>')
def get_logs(run_id):
    if run_id not in active_runs:
        return jsonify({'success': False, 'error': 'Run not found'}), 404
    offset = int(request.args.get('offset', 0))
    logs = active_runs[run_id]['logs'][offset:]
    return jsonify({
        'success': True,
        'logs': logs,
        'next_offset': len(active_runs[run_id]['logs'])
    })

@app.route('/api/cancel/<run_id>', methods=['POST'])
def cancel_run(run_id):
    if run_id not in active_runs:
        return jsonify({'success': False, 'error': 'Run not found'}), 404
    active_runs[run_id]['cancelled'] = True
    return jsonify({'success': True})

@app.route('/api/update/<run_id>', methods=['POST'])
def update_run_params(run_id):
    if run_id not in active_runs:
        return jsonify({'success': False, 'error': 'Run not found'}), 404
    data = request.get_json() or {}
    params = active_runs[run_id]['params']
    for k, v in data.items():
        if k in params:
            orig_type = type(params[k])
            try:
                if orig_type == bool:
                    params[k] = str(v).lower() == 'true'
                elif orig_type == int:
                    params[k] = int(v)
                elif orig_type == float:
                    params[k] = float(v)
                else:
                    params[k] = v
            except:
                params[k] = v
    log_to_run(run_id, f"Parameters updated dynamically: {data}")
    return jsonify({'success': True})

@app.route('/api/draft', methods=['GET', 'POST'])
def handle_draft():
    def get_default_draft():
        return {
            'prompt': '0:image another dimension on an alien planet, a tropical jungle is slightly visible. Cinematic lighting, photorealistic.\n50:create a stone path leading into the alien tropical jungle, foliage and tropical flowers, cinematic lighting, photorealistic.',
            'negative': 'text, watermark, blurry, low quality, bad anatomy, deformed, muted colors, washed out, desaturated',
            'model': get_default_checkpoint(),
            'lora_name': 'None',
            'lora_scale': 0.1,
            'vae_name': 'None',
            'device': detect_default_device(),
            'seed': '',
            'width': 768,
            'height': 512,
            'shift': 1,
            'frames': 90,
            'blur_width': 4,
            'blur_radius': 4,
            'keep_original': True,
            'color_correct': False,
            'inject_prompt': False,
            'injected_prompt': '',
            'direction': 'right',
            'steps': 25,
            'strength': 0.38,
            'guidance': 7.0,
            'fps': 30,
            'scale': 1.0,
            'sharpen': 0.0
        }

    if request.method == 'POST':
        data = request.get_json() or {}
        draft_data = get_default_draft()
        if os.path.exists(DRAFT_PATH):
            try:
                with open(DRAFT_PATH, 'r', encoding='utf-8') as f:
                    draft_data.update(json.load(f))
            except Exception:
                pass
        draft_data.update(data)
        try:
            with open(DRAFT_PATH, 'w', encoding='utf-8') as f:
                json.dump(draft_data, f, indent=4)
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    else:
        if os.path.exists(DRAFT_PATH):
            try:
                with open(DRAFT_PATH, 'r', encoding='utf-8') as f:
                    return jsonify({'success': True, 'draft': json.load(f)})
            except Exception:
                pass
        return jsonify({'success': True, 'draft': get_default_draft()})

@app.route('/api/runs')
def list_runs():
    load_all_runs_from_disk()
    summary = []
    for rid, run in active_runs.items():
        summary.append({
            'run_id': rid,
            'folder_name': run.get('folder_name', f"novel_images_{rid}"),
            'status': run['status'],
            'timestamp': run['timestamp'],
            'current_frame': run['current_frame'],
            'total_frames': run['total_frames'],
            'prompt': run['params'].get('prompt', '')
        })
    summary.sort(key=lambda x: x['timestamp'])
    return jsonify({'success': True, 'runs': summary})

# Static asset serving
@app.route('/<folder_name>/<filename>')
def serve_direct_frame(folder_name, filename):
    if folder_name.startswith("novel_images_"):
        target = os.path.join(OUTPUT_BASE, folder_name)
        if os.path.exists(target):
            return send_from_directory(target, filename)
    return jsonify({'error': 'File not found'}), 404

@app.route('/runs/<path:filename>')
def serve_run_file(filename):
    # Support both runs/folder_name/file and runs/file
    parts = filename.split('/', 1)
    if len(parts) == 2:
        folder, subpath = parts
        target = os.path.join(OUTPUT_BASE, folder)
        if os.path.exists(target):
            return send_from_directory(target, subpath)
    return send_from_directory(OUTPUT_BASE, filename)

@app.route('/novel_images/<path:filename>')
def serve_novel_images(filename):
    return serve_run_file(filename)

@app.route('/singles/<path:filename>')
def serve_single_file(filename):
    return send_from_directory(SINGLES_DIR, filename)

@app.route('/uploads/<path:filename>')
def serve_upload_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

def cli_list_models():
    print("\n" + "=" * 60)
    print(" 🎬 LatentMovement Studio Models Scanner")
    print("=" * 60)
    checkpoints = scan_checkpoints()
    loras = scan_loras()
    vaes = scan_vaes()
    default_ck = get_default_checkpoint()
    default_base = os.path.basename(default_ck) if default_ck else ""

    print(f"\n📂 Checkpoints ({len(checkpoints)} found) in {CHECKPOINTS_DIR}:")
    for m in checkpoints:
        default_mark = " (Default)" if m == default_base else ""
        print(f"  • {m}{default_mark}")
    
    print(f"\n📂 LoRAs ({len(loras)} found) in {LORAS_DIR}:")
    for lora in loras:
        print(f"  • {lora}")
        
    print(f"\n📂 VAEs ({len(vaes)} found) in {VAE_DIR}:")
    for vae in vaes:
        print(f"  • {vae}")
    print("\n" + "=" * 60 + "\n")

def cli_generate_single(args):
    print("\n" + "=" * 60)
    print(" 🎬 LatentMovement CLI // Single Image Generation")
    print("=" * 60)
    
    model = args.model
    if not model:
        model = get_default_checkpoint()
    if not model or not os.path.exists(model):
        alt = os.path.join(CHECKPOINTS_DIR, os.path.basename(model)) if model else None
        if alt and os.path.exists(alt):
            model = alt
        else:
            print(f"[!] Error: Model checkpoint not found: '{args.model}'")
            print("    Run with --list-models to see available checkpoints.")
            sys.exit(1)

    device = args.device or detect_default_device()
    seed = args.seed if args.seed is not None else int(time.time() * 1000) % 2147483647
    generator = torch.Generator(device=device).manual_seed(seed)
    
    print(f"[*] Model:    {os.path.basename(model)}")
    print(f"[*] Prompt:   {args.prompt}")
    print(f"[*] Negative: {args.negative}")
    print(f"[*] Seed:     {seed}")
    print(f"[*] Steps:    {args.steps} | Guidance: {args.guidance} | Size: {args.width}x{args.height}")
    print(f"[*] Device:   {device}")
    
    pipe = get_pipeline(model, args.vae, device)
    
    cross_kwargs = None
    if hasattr(pipe, "unload_lora_weights"):
        try: pipe.unload_lora_weights()
        except Exception: pass
        
    if args.lora and args.lora != "None":
        lora_path = os.path.join(LORAS_DIR, args.lora)
        if not os.path.exists(lora_path) and os.path.exists(args.lora):
            lora_path = args.lora
        if os.path.exists(lora_path):
            print(f"[*] Applying LoRA: {os.path.basename(lora_path)} (scale: {args.lora_scale})")
            pipe.load_lora_weights(lora_path)
            cross_kwargs = {"scale": args.lora_scale}
            
    print("[*] Encoding chunked text conditioning (unlimited token length)...")
    p_emb, n_emb = get_chunked_embed_pair(pipe, args.prompt, args.negative, device, args.lora_scale)
    chunks_used = p_emb.shape[1] // 77
    print(f"[*] Text token length: {p_emb.shape[1]} tokens ({chunks_used} chunk{'s' if chunks_used > 1 else ''} of 77)")

    with torch.no_grad():
        if args.input_image and os.path.exists(args.input_image):
            print(f"[*] Loading input image for img2img: {args.input_image} (strength: {args.strength})")
            source = Image.open(args.input_image).convert("RGB")
            source = source.resize((args.width, args.height), resample=LANCZOS)
            output_image = pipe(
                prompt_embeds=p_emb,
                negative_prompt_embeds=n_emb,
                image=source,
                strength=args.strength,
                guidance_scale=args.guidance,
                num_inference_steps=args.steps,
                generator=generator,
                cross_attention_kwargs=cross_kwargs
            ).images[0]
        else:
            txt2img_p = get_txt2img_pipeline(model, args.vae, device)
            output_image = txt2img_p(
                prompt_embeds=p_emb,
                negative_prompt_embeds=n_emb,
                width=args.width,
                height=args.height,
                guidance_scale=args.guidance,
                num_inference_steps=args.steps,
                generator=generator,
                cross_attention_kwargs=cross_kwargs
            ).images[0]

    # Save output
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.output:
        save_path = args.output
        parent_dir = os.path.dirname(os.path.abspath(save_path))
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
    else:
        save_path = os.path.join(SINGLES_DIR, f"single_{timestamp_str}_{seed}.png")

    pnginfo = PngInfo()
    pnginfo.add_text("prompt", args.prompt)
    pnginfo.add_text("negative", args.negative)
    pnginfo.add_text("model", os.path.basename(model))
    pnginfo.add_text("seed", str(seed))
    pnginfo.add_text("steps", str(args.steps))
    pnginfo.add_text("guidance", str(args.guidance))
    output_image.save(save_path, pnginfo=pnginfo)
    
    print(f"\n[SUCCESS] Single image saved to: {save_path}\n")
    return save_path

def cli_generate_motion(args):
    print("\n" + "=" * 60)
    print(" 🎬 LatentMovement CLI // Camera Motion Sequence")
    print("=" * 60)
    
    model = args.model or get_default_checkpoint()
    if not model or not os.path.exists(model):
        alt = os.path.join(CHECKPOINTS_DIR, os.path.basename(model)) if model else None
        if alt and os.path.exists(alt):
            model = alt
        else:
            print(f"[!] Error: Model checkpoint not found: '{args.model}'")
            print("    Run with --list-models to see available checkpoints.")
            sys.exit(1)

    run_id = f"cli_{uuid.uuid4().hex[:8]}"
    direction_val = args.direction.lower().replace("zoom-", "").replace("zoom_", "")
    run_output_dir = os.path.join(OUTPUT_BASE, f"novel_images_{run_id}")
    os.makedirs(run_output_dir, exist_ok=True)

    params = {
        'prompt': args.prompt,
        'negative': args.negative,
        'model': model,
        'lora_name': args.lora or 'None',
        'lora_scale': args.lora_scale,
        'vae_name': args.vae or 'None',
        'device': args.device or detect_default_device(),
        'width': args.width,
        'height': args.height,
        'steps': args.steps,
        'guidance': args.guidance,
        'strength': args.strength,
        'direction': direction_val,
        'shift': args.shift,
        'blur_width': 8,
        'blur_radius': 8,
        'frames': args.frames,
        'seed': args.seed if args.seed is not None else int(time.time() * 1000) % 2147483647,
        'input_image': args.input_image,
        'input_image_path': args.input_image,
        'mask_image': None,
        'mask_image_path': None,
        'keep_original': args.keep_original,
        'color_correct': args.color_correct,
        'injected_prompt': args.injected_prompt or '',
        'inject_prompt': bool(args.injected_prompt),
        'fps': args.fps,
        'scale': args.scale,
        'sharpen': args.sharpen
    }
    
    print(f"[*] Motion Run ID:  {run_id}")
    print(f"[*] Direction:      {params['direction'].upper()}")
    print(f"[*] Target Frames:  {params['frames']}")
    print(f"[*] Shift Step:     {params['shift']} px/frame")
    print(f"[*] Model:          {os.path.basename(params['model'])}")
    print(f"[*] Output Folder:  {run_output_dir}")
    
    # Run synchronously in CLI
    active_runs[run_id] = {
        'id': run_id,
        'params': params,
        'folder_name': f"novel_images_{run_id}",
        'output_dir': run_output_dir,
        'status': 'initializing',
        'current_frame': 0,
        'total_frames': params['frames'],
        'current_step': 0,
        'total_steps': 0,
        'logs': [f"[System] Initializing Latent Movement CLI session in novel_images_{run_id}"],
        'last_image': None,
        'video_path': None,
        'error': None,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'cancelled': False
    }
    
    run_generation_task(run_id, params)
    
    print(f"\n[SUCCESS] Motion generation complete for run {run_id}!")
    run_dir = os.path.join(OUTPUT_BASE, f"novel_images_{run_id}")
    video_path = os.path.join(run_dir, "output.mp4")
    if os.path.exists(video_path):
        print(f"[*] Video output: {video_path}")
    print("=" * 60 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LatentMovement Studio v2 - Server & Standalone CLI Generator")
    parser.add_argument("--cli", action="store_true", help="Run standalone generation directly from terminal without launching web server")
    parser.add_argument("--mode", choices=["single", "motion"], default="single", help="Generation mode: 'single' (image) or 'motion' (camera movement)")
    parser.add_argument("--prompt", type=str, default="A beautiful alien landscape, bioluminescent jungle, highly detailed, 8k", help="Text prompt (supports >77 token chunking and prompt traveling)")
    parser.add_argument("--negative", type=str, default="text, watermark, blurry, low quality, bad anatomy, deformed", help="Negative prompt")
    parser.add_argument("--model", type=str, default=None, help="Model checkpoint path or filename in models/checkpoints/")
    parser.add_argument("--vae", type=str, default=None, help="External VAE model path or filename in models/vae/")
    parser.add_argument("--lora", type=str, default=None, help="LoRA model path or filename in models/loras/")
    parser.add_argument("--lora-scale", type=float, default=0.8, help="LoRA scale (0.0 to 1.0)")
    parser.add_argument("--steps", type=int, default=5, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=1.5, help="Classifier-free guidance scale")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=768, help="Image height")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: random)")
    parser.add_argument("--input-image", type=str, default=None, help="Path to input image for img2img or base frame for motion")
    parser.add_argument("--strength", type=float, default=0.48, help="Denoising strength for img2img or motion")
    parser.add_argument("--output", type=str, default=None, help="Custom output image path or motion folder")

    # Motion-specific arguments:
    parser.add_argument("--direction", choices=["right", "left", "up", "down", "in", "out", "zoom-in", "zoom-out", "static"], default="right", help="Camera motion direction (right, left, up, down, in/zoom-in, out/zoom-out, static)")
    parser.add_argument("--frames", type=int, default=10, help="Number of motion frames to generate")
    parser.add_argument("--shift", type=int, default=2, help="Pixel shift amount per frame")
    parser.add_argument("--keep-original", action="store_true", help="Keep the input image as exact Frame 0000 without regenerating it")
    parser.add_argument("--color-correct", action="store_true", help="Apply color correction across motion frames")
    parser.add_argument("--injected-prompt", type=str, default=None, help="Prompt to dynamically inject into frames")
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate")
    parser.add_argument("--scale", type=float, default=1.0, help="Video upscale factor")
    parser.add_argument("--sharpen", type=float, default=0.0, help="Video unsharp filter strength")

    # Utilities:
    parser.add_argument("--list-models", action="store_true", help="List all detected checkpoints, LoRAs, and VAEs and exit")
    parser.add_argument("--port", type=int, default=5002, help="Port to bind when running as server")
    parser.add_argument("--device", type=str, default=None, help="Computation device (cpu, cuda)")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU thread limit (default: 4)")

    args, unknown = parser.parse_known_args()

    # Re-apply thread limits if specified via CLI
    if args.threads:
        try:
            torch.set_num_threads(args.threads)
        except Exception:
            pass

    if args.list_models:
        cli_list_models()
        sys.exit(0)

    if args.cli:
        if args.mode == "motion" or (args.frames > 1 and args.direction and args.input_image):
            cli_generate_motion(args)
        else:
            cli_generate_single(args)
        sys.exit(0)

    # Server mode (default when --cli is omitted)
    port = args.port
    if 'LATENT_MOVEMENT_PORT' in os.environ:
        try: port = int(os.environ['LATENT_MOVEMENT_PORT'])
        except: pass

    print(f"[*] LatentMovement Server v2 starting on http://127.0.0.1:{port}")
    print(f"[*] Models Base Directory: {MODELS_BASE}")
    print(f"[*] Output Base Directory: {OUTPUT_BASE}")
    print(f"[*] Singles Directory:     {SINGLES_DIR}")
    print(f"[*] Standalone CLI Mode:   Supported (run with --cli or --list-models)")
    print(f"[*] Long Prompts:          Chunked encoding enabled (>77 tokens)")
    
    load_all_runs_from_disk()
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
