import os
import sys
import json
import time
import uuid
import shutil
import threading
import traceback
import subprocess
import signal
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image, ImageFilter
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

# Limit PyTorch to leave CPU cores free to prevent the entire computer from locking up/freezing during CPU inference
try:
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    if cpu_count > 2:
        torch.set_num_threads(max(1, cpu_count - 2))
        print(f"[*] Set PyTorch thread limit to {max(1, cpu_count - 2)} out of {cpu_count} CPU cores.")
except Exception as e:
    print(f"[*] Warning: Could not configure PyTorch thread limit: {e}")


LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS

def create_feathered_mask(canvas_width, canvas_height, direction, start_blend, end_blend, blur_radius=8):
    """
    Creates a feathered alpha mask for the outpainting region based on direction.
    - 255 (white): use unblurred original (canvas).
    - 0 (black): use blurred edge (blurred_canvas).
    - values in between: blend.
    """
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
        
    elif direction == "out":
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
        # Fallback to horizontal right
        gradient = Image.new("L", (canvas_width, 1))
        for x in range(canvas_width):
            val = 255 if x < start_blend else 0
            gradient.putpixel((x, 0), val)
        mask = gradient.resize((canvas_width, canvas_height))
        
    return mask

def correct_colors(target_img, reference_img):
    """
    Adjusts the mean and standard deviation of target_img to match reference_img
    to prevent color drift/bleaching over multiple generations.
    """
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
    import re
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

def get_weighted_embeds(pipe, prompt_text, device, lora_scale_val=None):
    import torch
    if hasattr(pipe, "adjust_lora_scale_for_text_encoder") and lora_scale_val is not None:
        try:
            pipe.adjust_lora_scale_for_text_encoder(pipe.text_encoder, lora_scale_val)
        except Exception:
            pass
            
    bos_token_id = pipe.tokenizer.bos_token_id or 49406
    eos_token_id = pipe.tokenizer.eos_token_id or 49407
    max_length = pipe.tokenizer.model_max_length
    
    parts = parse_prompt_weights(prompt_text)
    
    token_ids = [bos_token_id]
    token_weights = [1.0]
    
    for text, weight in parts:
        ids = pipe.tokenizer(text, add_special_tokens=False).input_ids
        token_ids.extend(ids)
        token_weights.extend([weight] * len(ids))
        
    token_ids.append(eos_token_id)
    token_weights.append(1.0)
    
    if len(token_ids) > max_length:
        token_ids = token_ids[:max_length]
        token_weights = token_weights[:max_length]
    else:
        padding_len = max_length - len(token_ids)
        token_ids.extend([eos_token_id] * padding_len)
        token_weights.extend([1.0] * padding_len)
        
    input_ids_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
    
    with torch.no_grad():
        encoder_outputs = pipe.text_encoder(input_ids_tensor)
        prompt_embeds = encoder_outputs[0]
        
    weights_tensor = torch.tensor([token_weights], dtype=prompt_embeds.dtype, device=device).unsqueeze(-1)
    return prompt_embeds * weights_tensor

def compile_video(output_dir, fps=30, scale=1.0, sharpen=0.0):
    """
    Invokes FFmpeg subprocess to compile generated sequence frames into a video.
    Optionally upscales and sharpens the video during compilation.
    """
    print("\n" + "="*60)
    print("   AUTOMATIC VIDEO COMPILATION // FFMEG ENGINE")
    print("="*60)
    
    if not shutil.which("ffmpeg"):
        print("[!] WARNING: 'ffmpeg' command-line executable was not found in your system path.")
        print("    Please install FFmpeg to compile video automatically, or manually run:")
        print(f"    ffmpeg -framerate {fps} -i {output_dir}/frame%04d.png -c:v libx264 -pix_fmt yuv420p {output_dir}/output.mp4")
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
    
    # Build filtergraph
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
    
    print(f"[*] Compiling frames in '{output_dir}' at {fps} FPS...")
    if scale != 1.0 or sharpen > 0.0:
        print(f"[*] Custom filters applied: scale={scale}x | sharpen={sharpen}")
    print(f"[*] Command: {' '.join(cmd)}")
    
    try:
        # Run FFmpeg, redirecting stdout/stderr to capture errors
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        print(f"[SUCCESS] Video compiled successfully!")
        print(f"[+] Output File: {os.path.abspath(output_video_path)}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[!] ERROR: FFmpeg returned non-zero exit code during compilation.")
        print("    Stderr output from FFmpeg:")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"[!] ERROR: Failed to run FFmpeg command: {e}")
        return False

def parse_prompt_travel(prompt_str):
    """
    Parses keyframe prompts from a multi-line string.
    Expected format per line: 'frame_number: prompt_text'
    If no keyframes can be parsed, returns a single keyframe at frame 0.
    """
    keyframes = []
    lines = prompt_str.strip().split('\n')
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

app = Flask(__name__)

# Directory setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'upload7')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'run7')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

CHECKPOINTS_DIR = "/home/jack/Desktop/Comfy-UI/models/checkpoints"
LORAS_DIR = "/home/jack/Desktop/Comfy-UI/models/loras"
VAE_DIR = "/home/jack/Desktop/Comfy-UI/models/vae"
vae_name="vae-ft-mse-840000-ema-pruned.safetensors"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Global states
active_runs = {}
pipeline_cache = {
    'model_path': None,
    'vae_path': None,
    'device': None,
    'pipe': None
}
pipeline_lock = threading.Lock()


def detect_default_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        log_msg(f"[*] Loading Stable Diffusion model: {model_path} on {device}...")
        from diffusers import StableDiffusionImg2ImgPipeline
        
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
            log_msg("[*] Using built-in VAE (no external VAE selected/found).")
        
        if "LCM" in os.path.basename(model_path).upper() or "LCM" in model_path.upper():
            log_msg("[*] Detected LCM checkpoint. Swapping to LCMScheduler...")
            from diffusers import LCMScheduler
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            
        pipe = pipe.to(device)
        
        # Memory optimizations to prevent system OOM crashes on CPU/GPUs with constrained RAM
        try:
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing("max")
            if hasattr(pipe, "enable_vae_slicing"):
                pipe.enable_vae_slicing()
            if hasattr(pipe, "enable_vae_tiling"):
                pipe.enable_vae_tiling()
            log_msg("[*] Memory optimizations (attention slicing & VAE slicing/tiling) enabled.")
        except Exception as opt_err:
            log_msg(f"[*] Warning: Could not enable pipeline memory optimizations: {opt_err}")

        pipeline_cache['model_path'] = model_path
        pipeline_cache['vae_path'] = vae_path
        pipeline_cache['device'] = device
        pipeline_cache['pipe'] = pipe
        return pipe


def save_run_state(run_output_dir, params, current_frame, status="running"):
    state_path = os.path.join(run_output_dir, "state.json")
    
    # Scan generated images in the output directory
    generated_images = []
    for f in range(current_frame + 1):
        frame_name = f"frame{f:04d}.png"
        frame_path = os.path.join(run_output_dir, frame_name)
        if os.path.exists(frame_path):
            generated_images.append({
                "frame": f,
                "path": f"runs/{os.path.basename(run_output_dir)}/{frame_name}"
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
    if not os.path.exists(OUTPUT_FOLDER):
        return
    for run_id in os.listdir(OUTPUT_FOLDER):
        run_dir = os.path.join(OUTPUT_FOLDER, run_id)
        if not os.path.isdir(run_dir):
            continue
        state_path = os.path.join(run_dir, "state.json")
        if os.path.exists(state_path):
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    state_data = json.load(f)
                
                # Check status
                status = state_data.get("state", {}).get("status", "failed")
                if status == "running" or status == "pending":
                    # If server crashed, a "running" run is now "incomplete"
                    status = "incomplete"
                
                last_frame = state_data.get("state", {}).get("last_completed_frame", -1)
                
                # Self-healing: Sync state with actual contiguous frames on disk if out of sync
                actual_frames = []
                for fname in os.listdir(run_dir):
                    if fname.startswith("frame") and fname.endswith(".png"):
                        try:
                            f_num = int(fname[5:9])
                            actual_frames.append(f_num)
                        except ValueError:
                            pass
                if actual_frames:
                    max_actual = max(actual_frames)
                    is_contiguous = True
                    for i in range(max_actual + 1):
                        if not os.path.exists(os.path.join(run_dir, f"frame{i:04d}.png")):
                            is_contiguous = False
                            break
                    if is_contiguous and last_frame != max_actual:
                        print(f"[*] Contiguous frames on disk ({max_actual}) do not match state.json ({last_frame}). Syncing state to {max_actual}.")
                        last_frame = max_actual
                        # Save corrected state back to disk
                        save_run_state(run_dir, state_data.get("settings", {}), last_frame, status)
                
                total_frames = state_data.get("settings", {}).get("frames", 90)
                
                # Get last image path
                last_image = None
                if last_frame >= 0:
                    last_image = f"/runs/{run_id}/frame{last_frame:04d}.png"
                
                video_path = None
                if os.path.exists(os.path.join(run_dir, "output.mp4")):
                    video_path = f"/runs/{run_id}/output.mp4"
                    status = "success"
                
                logs = []
                log_path = os.path.join(run_dir, "log.txt")
                if os.path.exists(log_path):
                    try:
                        with open(log_path, "r", encoding="utf-8") as f:
                            logs = [line.rstrip("\n") for line in f.readlines()]
                    except Exception as e:
                        print(f"[!] Error reading log from disk for {run_id}: {e}")
                
                if not logs:
                    logs = [f"[System] Loaded state from disk. Status: {status}."]
                else:
                    logs.append(f"[System] Loaded state from disk. Resuming session. Status: {status}.")
                
                active_runs[run_id] = {
                    'status': status,
                    'current_frame': last_frame,
                    'total_frames': total_frames,
                    'current_step': 0,
                    'total_steps': 0,
                    'logs': logs,
                    'last_image': last_image,
                    'video_path': video_path,
                    'error': None if status != "failed" else "Run failed or was interrupted.",
                    'params': state_data.get("settings", {}),
                    'timestamp': datetime.fromtimestamp(os.path.getmtime(state_path)).strftime('%Y-%m-%d %H:%M:%S')
                }
            except Exception as e:
                print(f"[!] Error loading state for {run_id}: {e}")


def log_to_run(run_id, message):
    timestamp = datetime.now().strftime('%H:%M:%S')
    log_line = f"[{timestamp}] {message}"
    print(f"[Run {run_id}] {log_line}")
    if run_id in active_runs:
        active_runs[run_id]['logs'].append(log_line)
        
        # Append to log file on disk if the run folder exists
        run_output_dir = os.path.join(OUTPUT_FOLDER, run_id)
        if os.path.exists(run_output_dir):
            log_path = os.path.join(run_output_dir, "log.txt")
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(log_line + "\n")
            except Exception as e:
                print(f"[!] Error writing log to disk for run {run_id}: {e}")


def run_generation_task(run_id, params):
    run_output_dir = os.path.join(OUTPUT_FOLDER, run_id)
    os.makedirs(run_output_dir, exist_ok=True)
    active_runs[run_id]['output_dir'] = run_output_dir
    active_runs[run_id]['status'] = 'running'

    # Save existing memory logs to disk now that folder exists
    log_path = os.path.join(run_output_dir, "log.txt")
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            for log in active_runs[run_id]['logs']:
                f.write(log + "\n")
    except Exception as e:
        print(f"[!] Error initializing log file: {e}")

    try:
        log_to_run(run_id, "Checking run state for resume...")
        
        # Extract params
        input_image_path = params['input_image']
        mask_image_path = params.get('mask_image')
        prompt = params['prompt']
        negative = params['negative']
        width = int(params['width'])
        height = int(params['height'])
        shift = int(params['shift'])
        blur_width = int(params['blur_width'])
        blur_radius = int(params['blur_radius'])
        frames = int(params['frames'])
        steps = int(params['steps'])
        strength = float(params['strength'])
        guidance = float(params['guidance'])
        device = params['device']
        model = params['model']
        seed = int(params['seed'])
        keep_original = params.get('keep_original', False)
        color_correct = params.get('color_correct', False)
        
        fps = int(params.get('fps', 30))
        scale = float(params.get('scale', 1.0))
        sharpen = float(params.get('sharpen', 0.0))
        
        lora_name = params.get('lora_name', 'None')
        lora_scale = float(params.get('lora_scale', 0.8))
        vae_name = params.get('vae_name', 'None')
        direction = params.get('direction', 'right')

        # Check if state already exists on disk to resume
        state_path = os.path.join(run_output_dir, "state.json")
        last_frame = -1
        if os.path.exists(state_path):
            try:
                with open(state_path, "r", encoding="utf-8") as f:
                    state_data = json.load(f)
                last_frame = state_data.get("state", {}).get("last_completed_frame", -1)
            except Exception as e:
                log_to_run(run_id, f"Warning: Failed to parse existing state.json: {e}")

        # Check if there is an actual frame image matching last_frame
        if last_frame >= 0:
            last_frame_path = os.path.join(run_output_dir, f"frame{last_frame:04d}.png")
            if os.path.exists(last_frame_path):
                log_to_run(run_id, f"Resuming from frame {last_frame:04d} (frame {last_frame + 1} of {frames - 1})...")
                current = Image.open(last_frame_path).convert("RGB")
                current = current.resize((width, height), resample=LANCZOS)
                start_frame_idx = last_frame + 1
            else:
                log_to_run(run_id, f"Warning: frame{last_frame:04d}.png not found on disk. Re-compiling from scratch.")
                last_frame = -1
                start_frame_idx = 0
        else:
            start_frame_idx = 0

        # Check model file exists
        if not os.path.exists(model):
            log_to_run(run_id, f"WARNING: Model file '{model}' not found. Make sure path is valid.")

        # Initialize pipeline
        log_to_run(run_id, "Initializing Stable Diffusion pipeline...")
        pipe = get_pipeline(model, vae_name, device, run_id)
        
        # Reset any previous LoRAs
        try:
            log_to_run(run_id, "Resetting previously active LoRAs...")
            pipe.unload_lora_weights()
        except Exception as e:
            pass

        # Load LoRAs if specified
        cross_kwargs = None
        if lora_name and lora_name != "None":
            lora_path = os.path.join(LORAS_DIR, lora_name)
            if os.path.exists(lora_path):
                log_to_run(run_id, f"Loading LoRA weights: {lora_name}...")
                pipe.load_lora_weights(lora_path)
                cross_kwargs = {"scale": lora_scale}
                log_to_run(run_id, f"LoRA loaded. Applying scale: {lora_scale}")
            else:
                log_to_run(run_id, f"WARNING: LoRA file not found at: {lora_path}")

        generator = torch.Generator(device=device).manual_seed(seed)

        # Caching text embeddings to avoid redundant CLIP calls
        embeds_cache = {}
        def get_embeds(prompt_text, lora_scale_val=None):
            run_params = active_runs[run_id].get('params', params)
            neg_prompt = run_params.get('negative', negative)
            cache_key = (prompt_text, neg_prompt, lora_scale_val)
            if cache_key not in embeds_cache:
                log_to_run(run_id, f"Encoding text conditioning for: '{prompt_text}...'")
                p_emb = get_weighted_embeds(pipe, prompt_text, device, lora_scale_val)
                n_emb = get_weighted_embeds(pipe, neg_prompt, device, lora_scale_val)
                embeds_cache[cache_key] = (p_emb, n_emb)
            return embeds_cache[cache_key]

        # Step count callback for real-time progress overlay
        def step_callback(step, timestep, latents):
            if run_id not in active_runs:
                return

            active_runs[run_id]['current_step'] = step
            try:
                total_steps = len(pipe.scheduler.timesteps)
                active_runs[run_id]['total_steps'] = total_steps
            except:
                run_params = active_runs[run_id].get('params', params)
                current_steps = int(run_params.get('steps', steps))
                total_steps = int(current_steps * 0.38) # fallback estimation
                active_runs[run_id]['total_steps'] = total_steps

            # Decode latents to preview image only on CUDA to save memory and avoid OOM crash on CPU
            if "cuda" in device and (step % 5 == 0 or step == total_steps - 1):
                try:
                    import torch
                    import numpy as np
                    with torch.no_grad():
                        scaling_factor = getattr(pipe.vae.config, "scaling_factor", 0.18215)
                        latents_scaled = latents / scaling_factor
                        
                        # VAE Decode
                        sample_tensor = pipe.vae.decode(latents_scaled).sample
                        sample_tensor = (sample_tensor / 2 + 0.5).clamp(0, 1)
                        sample_tensor = sample_tensor.cpu().permute(0, 2, 3, 1).float()
                        image_numpy = sample_tensor.numpy()
                        image_numpy = (image_numpy * 255).round().astype(np.uint8)
                        
                        preview_img = Image.fromarray(image_numpy[0])
                        preview_path = os.path.join(run_output_dir, "preview.png")
                        preview_img.save(preview_path)
                        active_runs[run_id]['last_image'] = f"/runs/{run_id}/preview.png"
                        
                        # Explicit cleanup inside the callback
                        del latents_scaled, sample_tensor, image_numpy, preview_img
                        import gc
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                except Exception as e:
                    # Ignore decoding errors to avoid halting generation
                    pass

        # Perform generation inside a torch.no_grad() context to prevent memory leak via autograd graphs
        with torch.no_grad():
            # Build Frame 0 if not resuming
            if start_frame_idx == 0:
                log_to_run(run_id, f"Loading input image: {input_image_path}")
                source = Image.open(input_image_path).convert("RGB")
                source = source.resize((width, height), resample=LANCZOS)

                frame0_path = os.path.join(run_output_dir, "frame0000.png")
                if keep_original and not mask_image_path:
                    log_to_run(run_id, "Using original image as Frame 0000...")
                    current = source
                else:
                    log_to_run(run_id, "Compiling Frame 0000 (Base latents)...")
                    # Read latest params for Frame 0 compilation
                    run_params = active_runs[run_id].get('params', params)
                    current_prompt = run_params.get('prompt', prompt)
                    current_steps = int(run_params.get('steps', steps))
                    current_guidance = float(run_params.get('guidance', guidance))
                    current_lora_scale = float(run_params.get('lora_scale', lora_scale))
                    current_strength = float(run_params.get('strength', strength))

                    # Handle prompt traveling
                    keyframes = parse_prompt_travel(current_prompt)
                    log_to_run(run_id, f"Keyframes Parsed: {keyframes}")
                    inject_active = run_params.get('inject_prompt', False)
                    injected_text = run_params.get('injected_prompt', '')
                    if inject_active and injected_text:
                        for kf in keyframes:
                            kf['prompt'] = kf['prompt'] + ", " + injected_text
                    p_emb, n_emb = get_embeds(keyframes[0]['prompt'], current_lora_scale)

                    # Use current_strength if we are using masking, otherwise 0.35
                    run_strength = current_strength if mask_image_path else 0.35

                    generated_img = pipe(
                        prompt_embeds=p_emb,
                        negative_prompt_embeds=n_emb,
                        image=source,
                        strength=run_strength,
                        guidance_scale=current_guidance,
                        num_inference_steps=current_steps,
                        generator=generator,
                        cross_attention_kwargs=cross_kwargs,
                        callback=step_callback,
                        callback_steps=1
                    ).images[0]
                    generated_img = generated_img.resize((width, height), resample=LANCZOS)

                    if mask_image_path and os.path.exists(mask_image_path):
                        log_to_run(run_id, "Applying interactive mask to Frame 0...")
                        mask = Image.open(mask_image_path).convert("L")
                        mask = mask.resize((width, height), resample=LANCZOS)
                        current = Image.composite(generated_img, source, mask)
                    else:
                        current = generated_img

                current.save(frame0_path)
                active_runs[run_id]['current_frame'] = 0
                active_runs[run_id]['last_image'] = f"/runs/{run_id}/frame0000.png"
                save_run_state(run_output_dir, active_runs[run_id]['params'], 0, "running")
                log_to_run(run_id, "Frame 0000 saved.")
                start_frame_idx = 1
                
                # Cleanup Frame 0 variables
                if 'source' in locals():
                    del source
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Generation loop
            for frame in range(start_frame_idx, frames):
                if active_runs[run_id].get('cancelled', False):
                    log_to_run(run_id, "Run cancelled by user.")
                    active_runs[run_id]['status'] = 'cancelled'
                    save_run_state(run_output_dir, active_runs[run_id]['params'], frame - 1, "cancelled")
                    return

                log_to_run(run_id, f"Processing Frame {frame:04d} of {frames - 1}...")
                
                # Read parameters dynamically from active_runs in case they were updated!
                run_params = active_runs[run_id].get('params', params)
                current_prompt = run_params.get('prompt', prompt)
                current_steps = int(run_params.get('steps', steps))
                current_strength = float(run_params.get('strength', strength))
                current_guidance = float(run_params.get('guidance', guidance))
                current_lora_scale = float(run_params.get('lora_scale', lora_scale))
                direction = run_params.get('direction', direction)
                
                # Recalculate cross_kwargs scale if dynamically modified
                current_cross_kwargs = None
                if lora_name and lora_name != "None":
                    current_cross_kwargs = {"scale": current_lora_scale}
                
                # Interpolate prompt embeddings dynamically for prompt traveling
                keyframes = parse_prompt_travel(current_prompt)
                inject_active = run_params.get('inject_prompt', False)
                injected_text = run_params.get('injected_prompt', '')
                if inject_active and injected_text:
                    for kf in keyframes:
                        kf['prompt'] = kf['prompt'] + ", " + injected_text
                
                if len(keyframes) == 1:
                    current_prompt_embeds, current_negative_prompt_embeds = get_embeds(keyframes[0]['prompt'], current_lora_scale)
                    log_to_run(run_id, f"Active prompt: ({keyframes[0]['prompt']},1.00)")
                else:
                    if frame <= keyframes[0]['frame']:
                        current_prompt_embeds, current_negative_prompt_embeds = get_embeds(keyframes[0]['prompt'], current_lora_scale)
                        log_to_run(run_id, f"Active prompt: ({keyframes[0]['prompt']},1.00)")
                    elif frame >= keyframes[-1]['frame']:
                        current_prompt_embeds, current_negative_prompt_embeds = get_embeds(keyframes[-1]['prompt'], current_lora_scale)
                        log_to_run(run_id, f"Active prompt: ({keyframes[-1]['prompt']},1.00)")
                    else:
                        start_kf = None
                        end_kf = None
                        for i in range(len(keyframes) - 1):
                            if keyframes[i]['frame'] <= frame <= keyframes[i+1]['frame']:
                                start_kf = keyframes[i]
                                end_kf = keyframes[i+1]
                                break
                        # Interpolate
                        w = (frame - start_kf['frame']) / (end_kf['frame'] - start_kf['frame'])
                        p_start, n_start = get_embeds(start_kf['prompt'], current_lora_scale)
                        p_end, n_end = get_embeds(end_kf['prompt'], current_lora_scale)
                        current_prompt_embeds = (1.0 - w) * p_start + w * p_end
                        current_negative_prompt_embeds = (1.0 - w) * n_start + w * n_end
                        
                        w_start = 1.0 - w
                        w_end = w
                        w_start_str = f"{w_start:.2f}".replace("0.", ".") if w_start < 1.0 else "1.00"
                        w_end_str = f"{w_end:.2f}".replace("0.", ".") if w_end < 1.0 else "1.00"
                        log_to_run(run_id, f"Interpolation: ({start_kf['prompt']},{w_start_str}),({end_kf['prompt']},{w_end_str})")
                # Setup canvas, stretch padding, blur edge, and composite based on direction
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
                    canvas_width = width
                    canvas_height = height
                    
                    canvas = Image.new("RGB", (canvas_width, canvas_height))
                    
                    shrunk_width = width - 2 * shift
                    shrunk_width = max(8, (shrunk_width // 2) * 2)
                    shrunk_height = int(round(shrunk_width * height / width))
                    shrunk_height = max(8, (shrunk_height // 2) * 2)
                    
                    shrunk = current.resize((shrunk_width, shrunk_height), resample=LANCZOS)
                    
                    x_offset = (width - shrunk_width) // 2
                    y_offset = (height - shrunk_height) // 2
                    
                    canvas.paste(shrunk, (x_offset, y_offset))
                    
                    # Pad edges
                    # Top pad
                    top_row = shrunk.crop((0, 0, shrunk_width, 1))
                    top_pad = top_row.resize((shrunk_width, y_offset), resample=LANCZOS)
                    canvas.paste(top_pad, (x_offset, 0))
                    
                    # Bottom pad
                    bottom_row = shrunk.crop((0, shrunk_height - 1, shrunk_width, shrunk_height))
                    bottom_pad = bottom_row.resize((shrunk_width, height - (y_offset + shrunk_height)), resample=LANCZOS)
                    canvas.paste(bottom_pad, (x_offset, y_offset + shrunk_height))
                    
                    # Left pad (including top/bottom pads)
                    left_col = canvas.crop((x_offset, 0, x_offset + 1, height))
                    left_pad = left_col.resize((x_offset, height), resample=LANCZOS)
                    canvas.paste(left_pad, (0, 0))
                    
                    # Right pad (including top/bottom pads)
                    right_col = canvas.crop((x_offset + shrunk_width - 1, 0, x_offset + shrunk_width, height))
                    right_pad = right_col.resize((width - (x_offset + shrunk_width), height), resample=LANCZOS)
                    canvas.paste(right_pad, (x_offset + shrunk_width, 0))
                    
                    blurred_canvas = canvas.filter(ImageFilter.GaussianBlur(blur_radius))
                    
                    mask = create_feathered_mask(canvas_width, canvas_height, direction, x_offset, y_offset, blur_radius)
                    blended_canvas = Image.composite(canvas, blurred_canvas, mask)
                    
                elif direction == 'in':
                    canvas_width = width
                    canvas_height = height
                    
                    crop_width = width - 2 * shift
                    crop_width = max(8, (crop_width // 2) * 2)
                    crop_height = int(round(crop_width * height / width))
                    crop_height = max(8, (crop_height // 2) * 2)
                    
                    x_offset = (width - crop_width) // 2
                    y_offset = (height - crop_height) // 2
                    
                    cropped = current.crop((x_offset, y_offset, x_offset + crop_width, y_offset + crop_height))
                    canvas = cropped.resize((width, height), resample=LANCZOS)
                    blended_canvas = canvas
                elif direction == 'static':
                    canvas_width = width
                    canvas_height = height
                    blended_canvas = current
                
                # Prevent "tensor of 0 elements" VAE decode crash due to too few effective steps
                eff_steps = int(current_steps * current_strength)
                if eff_steps < 2:
                    adjusted_strength = min(1.0, max(current_strength, 2.0 / current_steps))
                    log_to_run(run_id, f"[System] Effective steps ({eff_steps}) is too low for scheduler. Adjusting strength from {current_strength} to {adjusted_strength:.2f} to prevent pipeline crash.")
                    current_strength = adjusted_strength

                # Outpaint
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
                
                expanded = expanded.resize((canvas_width, canvas_height), resample=LANCZOS)
                
                if direction == 'right':
                    current = expanded.crop((shift, 0, shift + width, height))
                elif direction == 'left':
                    current = expanded.crop((0, 0, width, height))
                elif direction == 'down':
                    current = expanded.crop((0, shift, width, shift + height))
                elif direction == 'up':
                    current = expanded.crop((0, 0, width, height))
                elif direction == 'out':
                    current = expanded
                elif direction == 'in':
                    current = expanded
                elif direction == 'static':
                    current = expanded
                
                # Apply color correction if requested to prevent desaturation drift
                if color_correct and frame > 0:
                    frame0_path = os.path.join(run_output_dir, "frame0000.png")
                    if os.path.exists(frame0_path):
                        try:
                            current = correct_colors(current, Image.open(frame0_path).convert("RGB"))
                        except Exception as e:
                            log_to_run(run_id, f"Warning: Color correction failed: {e}")

                frame_path = os.path.join(run_output_dir, f"frame{frame:04d}.png")
                current.save(frame_path)
                
                # Update state
                active_runs[run_id]['current_frame'] = frame
                active_runs[run_id]['last_image'] = f"/runs/{run_id}/frame{frame:04d}.png"
                save_run_state(run_output_dir, active_runs[run_id]['params'], frame, "running")
                log_to_run(run_id, f"Frame {frame:04d} saved.")

                # Cleanup frame-specific variables to prevent memory build-up
                try: del blended_canvas
                except NameError: pass
                try: del canvas
                except NameError: pass
                try: del blurred_canvas
                except NameError: pass
                try: del mask
                except NameError: pass
                try: del edge
                except NameError: pass
                try: del gap_padding
                except NameError: pass
                try: del last_col
                except NameError: pass
                try: del first_col
                except NameError: pass
                try: del last_row
                except NameError: pass
                try: del first_row
                except NameError: pass
                try: del shrunk
                except NameError: pass
                try: del top_row
                except NameError: pass
                try: del top_pad
                except NameError: pass
                try: del bottom_row
                except NameError: pass
                try: del bottom_pad
                except NameError: pass
                try: del left_col
                except NameError: pass
                try: del left_pad
                except NameError: pass
                try: del right_col
                except NameError: pass
                try: del right_pad
                except NameError: pass
                try: del expanded
                except NameError: pass
                try: del cropped
                except NameError: pass
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Compile video
        log_to_run(run_id, "Compiling video using FFmpeg...")
        # Read latest video compilation params
        run_params = active_runs[run_id].get('params', params)
        current_fps = int(run_params.get('fps', fps))
        current_scale = float(run_params.get('scale', scale))
        current_sharpen = float(run_params.get('sharpen', sharpen))

        video_success = compile_video(run_output_dir, current_fps, current_scale, current_sharpen)
        
        if video_success or os.path.exists(os.path.join(run_output_dir, "output.mp4")):
            active_runs[run_id]['status'] = 'success'
            active_runs[run_id]['video_path'] = f"/runs/{run_id}/output.mp4"
            save_run_state(run_output_dir, active_runs[run_id]['params'], frames - 1, "success")
            log_to_run(run_id, "SUCCESS! Generation and compilation completed.")
        else:
            active_runs[run_id]['status'] = 'failed'
            active_runs[run_id]['error'] = "Video compilation failed (FFmpeg issue)."
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
        except:
            pass


@app.route('/')
def index():
    default_dev = detect_default_device()
    default_model = "/home/jack/Desktop/Comfy-UI/models/checkpoints/dreamshaper_8LCM.safetensors"
    return render_template('index0.html', default_device=default_dev, default_model=default_model)


@app.route('/api/models')
def get_models():
    models = []
    if os.path.exists(CHECKPOINTS_DIR):
        for f in os.listdir(CHECKPOINTS_DIR):
            if f.endswith(('.safetensors', '.ckpt')):
                models.append(f)
    models.sort()
    return jsonify({'success': True, 'models': models})


@app.route('/api/loras')
def get_loras():
    loras = []
    if os.path.exists(LORAS_DIR):
        for root, dirs, files in os.walk(LORAS_DIR):
            for f in files:
                if f.endswith(('.safetensors', '.ckpt')):
                    rel_path = os.path.relpath(os.path.join(root, f), LORAS_DIR)
                    loras.append(rel_path)
    loras.sort()
    if "None" in loras:
        loras.remove("None")
    loras.insert(0, "None")
    return jsonify({'success': True, 'loras': loras})


@app.route('/api/vaes')
def get_vaes():
    vaes = []
    if os.path.exists(VAE_DIR):
        for root, dirs, files in os.walk(VAE_DIR):
            for f in files:
                if f.endswith(('.safetensors', '.ckpt', '.pt')):
                    rel_path = os.path.relpath(os.path.join(root, f), VAE_DIR)
                    vaes.append(rel_path)
    vaes.sort()
    if "None" in vaes:
        vaes.remove("None")
    vaes.insert(0, "None")
    return jsonify({'success': True, 'vaes': vaes})


@app.route('/api/start', methods=['POST'])
def start_generation():
    def get_form_int(key, default):
        val = request.form.get(key)
        if val is None or str(val).strip() == '':
            return default
        try:
            return int(val)
        except ValueError:
            return default

    def get_form_float(key, default):
        val = request.form.get(key)
        if val is None or str(val).strip() == '':
            return default
        try:
            return float(val)
        except ValueError:
            return default

    image_file = request.files.get('input_image_file')
    input_image_path = None
    
    if image_file and image_file.filename != '':
        filename = f"{str(uuid.uuid4())[:8]}_{image_file.filename}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(save_path)
        input_image_path = save_path
    else:
        input_image_path = request.form.get('input_image_path')
        if not input_image_path or not os.path.exists(input_image_path):
            return jsonify({'success': False, 'error': 'Invalid or missing input image.'}), 400

    mask_file = request.files.get('mask_image_file')
    mask_image_path = None
    if mask_file and mask_file.filename != '':
        mask_filename = f"mask_{str(uuid.uuid4())[:8]}.png"
        mask_save_path = os.path.join(app.config['UPLOAD_FOLDER'], mask_filename)
        mask_file.save(mask_save_path)
        mask_image_path = mask_save_path

    # Build params dict
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
        'device': request.form.get('device', 'cpu'),
        'model': request.form.get('model', ''),
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

    # Initialize active run structure
    active_runs[run_id] = {
        'status': 'pending',
        'current_frame': 0,
        'total_frames': params['frames'],
        'current_step': 0,
        'total_steps': 0,
        'logs': [f"[System] Creating new run folder runs/{run_id}"],
        'last_image': None,
        'video_path': None,
        'error': None,
        'params': params,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Start thread
    thread = threading.Thread(target=run_generation_task, args=(run_id, params))
    thread.daemon = True
    thread.start()

    return jsonify({'success': True, 'run_id': run_id})


@app.route('/api/resume/<run_id>', methods=['POST'])
def resume_generation(run_id):
    if run_id not in active_runs:
        load_all_runs_from_disk()
        if run_id not in active_runs:
            return jsonify({'success': False, 'error': 'Run ID not found.'}), 404
            
    run = active_runs[run_id]
    params = run['params']
    
    # Reset status in active runs
    active_runs[run_id]['status'] = 'pending'
    active_runs[run_id]['cancelled'] = False
    active_runs[run_id]['error'] = None
    active_runs[run_id]['current_step'] = 0
    active_runs[run_id]['total_steps'] = 0
    active_runs[run_id]['logs'].append(f"[System] Resuming run from the last completed frame...")
    
    # Start thread
    thread = threading.Thread(target=run_generation_task, args=(run_id, params))
    thread.daemon = True
    thread.start()
    
    return jsonify({'success': True, 'run_id': run_id})


@app.route('/api/delete/<run_id>', methods=['POST'])
def delete_run(run_id):
    if run_id in active_runs:
        del active_runs[run_id]
    
    run_dir = os.path.join(OUTPUT_FOLDER, run_id)
    if os.path.exists(run_dir):
        try:
            shutil.rmtree(run_dir)
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'success': False, 'error': f"Failed to delete directory: {e}"}), 500
    
    return jsonify({'success': True})


@app.route('/api/status/<run_id>')
def get_status(run_id):
    if run_id not in active_runs:
        return jsonify({'success': False, 'error': 'Run ID not found.'}), 404
    
    run = active_runs[run_id]
    
    # Scan output folder dynamically for generated images to keep list accurate
    run_output_dir = os.path.join(OUTPUT_FOLDER, run_id)
    generated_images = []
    if os.path.exists(run_output_dir):
        for f in range(run['total_frames']):
            frame_name = f"frame{f:04d}.png"
            frame_path = os.path.join(run_output_dir, frame_name)
            if os.path.exists(frame_path):
                generated_images.append({
                    "frame": f,
                    "path": f"runs/{run_id}/{frame_name}"
                })
                
    return jsonify({
        'success': True,
        'status': run['status'],
        'current_frame': run['current_frame'],
        'total_frames': run['total_frames'],
        'current_step': run.get('current_step', 0),
        'total_steps': run.get('total_steps', 0),
        'last_image': run['last_image'],
        'video_path': run['video_path'],
        'error': run['error'],
        'log_count': len(run['logs']),
        'params': run.get('params', {}),
        'generated_images': generated_images
    })


@app.route('/api/delete_frames/<run_id>/<int:frame_idx>', methods=['POST'])
def delete_frames(run_id, frame_idx):
    if run_id not in active_runs:
        return jsonify({'success': False, 'error': 'Run ID not found.'}), 404
        
    run = active_runs[run_id]
    if run['status'] == 'running':
        return jsonify({'success': False, 'error': 'Cannot delete frames while generation is active. Cancel first.'}), 400
        
    run_output_dir = os.path.join(OUTPUT_FOLDER, run_id)
    if not os.path.exists(run_output_dir):
        return jsonify({'success': False, 'error': 'Output directory not found.'}), 404
        
    try:
        # Delete frames starting from frame_idx
        deleted_count = 0
        total_frames = run['total_frames']
        for f in range(frame_idx, total_frames):
            frame_name = f"frame{f:04d}.png"
            frame_path = os.path.join(run_output_dir, frame_name)
            if os.path.exists(frame_path):
                os.remove(frame_path)
                deleted_count += 1
                
        # Remove output video if it exists since it's now out of date
        video_path = os.path.join(run_output_dir, "output.mp4")
        if os.path.exists(video_path):
            os.remove(video_path)
            
        # Remove preview.png if it exists
        preview_path = os.path.join(run_output_dir, "preview.png")
        if os.path.exists(preview_path):
            os.remove(preview_path)
            
        # Update last completed frame
        new_last_completed = frame_idx - 1
        run['current_frame'] = new_last_completed
        run['status'] = 'incomplete'
        run['video_path'] = None
        
        # Get last valid image path
        if new_last_completed >= 0:
            run['last_image'] = f"/runs/{run_id}/frame{new_last_completed:04d}.png"
        else:
            run['last_image'] = None
            
        # Save run state to disk
        save_run_state(run_output_dir, run['params'], new_last_completed, "incomplete")
        
        log_to_run(run_id, f"Deleted frames starting from frame {frame_idx:04d}. Reset last frame to {new_last_completed:04d}.")
        
        return jsonify({
            'success': True,
            'deleted_count': deleted_count,
            'new_last_completed': new_last_completed
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/logs/<run_id>')
def get_logs(run_id):
    if run_id not in active_runs:
        return jsonify({'success': False, 'error': 'Run ID not found.'}), 404
    
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
        return jsonify({'success': False, 'error': 'Run ID not found.'}), 404
    
    active_runs[run_id]['cancelled'] = True
    return jsonify({'success': True})


@app.route('/api/update/<run_id>', methods=['POST'])
def update_run_params(run_id):
    if run_id not in active_runs:
        return jsonify({'success': False, 'error': 'Run ID not found.'}), 404
    
    run = active_runs[run_id]
    params = run['params']
    data = request.json if request.is_json else request.form
    
    try:
        def get_val_int(key, default):
            val = data.get(key)
            if val is None or str(val).strip() == '':
                return default
            try:
                return int(val)
            except ValueError:
                return default

        def get_val_float(key, default):
            val = data.get(key)
            if val is None or str(val).strip() == '':
                return default
            try:
                return float(val)
            except ValueError:
                return default

        if 'prompt' in data:
            params['prompt'] = data.get('prompt')
        if 'negative' in data:
            params['negative'] = data.get('negative')
        if 'steps' in data:
            params['steps'] = get_val_int('steps', params.get('steps', 25))
        if 'strength' in data:
            params['strength'] = get_val_float('strength', params.get('strength', 0.38))
        if 'guidance' in data:
            params['guidance'] = get_val_float('guidance', params.get('guidance', 7.0))
        if 'lora_scale' in data:
            params['lora_scale'] = get_val_float('lora_scale', params.get('lora_scale', 0.8))
        if 'fps' in data:
            params['fps'] = get_val_int('fps', params.get('fps', 30))
        if 'scale' in data:
            params['scale'] = get_val_float('scale', params.get('scale', 1.0))
        if 'sharpen' in data:
            params['sharpen'] = get_val_float('sharpen', params.get('sharpen', 0.0))
        if 'direction' in data:
            params['direction'] = data.get('direction')
        if 'frames' in data:
            params['frames'] = get_val_int('frames', params.get('frames', 90))
            run['total_frames'] = params['frames']
        if 'shift' in data:
            params['shift'] = get_val_int('shift', params.get('shift', 4))
        if 'blur_width' in data:
            params['blur_width'] = get_val_int('blur_width', params.get('blur_width', 8))
        if 'blur_radius' in data:
            params['blur_radius'] = get_val_int('blur_radius', params.get('blur_radius', 8))
        if 'seed' in data:
            params['seed'] = get_val_int('seed', params.get('seed', 0))
        if 'model' in data:
            params['model'] = data.get('model')
        if 'lora_name' in data:
            params['lora_name'] = data.get('lora_name')
        if 'vae_name' in data:
            params['vae_name'] = data.get('vae_name')
        if 'color_correct' in data:
            val = data.get('color_correct')
            params['color_correct'] = str(val).lower() == 'true' or val is True
        if 'inject_prompt' in data:
            val = data.get('inject_prompt')
            params['inject_prompt'] = str(val).lower() == 'true' or val is True
        if 'injected_prompt' in data:
            params['injected_prompt'] = data.get('injected_prompt')
            
        # Save updated state to disk
        run_output_dir = run.get('output_dir') or os.path.join(OUTPUT_FOLDER, run_id)
        if os.path.exists(run_output_dir):
            save_run_state(run_output_dir, params, run.get('current_frame', 0), run.get('status', 'running'))
            
        log_to_run(run_id, f"Parameters updated dynamically: {data.to_dict() if hasattr(data, 'to_dict') else data}")
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/inpaint_frame/<run_id>/<int:frame_idx>', methods=['POST'])
def inpaint_frame(run_id, frame_idx):
    if run_id not in active_runs:
        load_all_runs_from_disk()
        if run_id not in active_runs:
            return jsonify({'success': False, 'error': 'Run ID not found.'}), 404
            
    run = active_runs[run_id]
    params = run['params']
    data = request.json if request.is_json else request.form
    
    if 'mask_image_file' not in request.files:
        return jsonify({'success': False, 'error': 'No mask file provided.'}), 400
        
    mask_file = request.files['mask_image_file']
    
    try:
        run_output_dir = run.get('output_dir') or os.path.join(OUTPUT_FOLDER, run_id)
        frame_name = f"frame{frame_idx:04d}.png"
        frame_path = os.path.join(run_output_dir, frame_name)
        
        if not os.path.exists(frame_path):
            return jsonify({'success': False, 'error': f'Frame #{frame_idx} file not found.'}), 404
            
        log_to_run(run_id, f"[Inpaint] Starting inpaint repair on frame #{frame_idx}...")
        
        # Load and convert mask to L (grayscale)
        mask_image = Image.open(mask_file.stream).convert("L")
        
        # Load original frame
        init_image = Image.open(frame_path).convert("RGB")
        
        # Get active model parameters
        checkpoint_path = params.get('model', "/home/jack/Desktop/Comfy-UI/models/checkpoints/dreamshaper_8LCM.safetensors")
        lora_name = params.get('lora_name', 'None')
        lora_scale = float(params.get('lora_scale', 0.8))
        vae_name = params.get('vae_name', 'None')
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Get the cached pipeline
        pipe = get_pipeline(checkpoint_path, vae_name, device, run_id)
        
        # Reset and load LoRA weights
        if hasattr(pipe, "unload_lora_weights"):
            pipe.unload_lora_weights()
            
        cross_kwargs = {}
        if lora_name and lora_name != "None":
            lora_path = os.path.join(LORAS_DIR, lora_name)
            if os.path.exists(lora_path):
                log_to_run(run_id, f"[Inpaint] Loading LoRA weights: {lora_name}...")
                pipe.load_lora_weights(lora_path)
                cross_kwargs = {"scale": lora_scale}
            else:
                log_to_run(run_id, f"[Inpaint] WARNING: LoRA file not found at: {lora_path}")
                
        # Parse inference parameters
        prompt = data.get('prompt', params.get('prompt', ''))
        negative = data.get('negative', params.get('negative', ''))
        steps = int(data.get('steps', params.get('steps', 6)))
        strength = float(data.get('strength', params.get('strength', 0.35)))
        guidance = float(data.get('guidance', params.get('guidance', 1.5)))
        
        # Setup generator
        seed = int(params.get('seed', -1))
        if seed == -1:
            generator = None
        else:
            generator = torch.Generator(device=device).manual_seed(seed)
            
        log_to_run(run_id, f"[Inpaint] Running img2img pipeline (steps={steps}, strength={strength}, guidance={guidance})...")
        
        # Prevent "tensor of 0 elements" VAE decode crash due to too few effective steps
        eff_steps = int(steps * strength)
        if eff_steps < 2:
            adjusted_strength = min(1.0, max(strength, 2.0 / steps))
            log_to_run(run_id, f"[Inpaint] Effective steps ({eff_steps}) is too low for scheduler. Adjusting strength from {strength} to {adjusted_strength:.2f} to prevent pipeline crash.")
            strength = adjusted_strength

        # Get weighted embeddings for inpainting
        p_emb = get_weighted_embeds(pipe, prompt, device, lora_scale)
        n_emb = get_weighted_embeds(pipe, negative, device, lora_scale)

        # Run img2img pipeline
        generated = pipe(
            prompt_embeds=p_emb,
            negative_prompt_embeds=n_emb,
            image=init_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            cross_attention_kwargs=cross_kwargs if cross_kwargs else None
        ).images[0]
        
        # Feather mask (blur)
        blur_radius = int(params.get('blur_radius', 16))
        blurred_mask = mask_image.filter(ImageFilter.GaussianBlur(blur_radius))
        
        # Composite generated and original frame using the feathered mask
        inpainted_image = Image.composite(generated, init_image, blurred_mask)
        
        # Save output image
        inpainted_image.save(frame_path)
        log_to_run(run_id, f"[Inpaint] Inpainted image saved to {frame_name}.")
        
        # Re-compile video
        log_to_run(run_id, "[Inpaint] Re-compiling output.mp4...")
        fps = int(params.get('fps', 30))
        scale = float(params.get('scale', 1.0))
        sharpen = float(params.get('sharpen', 0.0))
        compile_video(run_output_dir, fps, scale, sharpen)
        
        log_to_run(run_id, f"[Inpaint] SUCCESS: Frame #{frame_idx} repair finished successfully!")
        return jsonify({'success': True})
        
    except Exception as e:
        traceback.print_exc()
        log_to_run(run_id, f"[Inpaint] ERROR: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


DRAFT_PATH = os.path.join(BASE_DIR, "draft.json")

def get_default_draft():
    # Return default parameters
    return {
        'prompt': '0:image another dimension on an alien planet, a tropical jungle is slightly visible. Cinematic lighting, photorealistic.\n50:create a stone path leading into the alien tropical jungle, foliage and tropical flowers,  cinematic lighting, photorealistic.',
        'negative': 'text, watermark, blurry, low quality, bad anatomy, deformed, muted colors, washed out, desaturated',
        'model': "/home/jack/Desktop/Comfy-UI/models/checkpoints/dreamshaper_8LCM.safetensors",
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
        'steps': 25,
        'guidance': 7.0,
        'strength': 0.30,
        'fps': 30,
        'scale': 2.0,
        'sharpen': 1.2,
        'direction': 'right'
    }

@app.route('/api/draft', methods=['GET', 'POST'])
def handle_draft():
    if request.method == 'POST':
        # Save draft parameters
        data = request.json if request.is_json else request.form
        draft_data = {}
        
        # Read existing draft or default
        if os.path.exists(DRAFT_PATH):
            try:
                with open(DRAFT_PATH, 'r', encoding='utf-8') as f:
                    draft_data = json.load(f)
            except:
                draft_data = get_default_draft()
        else:
            draft_data = get_default_draft()
            
        # Update with new values
        for key in draft_data.keys():
            if key in data:
                val = data.get(key)
                # Cast based on default type
                orig_type = type(draft_data[key])
                try:
                    if orig_type == bool:
                        draft_data[key] = str(val).lower() == 'true'
                    elif orig_type == int:
                        draft_data[key] = int(val)
                    elif orig_type == float:
                        draft_data[key] = float(val)
                    else:
                        draft_data[key] = val
                except:
                    draft_data[key] = val
                    
        try:
            with open(DRAFT_PATH, 'w', encoding='utf-8') as f:
                json.dump(draft_data, f, indent=4)
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    else:
        # GET: return draft parameters
        if os.path.exists(DRAFT_PATH):
            try:
                with open(DRAFT_PATH, 'r', encoding='utf-8') as f:
                    draft_data = json.load(f)
                return jsonify({'success': True, 'draft': draft_data})
            except Exception as e:
                return jsonify({'success': False, 'error': f"Failed to read draft: {e}"}), 500
        else:
            return jsonify({'success': True, 'draft': get_default_draft()})


@app.route('/api/runs')
def list_runs():
    load_all_runs_from_disk()
    summary = []
    for rid, run in active_runs.items():
        summary.append({
            'run_id': rid,
            'status': run['status'],
            'timestamp': run['timestamp'],
            'current_frame': run['current_frame'],
            'total_frames': run['total_frames'],
            'prompt': run['params'].get('prompt', '')
        })
    summary.sort(key=lambda x: x['timestamp']) #, reverse=True)
    return jsonify({'success': True, 'runs': summary})


# Serve static files from runs directory
@app.route('/runs/<path:filename>')
def serve_run_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)


# Serve static files from uploads directory
@app.route('/uploads/<path:filename>')
def serve_upload_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == '__main__':
    shutdown_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server_shutdown.log")
    
    def log_shutdown_event(event_type, details=""):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] [{event_type}] {details}\n"
        print(log_entry, flush=True)
        try:
            with open(shutdown_log_path, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except Exception as e:
            print(f"[!] Could not write to shutdown log: {e}")

    def exit_gracefully(signum, frame):
        sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        stack = "".join(traceback.format_stack(frame))
        log_shutdown_event("SIGNAL_RECEIVED", f"Server received signal {signum} ({sig_name}). Stack trace:\n{stack}")
        os._exit(0)

    def custom_excepthook(exc_type, exc_value, exc_traceback):
        err_str = "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        log_shutdown_event("UNCAUGHT_EXCEPTION", f"Uncaught exception crashed the server:\n{err_str}")
        sys.__excepthook__(exc_type, exc_value, exc_traceback)

    def custom_thread_excepthook(args):
        err_str = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
        log_shutdown_event("THREAD_EXCEPTION", f"Thread '{args.thread.name}' crashed with exception:\n{err_str}")

    sys.excepthook = custom_excepthook
    if hasattr(threading, 'excepthook'):
        threading.excepthook = custom_thread_excepthook

    signal.signal(signal.SIGINT, exit_gracefully)
    signal.signal(signal.SIGTERM, exit_gracefully)
    if hasattr(signal, 'SIGHUP'):
        def handle_sighup(signum, frame):
            log_shutdown_event("SIGHUP_RECEIVED", "Terminal disconnected (SIGHUP received). Server will continue running.")
        signal.signal(signal.SIGHUP, handle_sighup)

    log_shutdown_event("SERVER_START", f"Starting Latent Horizon Server on host 0.0.0.0, port 5001. PID={os.getpid()}")
    print(f"[*] Upload folder: {UPLOAD_FOLDER}")
    print(f"[*] Output folder: {OUTPUT_FOLDER}")
    load_all_runs_from_disk()
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)
