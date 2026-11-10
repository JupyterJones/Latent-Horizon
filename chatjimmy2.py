import torch
from torch import nn
from torch import cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.optim import *
from torch.optim.lr import *
import torch.utils as utils

# Constants
WIDTH = 384
HEIGHT = 512
MODEL_PATH = "/home/jack/Desktop/Comfy-UI/models/checkpoints/dreamshaper_8LCM.safetensors"
VAE_PATH = "/home/jack/Desktop/Comfy-UI/models/checkpoints/vae-ft-mse-840000-ema-pruned.safetensors"
CFG = 1.5
SEED = 12345678
STRENGTH = 0.38
FRAMES = 120
ZOOM = 1.005

# Model and VAE loading
def load_pipeline(model_path, vae_path):
    try:
        from diffusers import StableDiffusionImg2ImgPipeline
        from diffusers import AutoencoderKL

        # Load model
        pipe = StableDiffusionImg2ImgPipeline.from_single_file(
            model_path,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            safety_checker=None,
            local_files_only=False
        )

        # Load VAE (if specified)
        if vae_path:
            try:
                pipe.vae = AutoencoderKL.from_single_file(
                    vae_path,
                    torch_dtype=torch.float16 if "cuda" in device else torch.float32,
                    local_files_only=False
                )
            except Exception as e:
                log_msg(f"[!] Warning: Failed to load external VAE '{vae_path}': {e}. Falling back to built-in VAE.")
        else:
            # Use built-in VAE
            log_msg("[*] Using built-in VAE (no external VAE selected/found).")

        # LCM scheduler
        if "LCM" in os.path.basename(model_path).upper() or "LCM" in model_path.upper():
            from diffusers import LCMScheduler
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

        pipe = pipe.to(device)

        return pipe
    except Exception as e:
        log_msg(f"[!] Error: Failed to load model {model_path}: {e}")
        return None

# Main script
if __name__ == "__main__":
    # Initialize pipeline
    pipe = load_pipeline(MODEL_PATH, VAE_PATH, device)

    # Define constants
    PROMPT = "a rocket launching, cinematic vast landscape, ultra detailed, coherent world, medieval timeline, brightly colored people exploring mountainous terrain, atmospheric depth, volumetric lighting, high contrast "
    NEG_PROMPT = "blurry, low quality, text, watermark, distorted"
    STEPS = 11
    CFG = 1.5
    SEED = 12345678

    # Load image
    img = Image.open("/home/jack/Desktop/ZOOMS/uploads/Untitled01.jpg").convert("RGB")

    # Main loop
    for i in range(FRAMES):
        ic("frame", i)

        # Encode image
        latent = encode(img)

        # Zoom and decode
        zoomed = torch.nn.functional.interpolate(
            latent,
            size=(int(HEIGHT/ZOOM), int(WIDTH/ZOOM)),
            mode="bilinear",
            align_corners=False
        )
        img = decode(zoomed)

        # Update latent and save frame
        latent = encode(img)
        latent_path = f"latent_{i:04d}.png"
        img.save(latent_path)
        ic("saved", latent_path)

    # Video encode
    VIDEO_OUT = "infinite_zoom.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate","12",
        "-i", f"latent_{i:04d}.png",
        "-c:v","libx264",
        "-pix_fmt","yuv420p",
        "-crf","18",
        VIDEO_OUT
    ]
    subprocess.run(cmd, check=True)

    ic("done ->", VIDEO_OUT)