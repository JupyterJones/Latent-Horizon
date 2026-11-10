import os
import json
from datetime import datetime
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from icecream import ic
import signal
from sys import argv
# 1. GENERATION CONFIGURATION
STEPS = 3
GUIDANCE_SCALE = 1.10
SIZE = (512, 768)
DENOISE = 0.76
# Backend-controlled fallback model list
AVAILABLE_MODELS = [
    "comiccraftLCM_beta4.safetensors",
    "cyberrealisticLCM_cyberrealistic42.safetensors",
    "dreamshaper_8LCM.safetensors",
    "xenogasmLCMEditionSFWNSFW_lcmV5.safetensors",
    "realisticLCMBYStable_v10.safetensors",
    "photonLCM_v10.safetensors",
    "silversRealmixLCM_v10.safetensors",
    "frtMixmodelLCM_realisticV10.safetensors",
    "epicphotogasmLCM_ultimatefidelity.safetensors",
    "realmodelbase02LCM_v10.safetensors",
    "epicrealism_naturalSinRC1VAE.safetensors",
    "delusionsLCM_v20.safetensors",
    "truevisionLCM77_v10.safetensors",
    "xenoxtcLCMEditionArtToon_v10.safetensors",
    "photographerAlpha7_v10.safetensors"
]

MODEL_FILE = "/home/jack/Desktop/Comfy-UI/models/checkpoints/xenogasmLCMEditionSFWNSFW_lcmV5.safetensors"
NAME = MODEL_FILE.split("/")[-1]
ic(NAME)

LORA_FILE = "/home/jack/Desktop/Comfy-UI/models/loras/more_details.safetensors"
LORA = LORA_FILE.split("/")[-1]
ic(f"LORA: {LORA}")

OUTPUT_DIR = "./EXPERIMENTAL"
JSON_DB = "./EXPERIMENTAL/generations.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

NEGATIVE_PROMPT = """bad anatomy, extra limbs, blurry, low quality, watermark, text, deformed face, mutated hands, worst quality, NSFW, nudity, nsfw, oversized breasts, nipples"""

def load_db():
    if os.path.exists(JSON_DB):
        with open(JSON_DB, "r") as f:
            return json.load(f)
    return {"next_image_id": 1, "runs": []}

def save_db(db):
    with open(JSON_DB, "w") as f:
        json.dump(db, f, indent=4)

db = load_db()

ic("Loading txt2img pipeline")
pipe = StableDiffusionPipeline.from_single_file(
    MODEL_FILE,
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False
)
pipe = pipe.to("cpu")

ic("Optimizing CPU pipeline")
pipe.enable_attention_slicing()

ic("Switching scheduler to DPM++ 2M Karras equivalent")
pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    pipe.scheduler.config,
    use_karras_sigmas=True
)

ic("Loading LoRA")
pipe.load_lora_weights(LORA_FILE, adapter_name="lora")
pipe.fuse_lora(lora_scale=0.1)

NAMED = NAME.replace(".", "-")

def get_filename(db):
    img_id = db["next_image_id"]
    db["next_image_id"] += 1
    return img_id, f"{NAMED}_img_{img_id:06d}.png"

def generate(seed, PROMPT):
    ic(f"Generating image with seed: {seed}")
    generator = torch.Generator(device="cpu").manual_seed(seed)

    with torch.inference_mode():
        image = pipe(
            prompt=PROMPT,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=STEPS,
            denoise=DENOISE,
            guidance_scale=GUIDANCE_SCALE,
            width=SIZE[0],
            height=SIZE[1],
            generator=generator
        ).images[0]

    img_id, filename = get_filename(db)
    path = os.path.join(OUTPUT_DIR, filename)
    image.save(path)

    db["runs"].append({
        "image_id": img_id,
        "file": filename,
        "timestamp": datetime.now().isoformat(),
        "prompt": PROMPT.strip(),
        "seed": seed
    })
    save_db(db)
    ic(f"Saved {path}")

def main(PROMPT, seed):
    
    generate(seed, PROMPT)
    seed = seed + 1
    ic("DONE")

if __name__ == "__main__":
    def exit_gracefully(signum, frame):
        print("\n[*] Ctrl+C / termination signal received. Shutting down Latent Horizon server instantly...")
        os._exit(0)
    signal.signal(signal.SIGINT, exit_gracefully)
    signal.signal(signal.SIGTERM, exit_gracefully)
    seed = 12345  # Starting seed
    PROMPT=argv[1]
    main(PROMPT, seed)
    seed = seed + 1