#!/usr/bin/env python3
"""
Stable Diffusion Local API Server (Standard-library only server)
Run this on your local machine containing your GPUs and model checkpoints:
    python sd_server.py --model "/path/to/your/models/directory" --port 5000
Then, in the React UI, switch the backend to "Local SD Server" and type "http://127.0.0.1:5000"!
"""
import os
import re
import json
import argparse
import torch
import base64
import gc
from io import BytesIO
from http.server import HTTPServer, BaseHTTPRequestHandler
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, LCMScheduler, StableDiffusionControlNetPipeline, ControlNetModel
import signal
# Global server states
pipe = None
device = "cpu"
loaded_model_path = ""
available_models = {}  # maps filename -> full path
available_loras = {}   # maps filename -> full path
loaded_loras = {"lora1": None, "lora2": None}  # tracks what's currently loaded in pipe
controlnet_model = None
loaded_controlnet_name = ""

def get_controlnet_model(model_name):
    global controlnet_model, loaded_controlnet_name, device
    repo_map = {
        "openpose": "lllyasviel/sd-controlnet-openpose",
        "canny": "lllyasviel/sd-controlnet-canny",
        "scribble": "lllyasviel/sd-controlnet-scribble"
    }
    repo = repo_map.get(model_name.lower(), "lllyasviel/sd-controlnet-openpose")
    if controlnet_model is None or loaded_controlnet_name != repo:
        print(f"[*] Loading ControlNet model: {repo}")
        controlnet_model = ControlNetModel.from_pretrained(repo, torch_dtype=torch.float32).to(device)
        loaded_controlnet_name = repo
    return controlnet_model

def scan_models(path_arg):
    models_dict = {}
    if os.path.isdir(path_arg):
        print(f"[*] Scanning directory for checkpoints: {path_arg}")
        for root, dirs, files in os.walk(path_arg):
            for file in files:
                if file.endswith((".safetensors", ".ckpt")):
                    full_path = os.path.abspath(os.path.join(root, file))
                    models_dict[file] = full_path
    elif os.path.isfile(path_arg):
        # Scan the directory containing this file as well!
        parent_dir = os.path.dirname(os.path.abspath(path_arg))
        print(f"[*] Detected file path. Scanning parent directory for sibling checkpoints: {parent_dir}")
        for root, dirs, files in os.walk(parent_dir):
            for file in files:
                if file.endswith((".safetensors", ".ckpt")):
                    full_path = os.path.abspath(os.path.join(root, file))
                    models_dict[file] = full_path
        # Guarantee that the specified file itself is in the dict
        if path_arg.endswith((".safetensors", ".ckpt")):
            models_dict[os.path.basename(path_arg)] = os.path.abspath(path_arg)
    return models_dict

def load_model_pipeline(model_path):
    global pipe, device, loaded_model_path, loaded_loras
    print(f"\n[*] Loading model checkpoint: {model_path}")
    # 1. Unload previous pipeline to prevent VRAM / RAM leaks
    if pipe is not None:
        print("[*] Unloading previous model from memory...")
        del pipe
        pipe = None
        loaded_loras = {"lora1": None, "lora2": None}
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("[*] Cleared CUDA cache.")

    # 2. Setup type and load pipeline
    torch_dtype = torch.float32 #torch.float16 if device == "cuda" else torch.float32
    try:
        pipe = StableDiffusionPipeline.from_single_file(
            model_path,
            torch_dtype=torch_dtype,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False
        )
        pipe = pipe.to(device)
        # CPU or GPU optimization
        if device == "cpu":
            pipe.enable_attention_slicing()
        else:
            pipe.enable_attention_slicing()
        loaded_model_path = model_path
        print(f"[+] Successfully loaded model: {os.path.basename(model_path)}")
        return True
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to load model. Details: {e}")
        return False

def parse_prompt_weights(prompt):
    """
    Parses a prompt with weighting syntax like (text:weight), (text) or [text]
    and returns:
      1. The clean prompt string (with parentheses/weights stripped)
      2. A list of dicts: {"phrase": "text", "weight": float}
    """
    clean_prompt = prompt
    weights_info = []
    # Match (text:weight)
    pattern_colon = re.compile(r'\(([^)]+):([0-9.]+)\)')
    while True:
        match = pattern_colon.search(clean_prompt)
        if not match:
            break
        full_match = match.group(0)
        phrase = match.group(1).strip()
        try:
            weight = float(match.group(2))
        except ValueError:
            weight = 1.0
        # Replace the first occurrence of this match with the phrase
        clean_prompt = clean_prompt.replace(full_match, phrase, 1)
        weights_info.append({"phrase": phrase, "weight": weight})
    # Match (text) -> weight 1.1
    pattern_paren = re.compile(r'\(([^)]+)\)')
    while True:
        match = pattern_paren.search(clean_prompt)
        if not match:
            break
        full_match = match.group(0)
        phrase = match.group(1).strip()
        if ":" in phrase:
            break
        weight = 1.1
        clean_prompt = clean_prompt.replace(full_match, phrase, 1)
        weights_info.append({"phrase": phrase, "weight": weight})
    # Match [text] -> weight 0.9
    pattern_bracket = re.compile(r'\[([^\]]+)\]')
    while True:
        match = pattern_bracket.search(clean_prompt)
        if not match:
            break
        full_match = match.group(0)
        phrase = match.group(1).strip()
        weight = 0.9
        clean_prompt = clean_prompt.replace(full_match, phrase, 1)
        weights_info.append({"phrase": phrase, "weight": weight})
    return clean_prompt, weights_info

def find_sublist_indices(full_list, sub_list):
    """
    Returns a list of start indices where sub_list occurs inside full_list.
    """
    indices = []
    n = len(full_list)
    m = len(sub_list)
    if m == 0 or n < m:
        return indices
    for i in range(n - m + 1):
        if full_list[i:i+m] == sub_list:
            indices.append(i)
    return indices

def scale_prompt_embeddings(pipe, prompt):
    """
    Parses weights from prompt, encodes the clean prompt, 
    scales embeddings of the weighted phrases, and returns prompt_embeds.
    """
    if not prompt:
        return None
    clean_prompt, weights_info = parse_prompt_weights(prompt)
    # Encode the clean prompt to get token IDs and initial embeddings
    text_inputs = pipe.tokenizer(
        clean_prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    text_input_ids = text_inputs.input_ids
    
    # Convert text_input_ids to list of python ints
    full_token_list = list(text_input_ids[0].cpu().numpy())
    
    # Get initial embeddings from text encoder
    device_used = pipe.text_encoder.device
    with torch.no_grad():
        prompt_embeds = pipe.text_encoder(text_input_ids.to(device_used))[0]
    
    # If there are no weights, just return the embeds as-is
    if not weights_info:
        return prompt_embeds
        
    # Scale embeddings for each weighted phrase
    for item in weights_info:
        phrase = item["phrase"]
        weight = item["weight"]
        if weight == 1.0:
            continue
            
        # Try matching with and without leading space to handle CLIP tokenizer spacing quirks
        sub_tokens_variants = [
            pipe.tokenizer.encode(" " + phrase, add_special_tokens=False),
            pipe.tokenizer.encode(phrase, add_special_tokens=False)
        ]
        
        matched = False
        for sub_tokens in sub_tokens_variants:
            if not sub_tokens:
                continue
            indices = find_sublist_indices(full_token_list, sub_tokens)
            if indices:
                # Multiply the embeddings for these tokens by weight
                for start_idx in indices:
                    for offset in range(len(sub_tokens)):
                        prompt_embeds[0, start_idx + offset, :] *= weight
                matched = True
                break
                
        if not matched:
            print(f"    [!] Warning: Could not match phrase tokens for '{phrase}' in clean prompt '{clean_prompt}'")
            
    return prompt_embeds

def scan_loras(path_arg):
    loras_dict = {}
    if not path_arg:
        return loras_dict
    if os.path.isdir(path_arg):
        print(f"[*] Scanning directory for LoRAs: {path_arg}")
        for root, dirs, files in os.walk(path_arg):
            for file in files:
                if file.endswith((".safetensors", ".ckpt")):
                    full_path = os.path.abspath(os.path.join(root, file))
                    loras_dict[file] = full_path
    return loras_dict

def apply_loras(lora1_name, lora1_weight, lora2_name, lora2_weight):
    global pipe, available_loras, loaded_loras
    if pipe is None:
        return
        
    lora1_path = available_loras.get(lora1_name) if lora1_name else None
    lora2_path = available_loras.get(lora2_name) if lora2_name else None
    
    # If no LoRAs are requested, check if we need to unload
    if not lora1_path and not lora2_path:
        if loaded_loras.get("lora1") or loaded_loras.get("lora2"):
            print("[*] Unloading all LoRAs...")
            try:
                pipe.unload_lora_weights()
            except Exception as e:
                print(f"[!] Warning unloading LoRAs: {e}")
            loaded_loras = {"lora1": None, "lora2": None}
        return

    # Check if loaded adapters match the request
    needs_unload = False
    if loaded_loras.get("lora1") != lora1_path or loaded_loras.get("lora2") != lora2_path:
        needs_unload = True
        
    if needs_unload:
        print("[*] LoRA configuration changed. Unloading previous adapters...")
        try:
            pipe.unload_lora_weights()
        except Exception as e:
            print(f"[!] Warning unloading LoRAs: {e}")
        loaded_loras = {"lora1": None, "lora2": None}
        
        # Load LoRA 1 if requested
        if lora1_path:
            print(f"[*] Loading LoRA 1 weights: {lora1_name}")
            try:
                pipe.load_lora_weights(lora1_path, adapter_name="lora1")
                loaded_loras["lora1"] = lora1_path
            except Exception as e:
                print(f"[ERROR] Failed to load LoRA 1: {e}")
                
        # Load LoRA 2 if requested
        if lora2_path:
            print(f"[*] Loading LoRA 2 weights: {lora2_name}")
            try:
                pipe.load_lora_weights(lora2_path, adapter_name="lora2")
                loaded_loras["lora2"] = lora2_path
            except Exception as e:
                print(f"[ERROR] Failed to load LoRA 2: {e}")
                
    # Activate and set weights
    active_adapters = []
    active_weights = []
    if loaded_loras.get("lora1"):
        active_adapters.append("lora1")
        active_weights.append(lora1_weight)
    if loaded_loras.get("lora2"):
        active_adapters.append("lora2")
        active_weights.append(lora2_weight)
        
    if active_adapters:
        print(f"[*] Setting active LoRAs: {active_adapters} with weights {active_weights}")
        try:
            pipe.set_adapters(active_adapters, adapter_weights=active_weights)
        except Exception as e:
            print(f"[ERROR] Failed setting adapters: {e}")

def save_image_to_disk(image, prompt, save_dir):
    try:
        os.makedirs(save_dir, exist_ok=True)
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_prompt = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in prompt[:30]).strip().replace(' ', '_')
        filename = f"{timestamp}_{safe_prompt}.png"
        filepath = os.path.join(save_dir, filename)
        image.save(filepath, format="PNG")
        return filepath
    except Exception as e:
        print(f"[ERROR] Failed to save image to disk: {e}")
        return None

class SDRequestHandler(BaseHTTPRequestHandler):
    def _set_headers(self, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_OPTIONS(self):
        self._set_headers(200)

    def do_GET(self):
        global available_models, loaded_model_path, available_loras
        if self.path == "/api/models":
            self._set_headers(200)
            models_list = list(available_models.keys())
            loras_list = list(available_loras.keys())
            active_name = os.path.basename(loaded_model_path) if loaded_model_path else ""
            response = {
                "models": models_list,
                "active_model": active_name,
                "loras": loras_list
            }
            self.wfile.write(json.dumps(response).encode('utf-8'))
        elif self.path in ("/", "/index.html", "/youtube.html"):
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                # Serve youtube.html as the primary UI if it exists, otherwise fall back to index.html
                target_file = 'youtube.html' if os.path.exists(os.path.join(script_dir, 'youtube.html')) else 'index.html'
                index_path = os.path.join(script_dir, target_file)
                with open(index_path, 'rb') as f:
                    content = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(content)
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-Type', 'text/plain')
                self.end_headers()
                self.wfile.write(f"Error loading index file: {str(e)}".encode('utf-8'))
        elif self.path.startswith("/static/"):
            try:
                # Remove query strings or hash parameters if any
                path_clean = self.path.split("?")[0].split("#")[0]
                relative_path = path_clean.lstrip("/")
                
                # Check for directory traversal attacks (ensure it remains inside the current folder)
                script_dir = os.path.dirname(os.path.abspath(__file__))
                target_path = os.path.abspath(os.path.join(script_dir, relative_path))
                if not target_path.startswith(script_dir):
                    self.send_error(403, "Access Denied")
                    return
                
                if os.path.exists(target_path) and os.path.isfile(target_path):
                    self.send_response(200)
                    content_type = "application/octet-stream"
                    if target_path.endswith(".png"):
                        content_type = "image/png"
                    elif target_path.endswith((".jpg", ".jpeg")):
                        content_type = "image/jpeg"
                    elif target_path.endswith(".json"):
                        content_type = "application/json"
                    elif target_path.endswith(".js"):
                        content_type = "application/javascript"
                    elif target_path.endswith(".css"):
                        content_type = "text/css"
                    elif target_path.endswith(".html"):
                        content_type = "text/html"
                    
                    self.send_header('Content-Type', content_type)
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    with open(target_path, 'rb') as f:
                        self.wfile.write(f.read())
                    return
                else:
                    self.send_error(404, "File not found")
                    return
            except Exception as e:
                self.send_error(500, f"Server error: {str(e)}")
                return
        else:
            self.send_error(404, "Endpoint not found")

    def do_POST(self):
        global pipe, device, available_models, loaded_model_path
        
        if self.path == "/api/select-model":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode('utf-8'))
            except Exception:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "Invalid JSON body"}).encode('utf-8'))
                return

            model_name = data.get("model", "")
            if not model_name or model_name not in available_models:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": f"Model '{model_name}' not found. Available: {list(available_models.keys())}"}).encode('utf-8'))
                return

            target_path = available_models[model_name]
            if target_path == loaded_model_path:
                self._set_headers(200)
                self.wfile.write(json.dumps({
                    "status": "success",
                    "active_model": model_name,
                    "info": f"Model '{model_name}' is already active."
                }).encode('utf-8'))
                return

            success = load_model_pipeline(target_path)
            if success:
                self._set_headers(200)
                self.wfile.write(json.dumps({
                    "status": "success",
                    "active_model": model_name,
                    "info": f"Successfully loaded model: {model_name}"
                }).encode('utf-8'))
            else:
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": f"Failed to load model checkpoint: {model_name}"}).encode('utf-8'))

        elif self.path == "/api/generate":
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                data = json.loads(post_data.decode('utf-8'))
            except Exception:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "Invalid JSON body"}).encode('utf-8'))
                return

            prompt = data.get("prompt", "")
            negative_prompt = data.get("negative_prompt", "blurry, low quality")
            steps = int(data.get("steps", 30))
            guidance_scale = float(data.get("guidance_scale", 1.2))
            width = int(data.get("width", 512))
            height = int(data.get("height", 512))
            seed = int(data.get("seed", 12345))
            requested_model = data.get("model", "")
            
            # Parse LoRA parameters
            lora1_name = data.get("lora1", "")
            lora1_weight = float(data.get("lora1_weight", 0.75))
            lora2_name = data.get("lora2", "")
            lora2_weight = float(data.get("lora2_weight", 0.75))
            
            # Parse save directory
            save_dir = data.get("save_dir", "")

            if not prompt:
                self._set_headers(400)
                self.wfile.write(json.dumps({"error": "Prompt is required"}).encode('utf-8'))
                return

            # Check if model swap is requested
            if requested_model and requested_model in available_models:
                target_path = available_models[requested_model]
                if target_path != loaded_model_path:
                    print(f"[*] Auto-swapping model to requested: {requested_model}")
                    success = load_model_pipeline(target_path)
                    if not success:
                        self._set_headers(500)
                        self.wfile.write(json.dumps({"error": f"Failed to auto-swap to model: {requested_model}"}).encode('utf-8'))
                        return

            print(f"\n[*] Received generation request:")
            print(f"    - Active Model: {os.path.basename(loaded_model_path)}")
            print(f"    - Prompt: {prompt[:60]}...")
            print(f"    - Resolution: {width}x{height}")
            print(f"    - Steps: {steps} | Guidance: {guidance_scale} | Seed: {seed}")

            # Reconfigure scheduler dynamically based on step count / LCM naming
            is_lcm = steps <= 12 or "lcm" in loaded_model_path.lower()
            try:
                if is_lcm:
                    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
                    print("    - Configured Scheduler: LCM Scheduler")
                else:
                    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                        pipe.scheduler.config,
                        use_karras_sigmas=True
                    )
                    print("    - Configured Scheduler: DPMSolverMultistepScheduler (Karras)")
            except Exception as sched_err:
                print(f"    [!] Scheduler adjustment warning: {sched_err}")

            # Run inference
            generator = torch.Generator(device=device).manual_seed(seed)
            try:
                # Custom prompt weighting support
                prompt_embeds = scale_prompt_embeddings(pipe, prompt)
                negative_prompt_embeds = scale_prompt_embeddings(pipe, negative_prompt)

                # Dynamically apply LoRAs
                apply_loras(lora1_name, lora1_weight, lora2_name, lora2_weight)

                control_image_b64 = data.get("control_image", "")
                controlnet_name = data.get("controlnet", "openpose")
                controlnet_scale = float(data.get("controlnet_conditioning_scale", 1.0))

                if control_image_b64:
                    print(f"    - ControlNet Active: {controlnet_name} | Scale: {controlnet_scale}")
                    import io
                    from PIL import Image
                    if "," in control_image_b64:
                        control_image_b64 = control_image_b64.split(",")[1]
                    image_data = base64.b64decode(control_image_b64)
                    pil_control_image = Image.open(io.BytesIO(image_data)).convert("RGB")
                    
                    cnet = get_controlnet_model(controlnet_name)
                    control_pipe = StableDiffusionControlNetPipeline(
                        vae=pipe.vae,
                        text_encoder=pipe.text_encoder,
                        tokenizer=pipe.tokenizer,
                        unet=pipe.unet,
                        controlnet=cnet,
                        scheduler=pipe.scheduler,
                        safety_checker=pipe.safety_checker,
                        feature_extractor=pipe.feature_extractor,
                        requires_safety_checker=pipe.config.requires_safety_checker if hasattr(pipe.config, "requires_safety_checker") else False
                    )
                    
                    with torch.inference_mode():
                        output = control_pipe(
                            prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=negative_prompt_embeds,
                            image=pil_control_image,
                            controlnet_conditioning_scale=controlnet_scale,
                            num_inference_steps=steps,
                            guidance_scale=guidance_scale,
                            width=width,
                            height=height,
                            generator=generator
                        )
                        image = output.images[0]
                else:
                    with torch.inference_mode():
                        output = pipe(
                            prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=negative_prompt_embeds,
                            num_inference_steps=steps,
                            guidance_scale=guidance_scale,
                            width=width,
                            height=height,
                            generator=generator
                        )
                        image = output.images[0]

                # Convert to base64
                buffered = BytesIO()
                image.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
                data_url = f"data:image/png;base64,{img_base64}"

                saved_path = None
                if save_dir:
                    saved_path = save_image_to_disk(image, prompt, save_dir)

                self._set_headers(200)
                response = {
                    "imageUrl": data_url,
                    "appliedPrompt": prompt,
                    "saved_path": saved_path,
                    "info": f"Generated offline using model: {os.path.basename(loaded_model_path)}"
                }
                self.wfile.write(json.dumps(response).encode('utf-8'))
                print("[+] Successfully generated image and returned base64 data to UI.")
            except Exception as gen_err:
                print(f"[ERROR] Generation failed: {gen_err}")
                self._set_headers(500)
                self.wfile.write(json.dumps({"error": f"Generation failed: {str(gen_err)}"}).encode('utf-8'))
        else:
            self.send_error(404, "Endpoint not found")

def start_server(port, model_arg, device_req, lora_arg=None):
    global pipe, device, loaded_model_path, available_models, available_loras

    # 1. Device Setup
    if device_req == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = device_req

    print(f"[*] Starting Stable Diffusion Local Server...")
    print(f"[*] Target Device: {device.upper()}")

    if device == "cpu":
        print("\n" + "="*70)
        print("[!] NOTICE: RUNNING IN CPU-ONLY MODE")
        print("    Standard Stable Diffusion (25-50 steps) will take 1-3 minutes per image on CPU.")
        print("    RECOMMENDATIONS FOR MAXIMUM SPEED:")
        print("    1. Use a Latent Consistency Model (LCM) checkpoint (e.g. 'lcm-sd15.safetensors').")
        print("    2. Adjust the slider in the frontend to use 4 to 8 steps.")
        print("    3. Change guidance scale (CFG) to 1.0 - 2.0 (standard for LCM).")
        print("    This will generate beautiful, high-quality images on CPU in under 15-30 seconds!")
        print("="*70 + "\n")

    # 2. Scan path for checkpoints
    available_models = scan_models(model_arg)
    if not available_models:
        print(f"\n[CRITICAL ERROR] The path '{model_arg}' contains no valid '.safetensors' or '.ckpt' model files!")
        print("[*] Please download a Stable Diffusion checkpoint (e.g. v1-5-pruned-emaonly.safetensors) and place it there.")
        return

    print(f"[+] Found {len(available_models)} model checkpoint(s):")
    for idx, name in enumerate(available_models.keys()):
        print(f"    {idx+1}. {name}")

    # Auto-detect LoRAs path if not specified
    if not lora_arg:
        # Check sibling directories
        if os.path.isdir(model_arg):
            sibling_lora = os.path.abspath(os.path.join(model_arg, "..", "loras"))
            if os.path.isdir(sibling_lora):
                lora_arg = sibling_lora
        elif os.path.isfile(model_arg):
            parent_dir = os.path.dirname(os.path.abspath(model_arg))
            sibling_lora = os.path.abspath(os.path.join(parent_dir, "..", "loras"))
            if os.path.isdir(sibling_lora):
                lora_arg = sibling_lora
        
        # Sibling absolute fallback
        if not lora_arg or not os.path.isdir(lora_arg):
            comfy_loras = "/home/jack/Desktop/Comfy-UI/models/loras"
            if os.path.isdir(comfy_loras):
                lora_arg = comfy_loras

    if lora_arg and os.path.isdir(lora_arg):
        available_loras = scan_loras(lora_arg)
        print(f"[+] Found {len(available_loras)} LoRA file(s) in: {lora_arg}")
    else:
        print("[!] No LoRA directory found or specified. LoRA functionality will be unavailable.")

    # 3. Load first discovered model as default
    default_model_name = list(available_models.keys())[0]
    default_model_path = available_models[default_model_name]
    print(f"[*] Loading default model: {default_model_name}")
    
    success = load_model_pipeline(default_model_path)
    if not success:
        print("[CRITICAL ERROR] Failed to load the default model at startup.")
        return

    # 4. Start Server
    server_address = ('', port)
    httpd = HTTPServer(server_address, SDRequestHandler)
    print(f"\n[+] API Server running locally on: http://localhost:{port}")
    print(f"[*] GET /api/models - List checkpoints & LoRAs")
    print(f"[*] POST /api/select-model - Swap active checkpoint")
    print(f"[*] POST /api/generate - Run image generation (with multi-LoRA support)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Server stopped.")

if __name__ == "__main__":

    def exit_gracefully(signum, frame):
        print("\n[*] Ctrl+C / termination signal received. Shutting down Latent Horizon server instantly...")
        os._exit(0)
    signal.signal(signal.SIGINT, exit_gracefully)
    signal.signal(signal.SIGTERM, exit_gracefully)



    parser = argparse.ArgumentParser(description="Local Stable Diffusion API Server")
    parser.add_argument("--model", type=str, required=True, help="Path to local checkpoint file or a directory of checkpoints")
    parser.add_argument("--port", type=int, default=5000, help="Port to bind the server to")
    parser.add_argument("--device", type=str, default="auto", help="Execution device: auto, cuda, cpu")
    parser.add_argument("--loras", type=str, default="", help="Path to local directory containing LoRA files")
    args = parser.parse_args()

    start_server(args.port, args.model, args.device, args.loras)
