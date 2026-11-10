#!/usr/bin/env python3
"""
CLI Keyframe Generator
Generates a 500-frame animation keyframe schedule (Deforum-style) using Ollama (mistral:latest).
Compatible with Stable Diffusion and deforum animation notebooks.
"""

import os
import sys
import json
import urllib.request
import urllib.error

# ASCII Art and Banner
BANNER = """
======================================================================
  🌌 CLI KEYFRAME GENERATOR v1.0 - Ollama Mistral Edition 🌌
======================================================================
This tool asks you 5 sci-fi/fantasy worldbuilding questions and 
generates a structured 500-frame keyframe schedule for animation prompts.
======================================================================
"""

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_input(prompt, default=""):
    try:
        user_val = input(prompt).strip()
        if not user_val:
            return default
        return user_val
    except (KeyboardInterrupt, SystemExit):
        print("\n\n[!] Generation cancelled. Goodbye!")
        sys.exit(0)

def main():
    clear_screen()
    print(BANNER)
    
    # 5 Questions to build the story
    print("✨ Let's construct your sci-fi/fantasy universe! Answer the 5 questions below:")
    print("-" * 70)
    
    q1_default = "dense alien tropical jungle, glowing foliage and giant tropical flowers"
    setting = get_input(
        f"1. What is the SETTING of your story?\n   [Default: {q1_default}]\n   > ", 
        q1_default
    )
    print()

    q2_default = "floating glowing blue glass sphere containing another dimension"
    object_of_power = get_input(
        f"2. What is the key MYSTICAL or TECHNOLOGICAL OBJECT of power?\n   [Default: {q2_default}]\n   > ", 
        q2_default
    )
    print()

    q3_default = "an ancient stone temple guarded by two giant stone Gods at the entrance"
    guardians_default = "floating red glowing spheres"
    guardians = get_input(
        f"3. Who or what are the GUARDIANS or threats in this setting?\n   [Default: {guardians_default}]\n   > ", 
        q3_default
    )
    print()

    q4_default = "squeezing through the slightly open stone temple doors, finding a maze of tunnels inside"
    conflict = get_input(
        f"4. What is the primary ACTION, conflict, or twist that happens?\n   [Default: {q4_default}]\n   > ", 
        q4_default
    )
    print()

    q5_default = "vibrant tropical jungle style, cinematic lighting, photorealistic, Unreal Engine 5 render"
    art_style = get_input(
        f"5. What is the VISUAL ART STYLE or aesthetic direction?\n   [Default: {q5_default}]\n   > ", 
        q5_default
    )
    print()

    clear_screen()
    print(BANNER)
    print("🪐 WORLD SPECIFICATIONS CONFIRMED:")
    print(f"   [Setting]        : {setting}")
    print(f"   [Power Object]   : {object_of_power}")
    print(f"   [Guardians]      : {guardians}")
    print(f"   [Action/Climax]  : {conflict}")
    print(f"   [Artistic Style] : {art_style}")
    print("-" * 70)
    
    print("\n🤖 Connecting to local Ollama (http://localhost:11434)...")
    print("   Make sure Ollama is running and you have run: 'ollama run mistral'")
    
    ollama_url = "http://localhost:11434/api/generate"
    
    prompt_instruction = f"""
You are an expert Deforum Stable Diffusion keyframe scheduler and story architect.
Write a 500-frame keyframe schedule for a sci-fi/fantasy animation.
The story and prompt progression must be based on the user's choices:
- Setting: {setting}
- Key Object of Power: {object_of_power}
- Guardians / Entities: {guardians}
- Conflict / Climax / Twist: {conflict}
- Artistic Style: {art_style}

Output only raw lines of keyframe prompt schedules starting at frame 0 and progressing chronologically up to 500 (spaced out appropriately e.g. at frame 0, 10, 20, 40, 50, 70, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360, 400, 440, 480, 500).
Format every line EXACTLY as:
frame_number: descriptive prompt text, detailed art style tags

Here is an example format:
0: this appears to be an another dimension on an alien planet, a faint sign of a tropical jungle is slightly visible, cinematic lighting, photorealistic
20: dense alien tropical jungle, foliage and tropical flowers, a stone path, leading into the depths of the jungle, cinematic lighting, photorealistic
50: the large stone temple doors appears to be slightly open, just enough to squeeze inside, cinematic lighting, photorealistic
90: zoom in to view inside the temple, inside is a maze of tunnels, and a blue glass sphere, cinematic lighting, photorealistic

Keep the prompt descriptions highly visual, evocative, and matching the requested {art_style} style.
Do not include any intro, outro, conversational talk, explanations, or codeblocks. Only return the raw keyframe lines, one per line.
"""

    payload = {
        "model": "llama2-uncensored:7b",
        "prompt": prompt_instruction,
        "stream": False
    }

    try:
        req = urllib.request.Request(
            ollama_url, 
            data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        
        print("\n⏳ Generating keyframes... (this may take 10-30 seconds)")
        with urllib.request.urlopen(req, timeout=1000) as response:
            res_data = json.loads(response.read().decode('utf-8'))
            raw_output = res_data.get("response", "").strip()
            
        if not raw_output:
            print("[!] Ollama returned an empty response. Please verify your mistral:latest model.")
            sys.exit(1)
            
        filename = "keyframes.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(raw_output)
            
        print("\n🎉 SUCCESS!")
        print(f"💾 Keyframe schedule written to: {os.path.abspath(filename)}")
        print("\nPreview of keyframes:")
        print("-" * 70)
        
        lines = raw_output.split("\n")
        preview_lines = lines[:10]
        for line in preview_lines:
            print(f"  {line}")
        if len(lines) > 10:
            print(f"  ... and {len(lines) - 10} more lines.")
            
        print("-" * 70)
        print("\nReady! You can copy and paste keyframes.txt into your Deforum stable diffusion notebook.")
        
    except urllib.error.URLError as e:
        print("\n❌ Error connecting to Ollama!")
        print(f"   Reason: {e.reason}")
        print("\n💡 Troubleshooting Tips:")
        print("   1. Verify Ollama is installed (https://ollama.com)")
        print("   2. Run 'ollama run mistral' in your terminal first")
        print("   3. Ensure Ollama is running on port 11434")
    except Exception as e:
        print(f"\n❌ Unexpected error occurred: {e}")

if __name__ == "__main__":
    main()
