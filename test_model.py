import os
import torch
from PIL import Image
from transparent_background import Remover

# Disable transparent_background GUI imports
import sys, types
sys.modules["transparent_background.gui"] = types.ModuleType("transparent_background.gui")
sys.modules["transparent_background.gui.gui"] = types.ModuleType("transparent_background.gui.gui")

# ------------------ CONFIG ------------------
# Path to fine-tuned U²-Net model
FINETUNED_MODEL_PATH = r"C:\Users\MirzaPC\Desktop\Custom_Bg_Model\u2net_finetuned.pth"

# Input and output images
INPUT_IMAGE = "check8.jpg"
OUTPUT_IMAGE = r"C:\Users\MirzaPC\Desktop\Custom_Bg_Model\output_result8.png"

# Make sure output directory exists
output_dir = os.path.dirname(OUTPUT_IMAGE)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# ------------------ LOAD MODEL ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Initializing Remover...")
remover = Remover(jit=False)  # disable JIT for local mode

# Try loading your fine-tuned weights
if os.path.exists(FINETUNED_MODEL_PATH):
    try:
        state_dict = torch.load(FINETUNED_MODEL_PATH, map_location=device)
        remover.model.load_state_dict(state_dict, strict=False)
        print(f"✅ Fine-tuned model loaded successfully from: {FINETUNED_MODEL_PATH}")
    except Exception as e:
        print(f"⚠️ Failed to load fine-tuned weights: {e}")
        print("Using default pre-trained model instead.")
else:
    print(f"⚠️ Fine-tuned model not found at: {FINETUNED_MODEL_PATH}")
    print("Using default pre-trained weights.")

# ------------------ PROCESS IMAGE ------------------
try:
    print(f"Opening image: {INPUT_IMAGE}")
    image = Image.open(INPUT_IMAGE).convert("RGB")

    print("Removing background...")
    result = remover.process(image, type="rgba")  # RGBA keeps transparency

    result.save(OUTPUT_IMAGE)
    print(f"✅ Background removed successfully and saved at:\n{OUTPUT_IMAGE}")

except Exception as e:
    print(f"❌ Error: {str(e)}")
