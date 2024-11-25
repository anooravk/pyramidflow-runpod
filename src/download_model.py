import base64
import os
import time
import requests
import runpod
import shutils
import os
import torch
from PIL import Image, ImageOps
from huggingface_hub import snapshot_download
from pyramid_dit import PyramidDiTForVideoGeneration
from diffusers.utils import export_to_video
import spaces 
import uuid





is_canonical = False

# Constants
MODEL_PATH = "pyramid-flow-model"
MODEL_REPO = "rain1011/pyramid-flow-sd3"
MODEL_VARIANT = "diffusion_transformer_768p"
MODEL_DTYPE = "bf16"
# Download and load the model
def load_model():
    if not os.path.exists(MODEL_PATH):
        snapshot_download(MODEL_REPO, local_dir=MODEL_PATH, local_dir_use_symlinks=False, repo_type='model')
    
    model = PyramidDiTForVideoGeneration(
        MODEL_PATH,
        MODEL_DTYPE,
        model_variant=MODEL_VARIANT,
    )
    
    model.vae.to("cuda")
    model.dit.to("cuda")
    model.text_encoder.to("cuda")
    model.vae.enable_tiling()
    
    return model

# Global model variable
model = load_model()
