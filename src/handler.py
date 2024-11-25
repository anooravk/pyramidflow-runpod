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

def center_crop(image, target_width, target_height):
    width, height = image.size
    aspect_ratio_target = target_width / target_height
    aspect_ratio_image = width / height

    if aspect_ratio_image > aspect_ratio_target:
        # Crop the width (left and right)
        new_width = int(height * aspect_ratio_target)
        left = (width - new_width) // 2
        right = left + new_width
        top, bottom = 0, height
    else:
        # Crop the height (top and bottom)
        new_height = int(width / aspect_ratio_target)
        top = (height - new_height) // 2
        bottom = top + new_height
        left, right = 0, width

    image = image.crop((left, top, right, bottom))
    return image

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

# Text-to-video generation function

def generate_video(prompt, image=None, duration=3, guidance_scale=9, video_guidance_scale=5, frames_per_second=8):
    multiplier = 1.2 if is_canonical else 3.0
    temp = int(duration * multiplier) + 1
    torch_dtype = torch.bfloat16 if MODEL_DTYPE == "bf16" else torch.float32
    if(image):
        cropped_image = center_crop(image, 1280, 768)
        resized_image = cropped_image.resize((1280, 768))
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch_dtype):
            frames = model.generate_i2v(
                prompt=prompt,
                input_image=resized_image,
                num_inference_steps=[10, 10, 10],
                temp=temp,
                guidance_scale=7.0,
                video_guidance_scale=video_guidance_scale,
                output_type="pil",
                save_memory=True,
            )
    else:
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch_dtype):
            frames = model.generate(
                prompt=prompt,
                num_inference_steps=[20, 20, 20],
                video_num_inference_steps=[10, 10, 10],
                height=1280,
                width=768,
                temp=temp,
                guidance_scale=guidance_scale,
                video_guidance_scale=video_guidance_scale,
                output_type="pil",
                save_memory=True,
            )
    output_path = f"{str(uuid.uuid4())}_output_video.mp4"
    export_to_video(frames, output_path, fps=frames_per_second)
    return output_path



def handler(job):
    try:
        job_input = job.get("input", {})
        prompt = job_input.get("prompt")
        duration = job_input.get("duration", 3)
        frames_per_second = job_input.get("frames_per_second", 8)

        if not prompt:
            return {"error": "Missing 'prompt' in input."}

        # Generate the video
        videoputfilepath = generate_video(
            prompt=prompt, 
            duration=duration, 
            frames_per_second=frames_per_second
        )

        # Convert video to Base64
        with open(videoputfilepath, "rb") as video_file:
            video_base64 = base64.b64encode(video_file.read()).decode("utf-8")

        # Clean up generated file
        os.remove(videoputfilepath)

        return {"video_base64": video_base64}

    except Exception as e:
        return {"error": f"Job failed: {str(e)}. Check input parameters or system setup."}

runpod.serverless.start({"handler": handler})
