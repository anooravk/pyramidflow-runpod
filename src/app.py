import os
import torch
import gradio as gr
from PIL import Image, ImageOps

from huggingface_hub import snapshot_download
from pyramid_dit import PyramidDiTForVideoGeneration
from diffusers.utils import export_to_video

import spaces 
import uuid

is_canonical = True if os.environ.get("SPACE_ID") == "Pyramid-Flow/pyramid-flow" else False

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
@spaces.GPU(duration=140)
def generate_video(prompt, image=None, duration=3, guidance_scale=9, video_guidance_scale=5, frames_per_second=8, progress=gr.Progress(track_tqdm=True)):
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
                height=768,
                width=1280,
                temp=temp,
                guidance_scale=guidance_scale,
                video_guidance_scale=video_guidance_scale,
                output_type="pil",
                save_memory=True,
            )
    output_path = f"{str(uuid.uuid4())}_output_video.mp4"
    export_to_video(frames, output_path, fps=frames_per_second)
    return output_path

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# Pyramid Flow")
    gr.Markdown("Pyramid Flow is a training-efficient Autoregressive Video Generation model based on Flow Matching. It is trained only on open-source datasets within 20.7k A100 GPU hours")
    gr.Markdown("[[Paper](https://arxiv.org/pdf/2410.05954)], [[Model](https://huggingface.co/rain1011/pyramid-flow-sd3)], [[Code](https://github.com/jy0205/Pyramid-Flow)]")
    
    with gr.Row():
        with gr.Column():
            with gr.Accordion("Image to Video (optional)", open=False):
                i2v_image = gr.Image(type="pil", label="Input Image")
            t2v_prompt = gr.Textbox(label="Prompt")
            with gr.Accordion("Advanced settings", open=False):
                t2v_duration = gr.Slider(minimum=1, maximum=3 if is_canonical else 10, value=3 if is_canonical else 5, step=1, label="Duration (seconds)", visible=not is_canonical)
                t2v_fps = gr.Slider(minimum=8, maximum=24, step=16, value=8 if is_canonical else 24, label="Frames per second", visible=is_canonical)
                t2v_guidance_scale = gr.Slider(minimum=1, maximum=15, value=9, step=0.1, label="Guidance Scale")
                t2v_video_guidance_scale = gr.Slider(minimum=1, maximum=15, value=5, step=0.1, label="Video Guidance Scale")
            t2v_generate_btn = gr.Button("Generate Video")
        with gr.Column():
            t2v_output = gr.Video(label=f"Generated Video")
            gr.HTML("""
                <div style="display: flex; flex-direction: column;justify-content: center; align-items: center; text-align: center;">
                    <p style="display: flex;gap: 6px;">
                         <a href="https://huggingface.co/spaces/Pyramid-Flow/pyramid-flow?duplicate=true">
                            <img src="https://huggingface.co/datasets/huggingface/badges/resolve/main/duplicate-this-space-lg.svg" alt="Duplicate this Space">
                        </a>
                    </p>
                    <p>to use privately and generate videos up to 10s at 24fps</p>
                </div>
                """)
    gr.Examples(
        examples=[
            "A movie trailer featuring the adventures of the 30 year old space man wearing a red wool knitted motorcycle helmet, blue sky, salt desert, cinematic style, shot on 35mm film, vivid colors",
            "Beautiful, snowy Tokyo city is bustling. The camera moves through the bustling city street, following several people enjoying the beautiful snowy weather and shopping at nearby stalls. Gorgeous sakura petals are flying through the wind along with snowflakes"
        ],
        fn=generate_video,
        inputs=t2v_prompt,
        outputs=t2v_output,
        cache_examples=True,
        cache_mode="lazy"
    )
    t2v_generate_btn.click(
        generate_video,
        inputs=[t2v_prompt, i2v_image, t2v_duration, t2v_guidance_scale, t2v_video_guidance_scale, t2v_fps],
        outputs=t2v_output
    )

demo.launch()