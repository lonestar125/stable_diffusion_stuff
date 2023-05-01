
import torch
import numpy as np
from PIL import Image
from diffusers import DiffusionPipeline, EulerDiscreteScheduler

device = "cuda" if torch.cuda.is_available() else "cpu"

scheduler = EulerDiscreteScheduler.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="scheduler", prediction_type="v_prediction")
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", scheduler=scheduler)
pipe = pipe.to(device)

prompt = "a realistic photo of a camera"
image = pipe(prompt).images[0]


image.save("cpu.png")
