import torch
from diffusers import DiffusionPipeline, EulerDiscreteScheduler

device = "cuda" if torch.cuda.is_available() else "cpu"

#scheduler = EulerDiscreteScheduler.from_pretrained("OfficialUnstableDiffusion/UnstablePhotoRealv.5")
pipe = DiffusionPipeline.from_pretrained("OfficialUnstableDiffusion/UnstablePhotoRealv.5 ")
pipe = pipe.to(device)

prompt = ""
image = pipe(prompt).images[0]


image.save("cpu.png")