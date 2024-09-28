from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

# 加载预训练模型
model_id = "stabilityai\stable-diffusion-2-base"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# 加载并准备输入图像
input_image = Image.open("demo/corgi.png")

# 提示词设置为生成背面图像
prompt = "A corgi dog viewed from the left side, with similar coloration and markings as the reference image, including its fluffy tail, back legs, and red collar. The dog should be in a similar dynamic pose as the provided image. "

# 生成图像
with torch.autocast("cuda"):
    generated_images = pipe(prompt=prompt, init_image=input_image, strength=0.75, guidance_scale=7.5).images

# 保存生成的图像
output_image = generated_images[0]
output_image.save("demo/corgi_back_image.jpg")
