from diffusers import AutoPipelineForText2Image
import os
import torch

from app import app

model = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to("cuda")

def generate_image(prompt, seed):        
        if seed != '':
            seed = int(seed)
        else:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()

        generator = torch.Generator("cuda").manual_seed(seed)
        
        image_dir = os.path.join(app.root_path, 'static', 'images')

        os.makedirs(image_dir, exist_ok=True)
        filename = f"{prompt[1].replace(' ', '_')}_{seed}.png"
        image_path = os.path.join(image_dir, filename)
        image = model(prompt, generator=generator, num_inference_steps=2, guidance_scale=0.0).images[0]
        image.save(image_path)
        
        return f'images/{filename}'
