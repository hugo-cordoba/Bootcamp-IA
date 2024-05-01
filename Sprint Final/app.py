from flask import Flask, request, render_template
# from diffusers import StableDiffusionPipeline
# from diffusers import DiffusionPipeline
from diffusers import AutoPipelineForText2Image
import os
import torch

app = Flask(__name__)
# model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda")
# model = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to("cuda")
model = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16").to("cuda")



@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html'), 500

@app.route('/', methods=['GET'])
def home():
    # Redirige a generar_imagenes por defecto
    return render_template('index.html', image_path=None)

@app.route('/generar-imagenes', methods=['GET', 'POST'])
def generar_imagenes():
    if request.method == 'POST':
        prompt = request.form['prompt']
        seed = request.form.get('seed')
        
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
        
        return render_template('index.html', image_path=f'images/{filename}', prompt=prompt)

    return render_template('index.html', image_path=None)


@app.route('/analisis-comentarios', methods=['GET', 'POST'])
def analisis_comentarios():
    if request.method == 'POST':
        # Procesa y analiza comentarios aquí
        pass
    return render_template('index.html', image_path=None)

if __name__ == '__main__':
    app.run(debug=True)
