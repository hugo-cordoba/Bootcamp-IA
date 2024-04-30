from flask import Flask, request, render_template
from diffusers import StableDiffusionPipeline
import os
import torch

app = Flask(__name__)
model = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda")

@app.route('/', methods=['GET'])
def home():
    # Redirige a generar_imagenes por defecto
    return render_template('index.html', image_path=None)

@app.route('/generar-imagenes', methods=['GET', 'POST'])
def generar_imagenes():
    if request.method == 'POST':
        prompt = request.form['prompt']
        seed = request.form.get('seed')
        
        if seed is not '':
            seed = int(seed)
        else:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()

        generator = torch.Generator("cuda").manual_seed(seed)
        
        image_dir = os.path.join(app.root_path, 'static', 'images')
        os.makedirs(image_dir, exist_ok=True)
        filename = f"{prompt[10].replace(' ', '_')}_{seed}.png"
        image_path = os.path.join(image_dir, filename)
        image = model(prompt, generator=generator).images[0]
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
