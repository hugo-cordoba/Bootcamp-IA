from flask import Flask, request, render_template

app = Flask(__name__)

from models.generar_imagenes import generate_image
from models.analisis_comentarios import load_comments



@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html'), 500

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', image_path=None)


@app.route('/generar-imagenes', methods=['GET', 'POST'])
def generar_imagenes():
    if request.method == 'POST':
        prompt = request.form['prompt']
        seed = request.form.get('seed')
        image_path = generate_image(prompt, seed)
        return render_template('index.html', image_path=image_path, prompt=prompt, active_section='generar-imagenes')
    return render_template('index.html', image_path=None, active_section='generar-imagenes')

@app.route('/analisis-comentarios', methods=['GET', 'POST'])
def analisis_comentarios():
    if request.method == 'POST':
        instagram_url = request.form['instagram_url']
        instagram_comments = load_comments(instagram_url)
        return render_template('index.html', instagram_comments=instagram_comments, active_section='analisis-comentarios')
    return render_template('index.html', instagram_comments=None, active_section='analisis-comentarios')


@app.route('/recomendacion-hastags', methods=['GET', 'POST'])
def recomendacion_hastags():
    if request.method == 'POST':
        # Procesa y analiza comentarios aquí
        pass
    return render_template('index.html', image_path=None)

if __name__ == '__main__':
    app.run(debug=True)
