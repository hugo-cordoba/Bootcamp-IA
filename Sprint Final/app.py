from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

import os

app = Flask(__name__)

from models.generar_imagenes import generate_image
from models.analisis_comentarios import load_comments
from models.analisis_comentarios import calculate_percentage
from models.detectar_objetos import detectar_objetos_en_imagen 
from models.detectar_objetos import get_message 



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
        comments, sentiment_count = load_comments(instagram_url)
        percentages, most_frequent = calculate_percentage(sentiment_count)

        return render_template('index.html', instagram_info=comments, percentages=percentages, most_frequent=most_frequent, active_section='analisis-comentarios', instagram_url=instagram_url)
    return render_template('index.html', instagram_info=None, active_section='analisis-comentarios')

@app.route('/recomendacion-hastags', methods=['GET', 'POST'])
def recomendacion_hastags():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', image_path=None)
        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', image_path=None)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.root_path, 'static', 'images', filename)
            file.save(filepath)

            # Llama a la función de detección de objetos
            objects_detected = detectar_objetos_en_imagen(filepath)
            
            # Esta ruta debe ser relativa a 'static'
            uploaded_image_path = f'images/{filename}'
            
            obtener_hashtags = get_message(objects_detected)
            
            return render_template('index.html', objects_detected=objects_detected, file_path=uploaded_image_path, hashtags=obtener_hashtags, active_section='recomendacion-hastags')
    return render_template('index.html', image_path=None, active_section='recomendacion-hastags')

if __name__ == '__main__':
    app.run(debug=True)
