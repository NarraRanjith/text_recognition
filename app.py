import cv2
from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import keras
import numpy as np
from werkzeug.utils import secure_filename
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Lazy load model
model = None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    global model
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)

        try:
            print("a"*100)
            model = keras.models.load_model('emnist_cnnmodel87.keras')
            print("c"*100)
            input_image = cv2.imread(path)
            gray_scale = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
            resized_image = cv2.resize(gray_scale, (28, 28))
            if(resized_image[0][0]==255):
                resized_image=abs(resized_image-255.0)
            normalized = resized_image.astype('float32') / 255.0
            print("b"*100)
            input_image = normalized.reshape(1, 28, 28)
            prediction = model.predict(input_image)
            confidence = float(prediction.max())
            label = np.argmax(prediction)
            if(label>9 and label<36):
                label=chr(label+55)
            elif(label>=36 and label<62):
                label=chr(label+61)

            print(f"Predicted: {label} with confidence {confidence:.2f}")
        except Exception as e:
            # Render the page with an error message instead of raising 500
            return render_template('index.html', filename=filename, error=str(e))
        
        return render_template('index.html', filename=filename, label=label, confidence=confidence)

    return redirect(url_for('index'))


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
