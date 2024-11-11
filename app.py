import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing.image import img_to_array
from io import BytesIO

app = Flask(__name__)

model = load_model("D:\\PLANT_Disease\\plant_disease_detection-main\\plant_disease_detection-main\\model.h5")
print('Model loaded. Check http://127.0.0.1:5000/')

labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

def getResult(img):
    img = img.resize((225, 225))  # Resize image to target size
    x = img_to_array(img)
    x = x.astype('float32') / 255.
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)[0]
    return predictions

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        
        # Load image directly from memory
        img = Image.open(BytesIO(f.read()))
        
        predictions = getResult(img)
        predicted_label = labels[np.argmax(predictions)]
        
        return str(predicted_label)
    return None

if __name__ == '__main__':
    app.run(debug=True)
