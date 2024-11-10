import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf

app = Flask(__name__)

# Load the TFLite model and allocate tensors
interpreter = tf.lite.Interpreter(model_path='compressed_model.tflite')

interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define the labels
labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

# Define a function to make predictions
def getResult(image_path):
    img = load_img(image_path, target_size=(225, 225))
    x = img_to_array(img)
    x = x.astype('float32') / 255.0
    x = np.expand_dims(x, axis=0)

    # Set the tensor to the input data
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()

    # Get the prediction result
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    return predictions

# Define the routes
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        
        # Ensure the uploads directory exists
        basepath = os.path.dirname(__file__)
        upload_folder = os.path.join(basepath, 'uploads')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)

        # Save the uploaded file
        file_path = os.path.join(upload_folder, secure_filename(f.filename))
        f.save(file_path)
        
        # Predict the result
        predictions = getResult(file_path)
        predicted_label = labels[np.argmax(predictions)]
        
        return str(predicted_label)
    return None

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)

