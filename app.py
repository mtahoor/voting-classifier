import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet_v2 import preprocess_input
import numpy as np

physical_devices = tf.config.experimental.list_physical_devices('CPU')
tf.config.experimental.set_visible_devices(physical_devices[0], 'CPU')

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained models
model_v5 = load_model("model_v5.h5")
model_v6 = load_model("model_v6.h5")
model_v7 = load_model("model_v7.h5")

# Define the input image size expected by the models
input_shape = (224, 224)

# Define class labels
class_labels = ["class1", "class2", "class3"]

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess and predict image for each model
def predict_image(model, image_path):
    """
    Preprocesses and predicts the class label for an image using a given model.
    
    Args:
        model (keras.Model): The pre-trained model.
        image_path (str): The path to the image file.
        
    Returns:
        np.ndarray: Predicted probabilities for each class.
    """
    img = image.load_img(image_path, target_size=input_shape)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    predictions = model.predict(img_array)
    return predictions

# Function to combine predictions using voting classifier
def combine_predictions(predictions):
    """
    Combines predictions from multiple models using a voting scheme.
    
    Args:
        predictions (list): List of predicted probabilities for each class.
        
    Returns:
        int: Index of the majority class.
    """
    return np.argmax(np.sum(predictions, axis=0))

# Function to predict image using ensemble of models
def predict_image_ensemble(image_path):
    """
    Predicts the class label for an image using an ensemble of pre-trained models.
    
    Args:
        image_path (str): The path to the image file.
        
    Returns:
        int: Index of the predicted class label.
    """
    predictions = []
    predictions.append(predict_image(model_v5, image_path))
    predictions.append(predict_image(model_v6, image_path))
    predictions.append(predict_image(model_v7, image_path))
    combined_prediction_index = combine_predictions(predictions)
    return combined_prediction_index

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/page1.html")
def page1():
    return render_template("page1.html")

@app.route("/page2.html", methods=['GET', 'POST'])
def page2():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template("page2.html", prediction="No file part")

        file = request.files['file']

        if file.filename == '':
            return render_template("page2.html", prediction="No selected file")

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            prediction_index = predict_image_ensemble(filepath)
            prediction_label = class_labels[prediction_index]
            print(prediction_label)
            # Since the page3.html template doesn't exist, we render the page2.html again to ensure that we don't get any errors from this section.
            return render_template("page2.html", prediction=prediction_label)
        else:
            return render_template("page2.html", prediction="Invalid file type")

    return render_template("page2.html")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)
