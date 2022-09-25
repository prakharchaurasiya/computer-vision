from flask import Flask, render_template, request
import numpy as np
from tensorflow import keras
import cv2
model = keras.models.load_model('model/MNIST.h5')

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"].read()
    if file:
        image = np.asarray(bytearray(file), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image,(28,28))
        prediction = np.argmax(model.predict(image.reshape(1,28,28)), axis=1).item()
        result = f'The number in image is {prediction}.'
        return render_template("predict.html", prediction=result)
    return render_template("error.html")
