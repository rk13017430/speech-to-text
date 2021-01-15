import os
from flask import Flask, request
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io


app = Flask(__name__)
model = None
N_DOC = 2
IMG_SIZE = 256
class_indices = {0: 'aadhar', 1: 'pan'}
MODEL_DIR = os.path.join('..', 'model')
MODEL_NAME = 'vgg16.h5'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)


def get_model():
	global model
	model = load_model(MODEL_PATH)
	print(" * Model loaded!")
print(" * Loading model...")
get_model()


# Process image and predict label
def preprocess_image(image, target_size):
	if image.mode != "RGB":
		image = image.convert("RGB")
	image = image.resize(target_size)
	image = img_to_array(image)

	image = np.expand_dims(image, axis=0)
	return image

@app.route('/predict', methods=['POST'])
def upload_file():
	response = {'success': False}
	if request.files.get('file'):
		print(' * Request Received')
		img_requested = request.files['file']
		image = Image.open(img_requested.stream)
		processed_image = preprocess_image(image, target_size=(IMG_SIZE, IMG_SIZE))

		prediction = np.argmax(model.predict(processed_image), axis=1)
		prediction_class = class_indices[prediction[0]]

		print(prediction_class)
	return prediction_class

if __name__ == "__main__":
	app.run()

