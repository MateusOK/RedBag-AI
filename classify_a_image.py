from keras.api.models import load_model
from keras.api.preprocessing import image
import numpy as np
from keras.api.applications.vgg16 import preprocess_input

model = load_model("path_to_model_trained")

img_path = "path_to_image_to_analize"

img_height, img_width = 224, 224
img = image.load_img(img_path, target_size=(img_height, img_width))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

predictions = model.predict(img_array)