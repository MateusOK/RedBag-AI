import os
import tensorflow as tf
from keras.api.utils import image_dataset_from_directory

# Ensure TensorFlow uses GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Use relative paths
data_dir = os.path.join(os.path.dirname(__file__), "data")
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

# Define parameters for image preprocessing
img_height, img_width = 224, 224
batch_size = 32

# Load and preprocess the image data
train_dataset = image_dataset_from_directory(
    train_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical',
    validation_split=0.2,
    subset='training',
    seed=123
)

validation_dataset = image_dataset_from_directory(
    train_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical',
    validation_split=0.2,
    subset='validation',
    seed=123
)

test_dataset = image_dataset_from_directory(
    test_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical'
)

# Extract class names before applying transformations
class_names = test_dataset.class_names

# Extract file paths for test dataset
file_paths = test_dataset.file_paths
