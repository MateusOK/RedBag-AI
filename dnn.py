import os
import tensorflow as tf
from keras.api.utils import image_dataset_from_directory
from keras.api.models import Sequential
from keras.api.layers import RandomFlip, RandomRotation, RandomZoom
from keras.api.layers import Rescaling, Dense, Dropout, GlobalAveragePooling2D
from keras.api.applications import VGG16
from keras.api.callbacks import ProgbarLogger

physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

data_dir = os.path.join(os.path.dirname(__file__), "data")
train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "test")

img_height, img_width = 224, 224
batch_size = 32

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

class_names = test_dataset.class_names

file_paths = test_dataset.file_paths

data_augmentation = Sequential([
    RandomFlip('horizontal'),
    RandomRotation(0.2),
    RandomZoom(0.2)
])

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=AUTOTUNE
)

train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 2

base_model = VGG16(input_shape=(img_height, img_width, 3), include_top=False, weights='imagenet')

base_model.trainable = False

model = Sequential([
    Rescaling(1./255, input_shape=(img_height, img_width, 3)),
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

num_epochs = 20

history = model.fit(train_dataset, validation_data=validation_dataset, epochs=num_epochs, callbacks=[ProgbarLogger()])