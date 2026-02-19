import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
from sklearn.metrics import confusion_matrix
import os

IMG_SIZE = (160,160)

val_dir = "../dataset/val"

model = tf.keras.models.load_model("model/plant_model.h5", compile=False)

val_gen = ImageDataGenerator(rescale=1./255)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

preds = model.predict(val_data)
y_pred = np.argmax(preds, axis=1)
y_true = val_data.classes

cm = confusion_matrix(y_true, y_pred)

os.makedirs("model", exist_ok=True)
np.save("model/confusion_matrix.npy", cm)

print("Confusion matrix saved.")
