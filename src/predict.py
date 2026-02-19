import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json
import tensorflow.keras.backend as K

# Define focal loss again
def focal_loss(gamma=2., alpha=.25):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, 1e-7, 1-1e-7)
        ce = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1-y_pred, gamma)
        return K.sum(weight * ce, axis=1)
    return loss

# Load model with custom loss
model = tf.keras.models.load_model(
    "model/plant_model.h5",
    custom_objects={'loss': focal_loss()}
)

with open("model/class_indices.json") as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(160,160))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)

    idx = np.argmax(predictions)
    confidence = float(np.max(predictions))

    label = idx_to_class[idx].replace("_", " ")

    return label, confidence
