import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
import numpy as np
import os
import json
from sklearn.utils import class_weight
import tensorflow.keras.backend as K

IMG_SIZE = (160,160)
BATCH_SIZE = 16
EPOCHS = 20

train_dir = "../dataset/train"
val_dir = "../dataset/val"

# Strong augmentation
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.3,
    horizontal_flip=True,
    brightness_range=[0.7,1.3],
    width_shift_range=0.1,
    height_shift_range=0.1
)

val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_data = val_gen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

print("Class mapping:", train_data.class_indices)

# Compute class weights
labels = train_data.classes
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels),
    y=labels
)
class_weights = dict(enumerate(class_weights))

print("Class weights:", class_weights)

# Focal loss
def focal_loss(gamma=2., alpha=.25):
    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, 1e-7, 1-1e-7)
        ce = -y_true * K.log(y_pred)
        weight = alpha * K.pow(1-y_pred, gamma)
        return K.sum(weight * ce, axis=1)
    return loss

base_model = MobileNetV2(weights='imagenet', include_top=False,input_shape=(160,160,3))

for layer in base_model.layers:
    layer.trainable = False

x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)

output = layers.Dense(train_data.num_classes, activation='softmax')(x)

model = models.Model(base_model.input, output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss=focal_loss(),
    metrics=['accuracy']
)

model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    class_weight=class_weights
)

os.makedirs("../model", exist_ok=True)

model.save("../model/plant_model.h5")

with open("../model/class_indices.json","w") as f:
    json.dump(train_data.class_indices,f)

print("Balanced training complete.")

metrics = {
    "val_acc": float(history.history['val_accuracy'][-1]),
    "train_acc": float(history.history['accuracy'][-1]),
    "loss": float(history.history['loss'][-1])
}

with open("../model/metrics.json", "w") as f:
    json.dump(metrics, f)
