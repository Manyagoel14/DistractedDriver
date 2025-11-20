import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

# ======================================
# PATHS
# ======================================
BASE_DIR = "/content/drive/MyDrive/Distracted-Driver-Detection_1"
NPY_DIR = f"{BASE_DIR}/npy_files"
MODEL_DIR = f"{BASE_DIR}/model/self_trained"
PICKLE_DIR = f"{BASE_DIR}/pickle_files"

os.makedirs(MODEL_DIR, exist_ok=True)

# GPU
if tf.config.list_physical_devices('GPU'):
    keras.mixed_precision.set_global_policy('mixed_float16')

# ======================================
# LOAD PREPROCESSED DATA
# ======================================
X = np.load(f"{NPY_DIR}/X.npy")
y = np.load(f"{NPY_DIR}/y.npy")

with open(f"{PICKLE_DIR}/labels_map.pkl", "rb") as f:
    labels_map = pickle.load(f)

labels_list = list(labels_map.keys())
y_cat = to_categorical(y)

# train split
Xtrain, Xval, ytrain, yval = train_test_split(X, y_cat, test_size=0.2, random_state=42)

# ======================================
# MODEL
# ======================================
model = Sequential([
    Input(shape=(128, 128, 3)),

    Conv2D(64, 2, padding='same', activation='relu'),
    MaxPooling2D(2),

    Conv2D(128, 2, padding='same', activation='relu'),
    MaxPooling2D(2),

    Conv2D(256, 2, padding='same', activation='relu'),
    MaxPooling2D(2),

    Conv2D(512, 2, padding='same', activation='relu'),
    MaxPooling2D(2),

    Dropout(0.5),
    Flatten(),

    Dense(500, activation='relu'),
    Dropout(0.5),

    Dense(len(labels_list), activation='softmax', dtype='float32')
])

model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

checkpoint = ModelCheckpoint(
    f"{MODEL_DIR}/best_model.keras",
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

# ======================================
# TRAIN
# ======================================
history = model.fit(
    Xtrain, ytrain,
    validation_data=(Xval, yval),
    epochs=25,
    batch_size=40,
    callbacks=[checkpoint]
)

# ======================================
# CONFUSION MATRIX
# ======================================
ypred = np.argmax(model.predict(Xval), axis=1)
ytrue = np.argmax(yval, axis=1)

cm = confusion_matrix(ytrue, ypred)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, cmap="Blues")
plt.show()

print("Accuracy:", accuracy_score(ytrue, ypred))
