import os
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import joblib

DATASET_PATH = r"C:\Users\TAY YU LIN\OneDrive\Desktop\dataset"
SEQUENCE_LENGTH = 70
LANDMARKS_PER_FRAME = 63

X = []
y = []

print("📦 Loading data...")
for category in os.listdir(DATASET_PATH):
    category_path = os.path.join(DATASET_PATH, category)
    if not os.path.isdir(category_path):
        continue

    for label in os.listdir(category_path):
        label_folder = os.path.join(category_path, label)
        if not os.path.isdir(label_folder):
            continue

        for file in os.listdir(label_folder):
            if file.endswith(".json"):
                filepath = os.path.join(label_folder, file)
                with open(filepath, "r") as f:
                    sequence = json.load(f)

                if len(sequence) == SEQUENCE_LENGTH:
                    X.append(sequence)
                    y.append(label)

X = np.array(X)
y = np.array(y)

print(f"✅ Loaded {len(X)} sequences with shape: {X.shape}")

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
num_classes = len(encoder.classes_)

X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, shuffle=True, stratify=y)

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, LANDMARKS_PER_FRAME)),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("🧠 Training model...")
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_data=(X_val, y_val))

#save
model.save("motion_gesture_model.keras")
joblib.dump(encoder, "motion_label_encoder.pkl")
print("💾 Model and label encoder saved!")
