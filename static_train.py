import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib

#load csv file
df = pd.read_csv("gesture_data.csv")

X = df.drop("label", axis=1).values
y = df["label"].values

#encoder label
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
joblib.dump(encoder, "static_encoder.pkl")

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

#build model
model = Sequential([
    Dense(256, activation='relu', input_shape=(63,)),
    Dense(256, activation='relu'),
    Dense(256, activation='relu'),
    Dense(len(np.unique(y_encoded)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#train model
model.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_test, y_test),
)

#save model and encoder
model.save("static_gesture_model.keras")
print("🎉 Model trained and saved as gesture_model.keras")
print("🧠 Label encoder saved as static_label_encoder.pkl")