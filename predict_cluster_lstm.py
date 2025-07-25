import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("window_cluster_mapping.csv")

# Parse flattened windows into numpy arrays
X = df["flattened_window"].apply(lambda x: np.array(eval(x))).to_list()
X = np.stack(X)

# Reshape each sample: (num_samples, window_size, num_features)
WINDOW_SIZE = 10
NUM_FEATURES = X.shape[1] // WINDOW_SIZE
X = X.reshape((-1, WINDOW_SIZE, NUM_FEATURES))

# Encode cluster IDs
le = LabelEncoder()
y = le.fit_transform(df["cluster_id"])
y_cat = to_categorical(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42, stratify=y)

# LSTM model
model = Sequential([
    Masking(mask_value=0.0, input_shape=(WINDOW_SIZE, NUM_FEATURES)),
    LSTM(64, return_sequences=False),
    Dense(64, activation='relu'),
    Dense(y_cat.shape[1], activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=30, batch_size=32)

# Plot accuracy
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title("Training and Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


model.save("saved_models/lstm_model.h5")