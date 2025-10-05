import tensorflow as tf
import numpy as np

# Create a very simple CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy')
model.save('model.h5')

# Create a corresponding class labels file
class_labels = [
    "Tomato Early Blight",
    "Apple Scab",
    "Healthy",
    "Unknown"
]
with open('class_labels.txt', 'w') as f:
    for label in class_labels:
        f.write(f"{label}\n")

print("✅ Dummy model.h5 and class_labels.txt created successfully!")