import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

# 🔥 SETTINGS (FAST TRAINING)
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 3

# 📂 DATA GENERATOR
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    "dataset_frames",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_data = datagen.flow_from_directory(
    "dataset_frames",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# 🧠 LOAD MODEL
base_model = MobileNetV2(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# ❄️ FREEZE BASE MODEL
for layer in base_model.layers:
    layer.trainable = False

# ⚙️ COMPILE
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 🚀 TRAIN (LIMITED DATA MODE)
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    steps_per_epoch=20,      # 🔥 LIMIT TRAINING
    validation_steps=10      # 🔥 LIMIT VALIDATION
)

# 💾 SAVE MODEL
os.makedirs("models", exist_ok=True)
model.save("models/deepfake_model.h5")

# 📊 SAVE GRAPH
os.makedirs("outputs/graphs", exist_ok=True)

plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Training vs Validation Accuracy")
plt.savefig("outputs/graphs/accuracy.png")
plt.show()

print("✅ Training Completed!")