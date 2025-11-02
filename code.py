# ===========================
# Emotion Detection - MobileNetV2 (5 Classes)
# ===========================

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import os

# ---------------------------
# 1. Dataset Path
# ---------------------------
dataset_dir = r"C:\Users\james\.vscode\face"  # <-- make sure this path has 5 folders

# ---------------------------
# 2. Data Preprocessing
# ---------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

img_size = (224, 224)
batch_size = 32

train_data = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    class_mode='categorical',
    batch_size=batch_size,
    subset='training'
)

val_data = datagen.flow_from_directory(
    dataset_dir,
    target_size=img_size,
    class_mode='categorical',
    batch_size=batch_size,
    subset='validation'
)

# ---------------------------
# 3. Base Model: MobileNetV2
# ---------------------------
base_model = MobileNetV2(
    weights='imagenet', 
    include_top=False, 
    input_shape=(224, 224, 3)
)

# Freeze base model layers
for layer in base_model.layers:
    layer.trainable = False

# ---------------------------
# 4. Custom Layers
# ---------------------------
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(5, activation='softmax')(x)  # ✅ 5 output classes (Angry, Happy, Neutral, Sad, Surprise)

model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ---------------------------
# 5. Train the Model
# ---------------------------
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15
)

# ---------------------------
# 6. Save the Model
# ---------------------------
os.makedirs("model", exist_ok=True)
model.save("model/emotion5.h5")
print("✅ Model saved as 'model/emotion5.h5'")

# ---------------------------
# 7. Evaluate
# ---------------------------
val_loss, val_acc = model.evaluate(val_data)
print(f"✅ Validation Accuracy: {val_acc*100:.2f}%")
