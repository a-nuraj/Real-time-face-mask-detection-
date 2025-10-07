import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
DATASET_DIR = "dataset" 
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10                        
LEARNING_RATE = 1e-4
NUM_CLASSES = 2
SAVED_MODEL_PATH = os.path.join("saved_models", "mask_detector.h5")
os.makedirs("saved_models", exist_ok=True)

train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest",
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
preds = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs=base.input, outputs=preds)

for layer in base.layers:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss="categorical_crossentropy", metrics=["accuracy"])
model.summary()


checkpoint = ModelCheckpoint(SAVED_MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)


history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=max(1, val_generator.samples // BATCH_SIZE),
    epochs=EPOCHS,
    callbacks=[checkpoint, reduce_lr]
)

# Optionally fine-tune: unfreeze some of base and train a few more epochs
for layer in base.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5), loss="categorical_crossentropy", metrics=["accuracy"])

fine_tune_epochs = 5
history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_data=val_generator,
    validation_steps=max(1, val_generator.samples // BATCH_SIZE),
    epochs=fine_tune_epochs,
    callbacks=[checkpoint, reduce_lr]
)

print(f"Training finished. Best model saved to: {SAVED_MODEL_PATH}")