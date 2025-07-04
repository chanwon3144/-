from google.colab import files
import zipfile
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization,
    Rescaling,
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing import image_dataset_from_directory

# ------------------------------------------
# Colab-> ZIP 업로드
# ------------------------------------------
uploaded = files.upload()
for fname in uploaded.keys():
    zip_path = os.path.join("/content", fname)
    break

# ------------------------------------------
# 압축 풀기
# ------------------------------------------
extract_path = "/content/cam"
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)
print("압축 해제 완료:", os.listdir(extract_path))

# ------------------------------------------
# Dataset & Augmentation
# ------------------------------------------
dataset_path = os.path.join(extract_path, "train_set")

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 32
img_size = (128, 128)

train_ds = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="categorical",
)
val_ds = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size,
    label_mode="categorical",
)

# 데이터 augmentation pipeline
data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.3),
        tf.keras.layers.RandomBrightness(0.3),
    ]
)

# pipeline 적용
train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE
)

# 정규화
normalization_layer = Rescaling(1.0 / 255)
train_ds = train_ds.map(
    lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE
)
val_ds = val_ds.map(
    lambda x, y: (normalization_layer(x), y), num_parallel_calls=AUTOTUNE
)

train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ------------------------------------------
# MobileNetV2 모델
# ------------------------------------------
base_model = MobileNetV2(
    weights="imagenet", include_top=False, input_shape=(128, 128, 3)
)
base_model.trainable = False  # 처음에는 freeze

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
output = Dense(4, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# ------------------------------------------
# 콜백
# ------------------------------------------
early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.5, patience=4, verbose=1, min_lr=1e-6
)

# ------------------------------------------
# 학습
# ------------------------------------------
history = model.fit(
    train_ds, validation_data=val_ds, epochs=50, callbacks=[early_stop, reduce_lr]
)

# ------------------------------------------
# 저장
# ------------------------------------------
model.save("hand_mobilenet_model_last.h5")
files.download("hand_mobilenet_model_last.h5")
