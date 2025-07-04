from google.colab import files
import zipfile
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    BatchNormalization,
    Rescaling,
)
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------------------------
# 📌 Colab에서 ZIP 업로드
# ------------------------------------------
uploaded = files.upload()

# 업로드된 파일명 확인 (하나만 업로드했을 때)
for fname in uploaded.keys():
    zip_path = os.path.join("/content", fname)
    break

# ------------------------------------------
# 📌 압축 풀기
# ------------------------------------------
extract_path = "/content/ai_project"  # zip을 풀 곳

with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(extract_path)

print("압축 해제 완료:", os.listdir(extract_path))

# 🚀 실제 클래스 폴더가 있는 경로로 지정
dataset_path = os.path.join(
    extract_path, "train_set"
)  # <= train_set 안에 클래스 폴더가 있으므로

train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(128, 128),
    batch_size=32,
    label_mode="categorical",
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(128, 128),
    batch_size=32,
    label_mode="categorical",
)


# ------------------------------------------
# 📌 정규화
# ------------------------------------------
normalization_layer = Rescaling(1.0 / 255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# ------------------------------------------
# 📌 CNN 모델
# ------------------------------------------
model = Sequential(
    [
        Conv2D(
            32, (3, 3), activation="relu", padding="same", input_shape=(128, 128, 3)
        ),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.3),
        # 추가
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2, 2),
        Dropout(0.3),
        Flatten(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(128, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(4, activation="softmax"),
    ]
)


model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# ------------------------------------------
# 📌 학습
# ------------------------------------------
early_stop = EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True)
model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=[early_stop])

# ------------------------------------------
# 📌 저장
# ------------------------------------------
model.save("hand_model_split_auto2.h5")
