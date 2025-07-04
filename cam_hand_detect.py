import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# -----------------------------
# 모델 로드 및 클래스
# -----------------------------
model = load_model("hand_mobilenet_model.h5")
class_names = ["fan_off", "fan_on", "light_off", "light_on"]

# -----------------------------
# Mediapipe hands 초기화
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# -----------------------------
# 웹캠 열기
# -----------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_min, y_min = w, h
            x_max, y_max = 0, 0

            # 랜드마크에서 bounding box 구하기
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)

            pad = 20
            x_min = max(x_min - pad, 0)
            y_min = max(y_min - pad, 0)
            x_max = min(x_max + pad, w)
            y_max = min(y_max + pad, h)

            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:
                continue

            # 모델 입력에 맞게 resize 및 normalize
            hand_input = cv2.resize(hand_img, (128, 128))
            hand_input = hand_input / 255.0
            hand_input = np.expand_dims(hand_input, axis=0)

            # 예측
            preds = model.predict(hand_input, verbose=0)
            print(f"Predictions: {preds[0]}")
            class_idx = np.argmax(preds[0])
            confidence = preds[0][class_idx]

            # 0.7 이상만 출력
            if confidence > 0.4:
                label = class_names[class_idx]
                cv2.putText(
                    frame,
                    f"{label} ({confidence:.2f})",
                    (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # 랜드마크 그리기
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 출력
    cv2.imshow("Smart Hand Control (Realtime Test)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
