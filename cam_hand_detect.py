import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# -----------------------------
# 모델 로드 및 클래스
# -----------------------------
model = load_model("hand_mobilenet_model_last.h5")
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

frame_count = 0
last_label = "No hand detected"

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1
    if frame_count % 2 != 0:
        # 홀수 프레임은 skip → 그래도 마지막 라벨은 출력
        cv2.putText(
            frame,
            last_label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),  # 노란색
            2,
        )
        cv2.imshow("Smart Hand Control (Realtime Test)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
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
            class_idx = np.argmax(preds[0])
            confidence = preds[0][class_idx]

            # 0.7 이상만 라벨 갱신
            if confidence > 0.7:
                last_label = f"{class_names[class_idx]} ({confidence:.2f})"
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # 랜드마크 그리기
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 항상 마지막 라벨 표시
    cv2.putText(
        frame,
        last_label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),  # 노란색
        2,
    )

    cv2.imshow("Smart Hand Control (Realtime Test)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

