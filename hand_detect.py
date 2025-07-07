import cv2
import numpy as np
import mediapipe as mp
import tflite_runtime.interpreter as tflite  # 라즈베리파이에서는 이걸 사용!

# -----------------------------
# TFLite 모델 로드
# -----------------------------
interpreter = tflite.Interpreter(model_path="hand_model_changed.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -----------------------------
# 클래스 이름
# -----------------------------
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
        # 홀수 프레임은 skip → 그래도 마지막 라벨은 표시
        cv2.putText(
            frame,
            last_label,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),  # 노란색
            2,
        )
        cv2.imshow("Smart Hand Control (TFLite)", frame)
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

            # TFLite 입력에 맞게 전처리
            hand_input = cv2.resize(hand_img, (128, 128))
            hand_input = hand_input.astype(np.float32) / 255.0
            hand_input = np.expand_dims(hand_input, axis=0)

            # TFLite 추론
            interpreter.set_tensor(input_details[0]["index"], hand_input)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]["index"])[0]

            class_idx = np.argmax(preds)
            confidence = preds[class_idx]

            if confidence > 0.7:
                last_label = f"{class_names[class_idx]} ({confidence:.2f})"
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 항상 마지막 라벨 표시
    cv2.putText(
        frame,
        last_label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
    )

    cv2.imshow("Smart Hand Control (TFLite)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
