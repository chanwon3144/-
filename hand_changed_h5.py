import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import requests
import time

# -----------------------------
# 모델 로드 및 클래스
# -----------------------------
model = load_model("hand_change.h5")
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
# GPIO 서버로 명령 전송 함수
# -----------------------------
def send_to_gpio_server(cmd):
    try:
        url = "http://localhost:5000/control"
        res = requests.post(url, json={"cmd": cmd})
        print(f"📡 명령 전송됨: {cmd}, 응답: {res.text}")
    except requests.exceptions.ConnectionError:
        print("❌ GPIO 서버 연결 실패. 'gpio_server.py' 실행 확인")
    except Exception as e:
        print(f"❌ 전송 실패 ({cmd}):", e)

# -----------------------------
# 제스처 상태 추적 변수
# -----------------------------
last_command_sent = None
last_command_timestamp = time.time()
command_debounce_time = 2  # 초

# -----------------------------
# 웹캠 열기
# -----------------------------
cap = cv2.VideoCapture(0)  # 외장카메라면 1, 2로 바꿔보세요

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

            # 랜드마크에서 bounding box 계산
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

            # MobileNetV2 입력 사이즈 & 정규화
            hand_input = cv2.resize(hand_img, (128, 128))
            hand_input = hand_input / 255.0
            hand_input = np.expand_dims(hand_input, axis=0)

            # 예측
            preds = model.predict(hand_input, verbose=0)
            class_idx = np.argmax(preds[0])
            confidence = preds[0][class_idx]

            if confidence > 0.8:
                gesture = class_names[class_idx]
                label = f"{gesture} ({confidence:.2f})"

                current_time = time.time()
                if (gesture != last_command_sent) or (current_time - last_command_timestamp > command_debounce_time):
                    if gesture == "fan_on":
                        send_to_gpio_server("motor_on")
                    elif gesture == "fan_off":
                        send_to_gpio_server("motor_off")
                    elif gesture == "light_on":
                        send_to_gpio_server("light_on")
                    elif gesture == "light_off":
                        send_to_gpio_server("light_off")

                    last_command_sent = gesture
                    last_command_timestamp = current_time

                # 표시
                cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            # 랜드마크 그리기
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 화면 출력
    cv2.imshow("Smart Hand Control (MobileNetV2)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
