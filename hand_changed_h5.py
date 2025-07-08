import cv2
import numpy as np
import threading
import time
import requests
from tensorflow.keras.models import load_model
from ultralytics import YOLO

# --- 부저 설정 ---
try:
    from gpiozero import TonalBuzzer
    buzzer = TonalBuzzer(13)
    IS_RASPBERRY_PI = True
    print("라즈베리 파이 환경으로 설정되었습니다. 부저를 사용합니다.")
except (ImportError, Exception):
    print("알림: 라즈베리 파이 환경이 아닙니다. 부저 대신 콘솔에 메시지를 출력합니다.")
    IS_RASPBERRY_PI = False

# --- GPIO 및 스피커 전송 함수 ---
def send_to_gpio(cmd):
    try:
        res = requests.post("http://localhost:5000/control", json={"cmd": cmd})
        print(f"📡 GPIO 서버 응답({cmd}):", res.text)
    except:
        print(f"❌ GPIO 전송 실패 ({cmd})")

def send_to_speaker(cmd):
    try:
        speaker_url = "http://10.10.15.167:8000/notify"
        res = requests.post(speaker_url, json={"cmd": cmd})
        print(f"🔊 스피커 응답({cmd}):", res.text)
    except Exception as e:
        print(f"❌ 스피커 전송 실패 ({cmd}):", e)

# --- 손 제스처 인식 스레드 ---
def hand_gesture_thread():
    model = load_model("hand_model.h5")
    class_names = ["fan_off", "fan_on", "light_off", "light_on"]

    import mediapipe as mp
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, model_complexity=0)
    mp_draw = mp.solutions.drawing_utils

    # 카메라 자동 감지
    cap = None
    for i in range(3):
        test_cap = cv2.VideoCapture(i)
        if test_cap.isOpened():
            print(f"✅ 카메라 {i} 사용 가능")
            cap = test_cap
            break
    if cap is None:
        print("❌ 카메라 열기 실패! 연결 상태 확인")
        return

    last_command_sent = None
    last_command_timestamp = time.time()
    command_debounce_time = 2
    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 프레임 읽기 실패")
            continue

        frame_counter += 1
        if frame_counter % 3 != 0:
            cv2.imshow("Hand Gesture Camera (0)", frame)
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

                hand_input = cv2.resize(hand_img, (128, 128)) / 255.0
                hand_input = np.expand_dims(hand_input, axis=0)

                preds = model.predict(hand_input, verbose=0)
                class_idx = np.argmax(preds[0])
                confidence = preds[0][class_idx]

                if confidence > 0.8:
                    gesture = class_names[class_idx]
                    current_time = time.time()
                    label = f"{gesture} ({confidence:.2f})"
                    cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    if (gesture != last_command_sent) or (current_time - last_command_timestamp > command_debounce_time):
                        cmd = ""
                        if gesture == "fan_on":
                            cmd = "motor_on"
                        elif gesture == "fan_off":
                            cmd = "motor_off"
                        elif gesture == "light_on":
                            cmd = "light_on"
                        elif gesture == "light_off":
                            cmd = "light_off"

                        if cmd:
                            send_to_gpio(cmd)
                            threading.Thread(target=send_to_speaker, args=(cmd,), daemon=True).start()
                            last_command_sent = gesture
                            last_command_timestamp = current_time

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Hand Gesture Camera (0)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# --- 프로그램 시작점 ---
if __name__ == "__main__":
    hand_gesture_thread()
