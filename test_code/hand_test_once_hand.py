import cv2
import numpy as np
import threading
import time
import requests
from tensorflow.keras.models import load_model
from ultralytics import YOLO # 이 모듈은 hand_gesture_thread에서는 직접 사용되지 않지만, 전체 프로젝트 구성상 필요할 수 있으니 그대로 둡니다.

# --- 부저 설정 ---
try:
    from gpiozero import TonalBuzzer
    buzzer = TonalBuzzer(13)
    IS_RASPBERRY_PI = True
    print("라즈베리 파이 환경으로 설정되었습니다. 부저를 사용합니다.")
except (ImportError, Exception):
    print("알림: 라즈베리 파이 환경이 아닙니다. 부저 대신 콘솔에 메시지를 출력합니다.")
    IS_RASPberry_PI = False

# --- GPIO 및 스피커 전송 함수 ---
def send_to_gpio(cmd):
    try:
        res = requests.post("http://localhost:5000/control", json={"cmd": cmd})
        print(f"📡 GPIO 서버 응답({cmd}):", res.text)
    except requests.exceptions.ConnectionError: # 연결 오류만 특정하여 출력
        print(f"❌ GPIO 서버 연결 실패. '{cmd}' 명령 전송 불가.")
    except Exception as e:
        print(f"❌ GPIO 전송 실패 ({cmd}):", e)

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
    for i in range(3): # 0, 1, 2번 카메라 시도
        test_cap = cv2.VideoCapture(i)
        if test_cap.isOpened():
            print(f"✅ 카메라 {i} 사용 가능")
            cap = test_cap
            break
    if cap is None:
        print("❌ 카메라 열기 실패! 연결 상태 확인")
        return

    # --- 변경된 부분 시작 ---
    last_recognized_gesture = None # 마지막으로 성공적으로 인식되어 명령을 보낸 제스처
    # command_debounce_time 변수는 더 이상 필요 없음
    # --- 변경된 부분 끝 ---

    frame_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 프레임 읽기 실패")
            continue

        frame_counter += 1
        # 홀수 프레임에서만 처리하도록 수정 (CPU 과부하 감소 목적)
        if frame_counter % 3 != 0: # 3프레임 중 1번만 처리 (1/3 프레임만 처리)
            cv2.imshow("Hand Gesture Camera (0)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        current_gesture = None # 현재 프레임에서 인식된 제스처 (일단 없음으로 초기화)

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
                    current_gesture = class_names[class_idx] # 현재 프레임에서 인식된 제스처 저장
                    label = f"{current_gesture} ({confidence:.2f})"
                    cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # --- 변경된 핵심 로직 ---
        if current_gesture is not None: # 현재 프레임에서 유효한 제스처가 인식된 경우
            if current_gesture != last_recognized_gesture: # 마지막으로 보낸 명령과 다른 제스처인 경우
                cmd = ""
                if current_gesture == "fan_on":
                    cmd = "motor_on"
                elif current_gesture == "fan_off":
                    cmd = "motor_off"
                elif current_gesture == "light_on":
                    cmd = "light_on"
                elif current_gesture == "light_off":
                    cmd = "light_off"

                if cmd: # 명령어가 유효하면
                    send_to_gpio(cmd)
                    threading.Thread(target=send_to_speaker, args=(cmd,), daemon=True).start()
                    last_recognized_gesture = current_gesture # 마지막으로 보낸 제스처 업데이트
                    print(f"🎉 새로운 제스처 인식: {current_gesture}, 명령 전송: {cmd}")
            # else: # 동일한 제스처가 연속으로 인식되었지만, 이미 해당 명령이 전송된 상태
            #     print(f"DEBUG: 동일 제스처 '{current_gesture}' 반복 인식 (명령 미전송)")
        else: # 현재 프레임에서 아무 제스처도 인식되지 않거나 신뢰도가 낮은 경우
            if last_recognized_gesture is not None: # 이전에 인식된 제스처가 있었다면 초기화
                print("DEBUG: 손 제스처 인식 없음. 상태 초기화.")
            last_recognized_gesture = None # 다음 제스처를 받기 위해 상태 초기화
        # --- 변경된 핵심 로직 끝 ---

        cv2.imshow("Hand Gesture Camera (0)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

# --- 프로그램 시작점 ---
if __name__ == "__main__":
    hand_gesture_thread()
