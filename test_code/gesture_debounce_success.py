import cv2
import numpy as np
import threading
import time
import requests
from tensorflow.keras.models import load_model
# mediapipe는 스레드 함수 내부에서 import 합니다.

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
    # --- ✨ 1. 디버깅 프린트 추가 ---
    print("✅ [1/5] 스레드 시작")
    model = load_model("hand_model.h5")
    # --- ✨ 2. 디버깅 프린트 추가 ---
    print("✅ [2/5] Keras 모델 로딩 성공")
    
    class_names = ["fan_off", "fan_on", "light_off", "light_on"]

    import mediapipe as mp
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, model_complexity=0)
    mp_draw = mp.solutions.drawing_utils

    # 카메라 자동 감지
    cap = None
    # --- ✨ 3. 디버깅 프린트 추가 ---
    print("🟡 [3/5] 카메라 탐색 시작...")
    for i in range(3):
        test_cap = cv2.VideoCapture(i)
        if test_cap.isOpened():
            print(f"✅ 카메라 {i} 사용 가능")
            cap = test_cap
            break
            
    if cap is None:
        print("❌ 카메라 열기 실패! 연결 상태를 확인하고 다른 프로그램이 카메라를 사용하고 있지 않은지 확인하세요.")
        return

    # --- ✨ 4. 디버깅 프린트 추가 ---
    print("✅ [4/5] 카메라 설정 완료")

    # 제스처 인식 시간 설정 (0.8초)
    CONFIRMATION_TIME = 0.8

    last_confirmed_gesture = None
    candidate_gesture = None
    candidate_timestamp = 0
    
    frame_counter = 0

    # --- ✨ 5. 디버깅 프린트 추가 ---
    print("✅ [5/5] 메인 루프 시작")
    while True:
        # print("메인 루프 실행 중...") # 루프가 도는지 확인하고 싶을 때 이 줄의 주석(#)을 제거하세요.
        ret, frame = cap.read()
        if not ret:
            print("❌ 프레임 읽기 실패. 카메라 연결이 불안정할 수 있습니다.")
            continue

        frame_counter += 1
        # 2프레임당 1번 분석
        if frame_counter % 2 != 0:
            cv2.imshow("Hand Gesture Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        hand_detected = False

        if results.multi_hand_landmarks:
            hand_detected = True
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min, y_min = min(x_min, x), min(y_min, y)
                    x_max, y_max = max(x_max, x), max(y_max, y)

                pad = 20
                x_min, y_min = max(x_min - pad, 0), max(y_min - pad, 0)
                x_max, y_max = min(x_max + pad, w), min(y_max + pad, h)
                
                hand_img = frame[y_min:y_max, x_min:x_max]
                if hand_img.size == 0:
                    continue

                hand_input = cv2.resize(hand_img, (128, 128)) / 255.0
                hand_input = np.expand_dims(hand_input, axis=0)

                preds = model.predict(hand_input, verbose=0)
                class_idx = np.argmax(preds[0])
                confidence = preds[0][class_idx]
                
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                if confidence > 0.8:
                    current_gesture = class_names[class_idx]
                    
                    if current_gesture != candidate_gesture:
                        candidate_gesture = current_gesture
                        candidate_timestamp = time.time()
                    else:
                        elapsed_time = time.time() - candidate_timestamp
                        label = f"{current_gesture} ({elapsed_time:.1f}s)"
                        cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                        if elapsed_time > CONFIRMATION_TIME and candidate_gesture != last_confirmed_gesture:
                            print(f"▶️ 제스처 확정: {candidate_gesture}")
                            cmd = ""
                            if candidate_gesture == "fan_on": cmd = "motor_on"
                            elif candidate_gesture == "fan_off": cmd = "motor_off"
                            elif candidate_gesture == "light_on": cmd = "light_on"
                            elif candidate_gesture == "light_off": cmd = "light_off"

                            if cmd:
                                send_to_gpio(cmd)
                                threading.Thread(target=send_to_speaker, args=(cmd,), daemon=True).start()
                                last_confirmed_gesture = candidate_gesture
                
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        if not hand_detected:
            candidate_gesture = None
            if last_confirmed_gesture and 'on' in last_confirmed_gesture:
                last_confirmed_gesture = None

        cv2.imshow("Hand Gesture Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print("메인 루프 종료. 프로그램 정리 중...")
    cap.release()
    cv2.destroyAllWindows()

# --- 프로그램 시작점 ---
if __name__ == "__main__":
    hand_gesture_thread()
