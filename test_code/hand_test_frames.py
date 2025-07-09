import cv2
import numpy as np
import threading
import time
import requests
from tensorflow.keras.models import load_model
from ultralytics import YOLO # 이 모듈은 hand_gesture_thread에서는 직접 사용되지 않지만, 전체 프로젝트 구성상 필요할 수 있으니 그대로 둡니다.

# --- 부저 설정 (gpiozero) ---
try:
    from gpiozero import TonalBuzzer
    # buzzer = TonalBuzzer(13) # 실제 사용 시 주석 해제
    IS_RASPBERRY_PI = True
    print("라즈베리 파이 환경으로 설정되었습니다. 부저 사용 가능.")
except (ImportError, Exception):
    print("알림: 라즈베리 파이 환경이 아닙니다. 부저 대신 콘솔에 메시지를 출력합니다.")
    IS_RASPBERRY_PI = False

# --- GPIO 및 스피커 전송 함수 ---
def send_to_gpio(cmd):
    try:
        # GPIO 서버 URL은 프로젝트 설정에 맞게 변경 (localhost:5000 또는 라즈베리파이 IP:5000)
        gpio_server_url = "http://localhost:5000/control"
        res = requests.post(gpio_server_url, json={"cmd": cmd})
        print(f"📡 GPIO 서버 응답({cmd}):", res.text)
    except requests.exceptions.ConnectionError:
        print(f"❌ GPIO 서버 연결 실패. '{cmd}' 명령 전송 불가. 서버가 실행 중인지 확인하세요.")
    except Exception as e:
        print(f"❌ GPIO 전송 실패 ({cmd}):", e)

def send_to_speaker(cmd):
    try:
        # 스피커 서버 URL은 실제 스피커 서버의 IP와 포트에 맞게 변경
        speaker_url = "http://10.10.15.167:8000/notify"
        res = requests.post(speaker_url, json={"cmd": cmd})
        print(f"🔊 스피커 응답({cmd}):", res.text)
    except Exception as e:
        print(f"❌ 스피커 전송 실패 ({cmd}):", e)

# --- 손 제스처 인식 스레드 ---
def hand_gesture_thread():
    # 모델 로드 및 클래스 이름 정의
    try:
        model = load_model("hand_model.h5")
        print("✅ hand_model.h5 모델 로드 성공.")
    except Exception as e:
        print(f"❌ hand_model.h5 모델 로드 실패: {e}. 경로 및 파일 존재 여부 확인.")
        return

    class_names = ["fan_off", "fan_on", "light_off", "light_on"]

    # MediaPipe Hands 초기화
    try:
        import mediapipe as mp
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, model_complexity=0)
        mp_draw = mp.solutions.drawing_utils
        print("✅ MediaPipe Hands 초기화 성공.")
    except ImportError:
        print("❌ MediaPipe가 설치되지 않았습니다. 'pip install mediapipe' 명령으로 설치해주세요.")
        return
    except Exception as e:
        print(f"❌ MediaPipe Hands 초기화 실패: {e}")
        return

    # 카메라 자동 감지 및 열기
    cap = None
    for i in range(3): # 0, 1, 2번 카메라 시도
        test_cap = cv2.VideoCapture(i)
        if test_cap.isOpened():
            print(f"✅ 카메라 {i} 사용 가능. 연결합니다.")
            cap = test_cap
            break
        test_cap.release() # 열리지 않으면 바로 닫음
    if cap is None:
        print("❌ 카메라 열기 실패! 카메라가 연결되어 있고 다른 프로그램에서 사용 중이 아닌지 확인하세요.")
        return

    # --- 제스처 인식 안정화를 위한 변수 ---
    last_recognized_command_sent = None # 마지막으로 성공적으로 전송된 명령어 (중복 방지용)
    current_frame_detected_gesture = None # 현재 프레임에서 모델이 예측한 제스처 (신뢰도 충족 시)

    last_confirmed_gesture = None # 연속적으로 인식되어 '확정'된 제스처
    consecutive_frames_count = 0 # 동일 제스처 연속 인식 프레임 카운트
    required_consecutive_frames = 5 # 명령 전송을 위해 필요한 연속 인식 프레임 수 (조절 가능)

    frame_counter = 0

    print("\n--- 제스처 인식 시작. 'q'를 눌러 종료 ---")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ 프레임 읽기 실패! 카메라 연결이 끊어졌을 수 있습니다.")
            break # 프레임 읽기 실패 시 루프 종료

        frame_counter += 1
        # 라즈베리파이 CPU 과부하 감소를 위해 3프레임 중 1번만 분석
        if frame_counter % 3 != 0:
            cv2.imshow("Hand Gesture Camera (0)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        current_frame_detected_gesture = None # 매 프레임 시작 시 초기화

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

                # 손 영역 추출 및 패딩 적용
                pad = 20
                x_min = max(x_min - pad, 0)
                y_min = max(y_min - pad, 0)
                x_max = min(x_max + pad, w)
                y_max = min(y_max + pad, h)
                hand_img = frame[y_min:y_max, x_min:x_max]

                if hand_img.size == 0 or hand_img.shape[0] < 10 or hand_img.shape[1] < 10:
                    continue # 유효하지 않은 손 영역은 건너뛰기

                # 모델 입력에 맞게 이미지 전처리
                hand_input = cv2.resize(hand_img, (128, 128)) / 255.0
                hand_input = np.expand_dims(hand_input, axis=0)

                # 모델 예측
                preds = model.predict(hand_input, verbose=0)
                class_idx = np.argmax(preds[0])
                confidence = preds[0][class_idx]

                # 신뢰도 0.8 이상일 때만 유효한 제스처로 간주
                if confidence > 0.8:
                    current_frame_detected_gesture = class_names[class_idx]
                    label = f"{current_frame_detected_gesture} ({confidence:.2f})"
                    cv2.putText(frame, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # --- 제스처 인식 후 처리 로직 (핵심 변경 부분) ---
        if current_frame_detected_gesture is not None: # 현재 프레임에서 유효한 제스처가 인식된 경우
            if current_frame_detected_gesture == last_confirmed_gesture:
                consecutive_frames_count += 1
            else:
                consecutive_frames_count = 1 # 새로운 제스처가 인식되면 카운트 1부터 시작
                last_confirmed_gesture = current_frame_detected_gesture # 확정 대기 제스처 업데이트

            # 충분한 프레임 동안 동일 제스처가 연속적으로 인식되었고,
            # 이전에 보낸 명령과 다른 제스처인 경우에만 명령 전송
            if consecutive_frames_count >= required_consecutive_frames:
                if last_confirmed_gesture != last_recognized_command_sent:
                    cmd_to_send = ""
                    if last_confirmed_gesture == "fan_on":
                        cmd_to_send = "motor_on"
                    elif last_confirmed_gesture == "fan_off":
                        cmd_to_send = "motor_off"
                    elif last_confirmed_gesture == "light_on":
                        cmd_to_send = "light_on"
                    elif last_confirmed_gesture == "light_off":
                        cmd_to_send = "light_off"

                    if cmd_to_send:
                        send_to_gpio(cmd_to_send)
                        threading.Thread(target=send_to_speaker, args=(cmd_to_send,), daemon=True).start()
                        last_recognized_command_sent = last_confirmed_gesture # 명령 전송된 제스처로 업데이트
                        print(f"✨ 명령 전송! 제스처: '{last_confirmed_gesture}', 명령: '{cmd_to_send}'")
                        # 명령 전송 후, 다음 새로운 제스처를 받기 위해 확정 상태 초기화 (선택적)
                        # 여기서는 동일 제스처는 한 번만 보내고 다음 다른 제스처까지 기다리는 로직이므로,
                        # last_recognized_command_sent만 업데이트하고 consecutive_frames_count는 유지하여
                        # 같은 제스처를 계속 해도 다시 보내지 않도록 함.
                # else:
                #     print(f"DEBUG: 동일 제스처 '{last_confirmed_gesture}' 연속 인식 중, 이미 명령 전송됨.")

        else: # 현재 프레임에서 유효한 제스처가 인식되지 않은 경우 (손을 치웠거나, 신뢰도 미달)
            # 이전에 명령을 보낸 상태를 초기화하여 다음 새로운 제스처를 받을 준비
            if last_recognized_command_sent is not None:
                print("--- 제스처 인식 없음. 다음 명령 대기 상태로 초기화 ---")
            last_recognized_command_sent = None
            last_confirmed_gesture = None
            consecutive_frames_count = 0
        # --- 제스처 인식 후 처리 로직 끝 ---

        cv2.imshow("Hand Gesture Camera (0)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()
    print("👋 제스처 인식 스레드 종료.")


# --- 프로그램 시작점 ---
if __name__ == "__main__":
    hand_gesture_thread()
