import cv2
import numpy as np
import mediapipe as mp
import tflite_runtime.interpreter as tflite # 라즈베리파이에서는 이걸 사용!
import requests # HTTP 요청을 보내기 위해 requests 모듈 추가
import time # 지연 시간 관리를 위해 time 모듈 추가

# -----------------------------
# TFLite 모델 로드
# -----------------------------
try:
    interpreter = tflite.Interpreter(model_path="hand_model_last_compatible.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ TFLite 모델 로드 성공.")
except Exception as e:
    print(f"❌ TFLite 모델 로드 실패: {e}")
    exit()

# -----------------------------
# 클래스 이름 (학습된 모델의 class_names와 정확히 일치해야 함)
# -----------------------------
# 이 부분은 hand_model_last_compatible.tflite 모델이 학습된 클래스 순서와 일치해야 합니다.
# 만약 모델이 학습된 클래스 순서가 다르다면 이 리스트를 수정해야 합니다.
class_names = ["fan_off", "fan_on", "light_off", "light_on"] 
print(f"🌟 인식할 제스처 클래스: {class_names}")

# -----------------------------
# Mediapipe hands 초기화
# -----------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=1, 
    min_detection_confidence=0.7 # 이 신뢰도보다 낮으면 제스처로 인식 안함
)
mp_draw = mp.solutions.drawing_utils
print("✅ MediaPipe Hands 초기화 완료.")

# -----------------------------
# 웹캠 열기
# -----------------------------
cap = cv2.VideoCapture(0) # 0은 기본 웹캠
if not cap.isOpened():
    print("❌ 웹캠을 열 수 없습니다. 카메라가 연결되어 있고 다른 프로그램에서 사용 중이 아닌지 확인하세요.")
    exit()
print("✅ 웹캠 열기 성공.")

# -----------------------------
# GPIO 서버로 명령 전송 함수
# -----------------------------
def send_to_gpio_server(cmd):
    try:
        url = "http://localhost:5000/control" # gpio_server.py의 /control 엔드포인트
        res = requests.post(url, json={"cmd": cmd})
        print(f"📡 명령 전송됨: {cmd}, 응답: {res.text}")
    except requests.exceptions.ConnectionError:
        print(f"❌ GPIO 서버 연결 실패. 'gpio_server.py'가 실행 중인지 확인하세요.")
    except Exception as e:
        print(f"❌ GPIO 서버 요청 실패 ({cmd}):", e)

# -----------------------------
# 제스처 상태 추적 변수 (불필요한 반복 전송 방지)
# -----------------------------
last_command_sent = None # 마지막으로 전송한 명령
command_debounce_time = 2 # 명령 재전송까지의 최소 시간 (초)
last_command_timestamp = time.time() # 마지막 명령 전송 시간

# -----------------------------
# 메인 루프
# -----------------------------
frame_count = 0
current_label = "No hand detected" # 현재 화면에 표시될 라벨
last_detected_gesture = None # 마지막으로 감지된 제스처 클래스 이름

print("🚀 제스처 인식을 시작합니다. 'q'를 눌러 종료하세요.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ 프레임을 읽을 수 없습니다.")
        break

    frame_count += 1
    
    # 이미지 반전 (필요한 경우, 셀카 모드처럼 보이게 함)
    # frame = cv2.flip(frame, 1) 

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    detected_this_frame = False

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

            # 패딩 추가
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
            interpreter.set_tensor(input_details[0]['index'], hand_input)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]['index'])[0]

            class_idx = np.argmax(preds)
            confidence = preds[class_idx]

            # 일정 신뢰도 이상일 때만 제스처 인식 및 명령 전송
            if confidence > 0.7: 
                current_label = f"{class_names[class_idx]} ({confidence:.2f})"
                detected_this_frame = True
                last_detected_gesture = class_names[class_idx] # 마지막으로 감지된 제스처 업데이트

                # 명령 전송 로직
                current_time = time.time()
                # 마지막으로 보낸 명령과 다른 명령이 감지되었거나,
                # 같은 명령이라도 일정 시간(debounce_time)이 지났을 경우에만 전송
                if (last_command_sent != last_detected_gesture) or \
                   (current_time - last_command_timestamp > command_debounce_time):
                    
                    if last_detected_gesture == "fan_on":
                        send_to_gpio_server("motor_on")
                        last_command_sent = "fan_on"
                        last_command_timestamp = current_time
                    elif last_detected_gesture == "fan_off":
                        send_to_gpio_server("motor_off")
                        last_command_sent = "fan_off"
                        last_command_timestamp = current_time
                    elif last_detected_gesture == "light_on":
                        send_to_gpio_server("light_on")
                        last_command_sent = "light_on"
                        last_command_timestamp = current_time
                    elif last_detected_gesture == "light_off":
                        send_to_gpio_server("light_off")
                        last_command_sent = "light_off"
                        last_command_timestamp = current_time
            else:
                # 신뢰도가 낮으면 "No hand detected"로 표시하지 않고, 이전 라벨 유지 (시각적 안정성)
                pass # 이 경우 current_label은 이전 값 유지

            # 감지된 손 바운딩 박스 및 랜드마크 그리기
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # 손이 감지되지 않았을 때 라벨 업데이트
    if not detected_this_frame:
        current_label = "No hand detected"
        # 손이 없어도 last_command_sent를 초기화하지 않음 (이전 명령 유지)

    # 화면에 현재 라벨 표시
    cv2.putText(
        frame,
        current_label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),  # 노란색
        2,
    )

    cv2.imshow("Smart Hand Control (TFLite)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("👋 프로그램 종료 요청.")
        break

# 종료 시 자원 해제
cap.release()
cv2.destroyAllWindows()
print("Clean up: 웹캠 및 창 해제.")
