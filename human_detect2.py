from ultralytics import YOLO
import cv2
import requests
import time

# -----------------------------
# YOLOv5n 모델 로드
# -----------------------------
model = YOLO("yolov5n.pt")
PERSON_CLASS_ID = 0

# -----------------------------
# 카메라 설정
# -----------------------------
cap = cv2.VideoCapture(2)  # 카메라 인덱스는 상황에 따라 조정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# -----------------------------
# 상태 관리 변수
# -----------------------------
frame_count = 0
last_status = None
last_buzzer_time = time.time()
debounce_sec = 2

# "최근 본 사람" 상태 유지 시간
person_seen_timeout = 0.5  # 초
last_person_seen = 0

# 마지막으로 인식된 사람 위치 (UI 유지용)
last_box = None

# -----------------------------
# 부저 전송 함수
# -----------------------------
def send_buzzer(cmd):
    try:
        res = requests.post("http://localhost:5000/control", json={"cmd": cmd})
        print(f"📡 부저 명령: {cmd} → 응답: {res.text}")
    except:
        print("❌ 부저 명령 실패")

# -----------------------------
# 메인 루프
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame_count += 1

    # 매 프레임마다 추론하지 않고 간격 둠 (속도 향상)
    if frame_count % 3 != 0:
        # 최근에 본 사람 있을 경우, 박스 유지
        if time.time() - last_person_seen < person_seen_timeout and last_box:
            x1, y1, x2, y2 = last_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("Smooth Person Detection", frame)
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break
        continue

    # 해상도 줄여서 YOLO 추론 속도 향상
    small_frame = cv2.resize(frame, (320, 240))
    results = model.predict(small_frame, conf=0.5, verbose=False, imgsz=224)

    person_detected = False
    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            if int(cls) == PERSON_CLASS_ID:
                person_detected = True
                x1, y1, x2, y2 = map(int, box)

                # 원래 프레임에 맞춰 좌표 확대 (320 → 640)
                x1, y1, x2, y2 = x1*2, y1*2, x2*2, y2*2
                last_person_seen = time.time()
                last_box = (int(x1), int(y1), int(x2), int(y2))

    # 최근에 본 사람이 있다면 UI 표시 유지
    if time.time() - last_person_seen < person_seen_timeout and last_box:
        x1, y1, x2, y2 = last_box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # 부저 명령 (debounce)
    current_time = time.time()
    if person_detected and (last_status != "on" or current_time - last_buzzer_time > debounce_sec):
        send_buzzer("buzzer_on")
        last_status = "on"
        last_buzzer_time = current_time
    elif not person_detected and (last_status != "off" or current_time - last_buzzer_time > debounce_sec):
        send_buzzer("buzzer_off")
        last_status = "off"
        last_buzzer_time = current_time

    # 출력
    cv2.imshow("Smooth Person Detection", frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

