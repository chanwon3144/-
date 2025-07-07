import cv2
from time import sleep, time
from ultralytics import YOLO

# --- 라즈베리 파이 환경 자동 감지 및 부저 설정 ---
try:
    from gpiozero import TonalBuzzer
    buzzer = TonalBuzzer(13)
    IS_RASPBERRY_PI = True
    print("라즈베리 파이 환경으로 설정되었습니다. 부저를 사용합니다.")
except (ImportError, Exception):
    print("알림: 라즈베리 파이 환경이 아닙니다. 부저 대신 콘솔에 메시지를 출력합니다.")
    IS_RASPBERRY_PI = False

# --- YOLOv5 모델 로드 ---
print("YOLOv5 모델을 로드하는 중입니다...")
model = YOLO("yolov5n.pt")  # 'n' 모델은 가볍고 빠릅니다.
PERSON_CLASS_ID = 0  # YOLO 모델에서 'person' 클래스는 0번입니다.

# --- 웹캠 설정 ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("오류: 웹캠을 열 수 없습니다.")
    exit()

# --- 상태 변수 및 상수 초기화 ---
detection_state = "NO_PERSON"
last_person_seen_time = 0
PERSON_SEEN_TIMEOUT = 2.0  # 초
last_known_box = None
frame_count = 0
PROCESS_FRAME_INTERVAL = 3  # 3프레임마다 한 번씩만 추론하여 성능 확보

# --- 함수 정의: 부저 제어 ---
def buzzer_on():
    """사람이 나타났을 때 호출되는 함수"""
    print("🟢 사람 발견! 부저를 켭니다.")
    if IS_RASPBERRY_PI:
        try:
            buzzer.play('G5')
            sleep(0.3)
            buzzer.play('C5')
            sleep(0.3)
            buzzer.stop()
        except Exception as e:
            print(f"부저 재생 중 오류 발생: {e}")
    else:
        print("🎶 딩동! (사람 감지 알림)")

def buzzer_off():
    """사람이 사라졌을 때 호출되는 함수"""
    # 사용자가 요청한 대로 문구 수정
    print("🔴 사람 사라짐. 사람 다시 나타나면 부저를 킴!")
    if IS_RASPBERRY_PI:
        buzzer.stop()

# --- 메인 프로그램 루프 ---
print("출입 감지 시스템을 시작합니다. (YOLOv5, 종료: 'q' 키)")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("오류: 웹캠에서 프레임을 읽을 수 없습니다.")
            break

        frame_count += 1
        person_found_this_frame = False

        # 지정된 간격마다 YOLO 추론 수행
        if frame_count % PROCESS_FRAME_INTERVAL == 0:
            # YOLO 모델로 추론 수행
            results = model.predict(frame, conf=0.5, verbose=False, classes=[PERSON_CLASS_ID])

            # 결과에서 사람 찾기
            for result in results:
                if len(result.boxes) > 0:
                    person_found_this_frame = True
                    # 가장 신뢰도 높은 사람의 박스 정보 저장
                    best_box = result.boxes[0].xyxy[0]
                    last_known_box = tuple(map(int, best_box))
                    last_person_seen_time = time()
                    break # 한 프레임에 한 명만 추적

        # --- 상태에 따른 부저 제어 ---
        is_person_currently_visible = time() - last_person_seen_time < PERSON_SEEN_TIMEOUT

        if is_person_currently_visible and detection_state == "NO_PERSON":
            detection_state = "PERSON_SEEN"
            buzzer_on()
        elif not is_person_currently_visible and detection_state == "PERSON_SEEN":
            detection_state = "NO_PERSON"
            buzzer_off()
            last_known_box = None # 사람이 사라지면 박스 정보도 삭제

        # --- 화면 출력 ---
        if last_known_box:
            x1, y1, x2, y2 = last_known_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow("Person Detection System (YOLOv5)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n프로그램을 강제 종료합니다.")

finally:
    # 프로그램 종료 시 모든 리소스 정리
    print("시스템을 종료하고 리소스를 해제합니다.")
    cap.release()
    cv2.destroyAllWindows()
    if IS_RASPBERRY_PI:
        buzzer.off()
        buzzer.close()
