from ultralytics import YOLO
import cv2

# YOLO 모델 로드
model = YOLO("yolov5s.pt")  # 자동 다운로드됨

# COCO dataset에서 person 클래스는 항상 index=0
PERSON_CLASS_ID = 0

# 웹캠 열기
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO 예측
    results = model.predict(frame, conf=0.5, verbose=False)

    # 탐지된 사람만 필터링
    for result in results:
        boxes = result.boxes
        for box, cls in zip(boxes.xyxy, boxes.cls):
            if int(cls) == PERSON_CLASS_ID:
                x1, y1, x2, y2 = map(int, box)
                # 사각형 + 라벨
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    "Person",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

    # 출력
    cv2.imshow("Person Detection", frame)

    # 'q' 키로 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
