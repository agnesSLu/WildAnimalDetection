import cv2
from ultralytics import YOLO

model = YOLO('../runs/yolo-train/detect/train4/weights/best.pt')


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Optional: resize for faster processing
    frame_resized = cv2.resize(frame, (640, 640))

    # Inference
    results = model.predict(source=frame_resized, conf=0.8)

    # Plot and show
    annotated = results[0].plot()
    cv2.imshow("Real-Time Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
