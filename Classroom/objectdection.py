import cv2
from ultralytics import settings

# Disable data and crash report tracking
settings.update({"sync": False})
from ultralytics import YOLO

# 1. Load the lightweight YOLOv8 Nano model (~6MB, runs fast on CPU)
model = YOLO('yolov8n.pt')

# 2. Open the webcam (0 is default built-in camera)
cap = cv2.VideoCapture(0)

print("Starting Object Counter... Press 'q' on the video window to quit.")

while cap.isOpened():
    success, frame = cap.read()

    if not success:
        print("Failed to grab camera frame.")
        break

    # 3. Perform object detection on the current frame
    results = model(frame, imgsz=320, verbose=False)

    # Get total number of objects detected in this frame
    detected_count = len(results[0].boxes)

    # 4. Render detection boxes directly on the frame
    annotated_frame = results[0].plot()

    # 5. Display the object count overlay on screen
    cv2.putText(
        annotated_frame,
        f'Total Objects Counted: {detected_count}',
        (20, 50),  # Position (x, y)
        cv2.FONT_HERSHEY_SIMPLEX,
        1,  # Font scale
        (0, 255, 0),  # Text color (Green)
        2,  # Thickness
        cv2.LINE_AA,
    )

    # 6. Show the frame in a window
    cv2.imshow('BCA AI Demo - Live Object Counter', annotated_frame)

    # 7. Terminate program cleanly when pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Terminating program...")
        break

# Release camera and close all windows cleanly
cap.release()

# 2. Add short wait key loops to let Windows pump destroy events cleanly
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)

print("Camera feed closed successfully.")