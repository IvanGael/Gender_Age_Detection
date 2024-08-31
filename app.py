import cv2
import torch
import numpy as np
from deepface import DeepFace
from cvzone import overlayPNG

# Load YOLOv5 model using torch.hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Initialize video capture
cap = cv2.VideoCapture("video6.mp4")  # Use 0 for webcam or provide video file path

fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output3.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

def create_box(img, x, y, w, h, text, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.rectangle(img, (x, y - 20), (x + w, y), color, -1)
    cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

while True:
    success, img = cap.read()
    if not success:
        break

    # Detect persons using YOLOv5
    results = model(img)

    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        # Only process if the detected class is a person (class 0 in COCO dataset)
        if int(cls) == 0:
            # Draw person bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Crop the face for DeepFace analysis
            face = img[y1:y2, x1:x2]

            try:
                # Analyze age and gender using DeepFace
                result = DeepFace.analyze(face, actions=['age', 'gender'], enforce_detection=False)

                print(result)

                # Check if result is not empty
                if result:
                    age = result[0]['age']
                    gender = result[0]['dominant_gender']

                    # Create cvzone boxes for age category and gender
                    age_category = "Adult" if age > 15 else "Young"
                    create_box(img, x1, y1 - 50, 100, 25, age_category, (0, 255, 0))
                    create_box(img, x1 + 110, y1 - 50, 100, 25, gender, (255, 0, 0))
                else:
                    print("No face detected in the cropped image")

            except Exception as e:
                print(f"Error in face analysis: {e}")

    # Write the frame to the output video file
    out.write(img)

    # Display the result
    cv2.namedWindow("Person Detection", cv2.WINDOW_NORMAL)
    cv2.imshow("Person Detection", img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()