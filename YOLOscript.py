from ultralytics import YOLO
import cv2

# Load trained model
# make sure path is correct, put in same project folder
model = YOLO("insert_model_here.pt")

# Open webcam
webcam = cv2.VideoCapture(0)

while webcam.isOpened():
    ret, image = webcam.read()

    if not ret:
        break

    # run YOLO
    results = model(image)

    # Draw boxes and get dice classes
    total = 0
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])  # Class index
            confidence = float(box.conf[0])  # Confidence score
            label = model.names[class_id]  # Class label

            # Add to dice total
            try:
                total += int(label)
            except ValueError:
                pass

            # Draw green box and label all detected dice
            bounding_box = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = bounding_box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    # Display total on screen
    cv2.putText(image, f"Total: {total}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

    # show image, press q to exit
    cv2.imshow("Dice Scanner", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close window
webcam.release()
cv2.destroyAllWindows()
