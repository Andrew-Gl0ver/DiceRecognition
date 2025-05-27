from ultralytics import YOLO
import cv2
import time

# Load trained model
# make sure path is correct, put in same project folder
model = YOLO("insert_model_here.pt")

# Open webcam
webcam = cv2.VideoCapture(0)


def compare_images(image1, image2):
    diff = cv2.absdiff(image1, image2)
    _, thresh = cv2.threshold(diff, 85, 255, cv2.THRESH_BINARY)
    return cv2.countNonZero(thresh)


has_scanned = False
prev_gray = None
dice_settled = False
STILLNESS_LEVEL = 1000
STILLNESS_DURATION = 2
stopped_moving_time = None

while webcam.isOpened():
    ret, image = webcam.read()

    if not ret:
        break

    # make image gray and compare to previous frame
    # make sure the dice have stopped
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if prev_gray is not None:
        movement = compare_images(prev_gray, gray_image)

        if movement < STILLNESS_LEVEL:
            if stopped_moving_time is None:
                stopped_moving_time = time.time()

            time_elapsed = time.time() - stopped_moving_time
            if time_elapsed >= STILLNESS_DURATION:
                dice_settled = True
            else:
                dice_settled = False
        else:
            # if dice start moving again, reset
            stopped_moving_time = None
            dice_settled = False
            has_scanned = False

    prev_gray = gray_image.copy()

    if dice_settled and not has_scanned:
        has_scanned = True

        # run YOLO and draw image boxes
        results = model(image)
        total = 0
        for result in results:
            for box in result.boxes:
                # get class label and confidence level for dice
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                label = model.names[class_id]

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
