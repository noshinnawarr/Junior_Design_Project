from ultralytics import YOLO
import cv2

# Load trained YOLOv8 model (detection model for emotions)
model = YOLO("best.pt")  # Make sure this is a detection model, not a classification model

# Path to the image you want to test
image_path = "test.jpg"

# Run prediction
results = model(image_path)

# Show the result with bounding boxes in a window
results[0].show()  # This opens an OpenCV window

# Save the result to file
results[0].save(filename="output.jpg")

# Print detections
for box in results[0].boxes:
    cls_id = int(box.cls)
    conf = float(box.conf)
    emotion = results[0].names[cls_id]
    print(f"Detected {emotion} ({conf:.2f})")
