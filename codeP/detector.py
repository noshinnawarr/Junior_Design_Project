import os
import numpy as np
import cv2
import sys

# Hardcoded prototxt and model paths
prototxt_path = r"C:\Users\HP\Documents\Junior_Design_Project\codeP\model\deploy.prototxt.txt"
model_path = r"C:\Users\HP\Documents\Junior_Design_Project\codeP\model\res10_300x300_ssd_iter_140000.caffemodel"
confidence_threshold = 0.5

def run(dataset_path, otherOptions):
    """
    dataset_path: str, not used here but passed by main.py
    otherOptions: str, path to input image
    """
    image_path = otherOptions

    # Check if files exist
    for path, name in [(image_path, "Image"), (prototxt_path, "Prototxt"), (model_path, "Model")]:
        if not os.path.exists(path):
            print(f"❌ Error: The specified {name} file does not exist: {path}")
            sys.exit(1)

    # Load the model
    print("[INFO] Loading model...")
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Unable to read the image file: {image_path}")
        sys.exit(1)

    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # Perform face detection
    print("[INFO] Computing object detections...")
    net.setInput(blob)
    detections = net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence >= confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            text = f"{(confidence * 100):.2f}%"
            y = startY - 10 if startY - 10 > 10 else startY + 10

            cv2.rectangle(image, (startX, startY), (endX, endY),
                          (0, 255, 0), 2)
            cv2.putText(image, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

    # Show the output image
    cv2.imshow("Output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
