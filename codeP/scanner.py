import cv2
from skimage.filters import threshold_local
import numpy as np
import imutils
import os
from pyimagesearch.transform import four_point_transform

def run(dataset_path, image_relative_path):
    image_path = os.path.join(dataset_path, image_relative_path)

    # Check if the file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Load the image and compute the ratio of the old height to the new height
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image file: {image_path}")

    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = imutils.resize(image, height=500)

    # Convert to grayscale, blur, and detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)

    print("STEP 1: Edge Detection")
    cv2.imshow("Output", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow("Output", edged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Find contours and approximate
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    print("STEP 2: Find contours of paper")
    cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
    cv2.imshow("Outline", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Perspective transform
    warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)

    # Thresholding
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    T = threshold_local(warped, 11, offset=10, method="gaussian")
    warped = (warped > T).astype("uint8") * 255

    print("STEP 3: Apply perspective transform")
    cv2.imshow("Original", imutils.resize(orig, height=650))
    cv2.imshow("Scanned", imutils.resize(warped, height=650))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
