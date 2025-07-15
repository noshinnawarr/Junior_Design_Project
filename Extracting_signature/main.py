import cv2

# Load the image
source_image = cv2.imread("test.jpg")
img = 0

# Step 1: Dewarp
try:
    img = dewarp_book(source_image)
    cv2.imwrite("step 1 - page_dewarped.jpg", img)
    print("- step1 (cropping with the margins + book dewarping): OK")
except Exception as e:
    print("Error in dewarping:", e)

# Step 2: Signature Extraction
try:
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = extract_signature(gray_img)
    cv2.imwrite("step 2 - signature_extracted.jpg", img)
    print("- step2 (signature extractor): OK")
except Exception as e:
    print("Error in signature extraction:", e)

# Step 3: Unsharpen
try:
    img = unsharpen_mask(img)
    cv2.imwrite("step 3 - unsharpen_mask.jpg", img)
    print("- step3 (unsharpening mask): OK")
except Exception as e:
    print("Error in unsharpening:", e)

# Step 4: Brightness/Contrast
try:
    img = funcBrightContrast(img)
    cv2.imwrite("step 4 - color_correlated.jpg", img)
    print("- step4 (color correlation): OK")
except Exception as e:
    print("Error in brightness/contrast:", e)

cv2.imwrite("output.jpg", img)
print("✅ Final output saved as output.jpg")
