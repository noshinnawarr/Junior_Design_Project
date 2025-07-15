from ultralytics import YOLO
import cv2
from google.colab.patches import cv2_imshow
import csv
from google.colab import files

# Load YOLOv8 small model for better accuracy
model = YOLO("yolov8s.pt")

csv_filename = "bottle_counts.csv"

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["image_name", "bottle_count"])  # CSV header

    for image_file in image_paths:
        print(f"Processing {image_file} ...")

        results = model.predict(source=image_file, conf=0.15, augment=True)
        
        total_bottles = 0
        for result in results:
            annotated_img = result.plot()
            cv2_imshow(annotated_img)
            output_filename = "detected_" + image_file
            cv2.imwrite(output_filename, annotated_img)
            print(f"Saved annotated image as {output_filename}")
            
            for box in result.boxes:
                cls_id = int(box.cls)
                cls_name = result.names[cls_id]
                if cls_name.lower() == "bottle":
                    total_bottles += 1
                conf = float(box.conf)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                print(f"Class: {cls_name}, Confidence: {conf:.2f}, Box: [{x1}, {y1}, {x2}, {y2}]")
        
        print(f"Total bottles detected in {image_file}: {total_bottles}")
        writer.writerow([image_file, total_bottles])

print(f"\nBottle counts saved in {csv_filename}")

# Automatically download the CSV file to your local machine
files.download(csv_filename)
