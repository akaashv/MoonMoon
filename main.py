from ultralytics import YOLO
import cv2
import os

# Load YOLO model (downloads automatically the first time)
model = YOLO("yolov8l.pt")

# Read the parking lot image
image_folder = "images"

for filename in os.listdir(image_folder):

    # Skip non-image files
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(image_folder, filename)

    print(f"\nProcessing: {filename}")

    image = cv2.imread(image_path)

    results = model(image)
car_count = 0

# COCO class IDs
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

for result in results:
    for box in result.boxes:
        class_id = int(box.cls)

        if class_id in VEHICLE_CLASSES:
            car_count += 1

print(f"Vehicles detected: {car_count}")

# Decide if there is a gathering
THRESHOLD = 10

if car_count > THRESHOLD:
    print("✅ Car gathering detected!")
    output_signal = 1
else:
    print("❌ No gathering detected.")
    output_signal = 0

print(f"Output signal: {output_signal}")

# Draw detections
annotated = results[0].plot()

# Save the annotated image
output_path = os.path.join("output", f"result_{filename}")
cv2.imwrite(output_path, annotated)

# Show the result
cv2.imshow("Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()