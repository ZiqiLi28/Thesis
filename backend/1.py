11111111111

import re
import cv2
import util
import easyocr
import numpy as np
import matplotlib.pyplot as plt
from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures

# # Define constants for Azure
# endpoint = "https://thsis.cognitiveservices.azure.com/"
# key = "4U0j6yDtILKRXL86eYiSCKtXgw97XvwAq6TJW0KEB5t4DR8FFRsdJQQJ99ALACi5YpzXJ3w3AAAFACOGKe6y"
# client = ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

# Define constants
MODEL_CFG_PATH = './model/cfg/yolov3.cfg'
MODEL_WEIGHTS_PATH = './model/weights/model.weights'
IMG_PATH = './data/car2.jpg'

# Load YOLO model
net = cv2.dnn.readNetFromDarknet(MODEL_CFG_PATH, MODEL_WEIGHTS_PATH)

# Load image
img = cv2.imread(IMG_PATH)
H, W, _ = img.shape

# Convert image to blob format for YOLO
blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False)
net.setInput(blob)

# Get detections
detections = util.get_outputs(net)

# Process detections
bboxes, class_ids, scores = [], [], []
for detection in detections:
    xc, yc, w, h = detection[:4]
    bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]

    score = np.max(detection[5:])
    class_id = np.argmax(detection[5:])

    bboxes.append(bbox)
    class_ids.append(class_id)
    scores.append(score)

# Apply Non-Maximum Suppression (NMS)
bboxes, class_ids, scores = util.NMS(bboxes, class_ids, scores)

print("\n--- YOLO + OCR Results ---")

# Check if license plates are detected
if bboxes is None or len(bboxes) == 0:
    print("No license plates detected.")
else:
    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])

    # List to store OCR results
    ocr_results = []

    # Process detected bounding boxes
    for bbox in bboxes:
        xc, yc, w, h = bbox
        x1, y1 = int(xc - w / 2), int(yc - h / 2)
        x2, y2 = int(xc + w / 2), int(yc + h / 2)

        # Extract and preprocess license plate region
        license_plate = img[y1:y2, x1:x2].copy()
        license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)

        # Generate augmented images for the license plate
        augmentations = [license_plate_gray]
        for angle in [-5, 5]:  # Slight rotation
            M = cv2.getRotationMatrix2D((license_plate.shape[1] // 2, license_plate.shape[0] // 2), angle, 1)
            rotated = cv2.warpAffine(license_plate_gray, M, (license_plate.shape[1], license_plate.shape[0]))
            augmentations.append(rotated)

        for gamma in [0.5, 1.5]:  # Brightness adjustments
            bright = np.clip(((license_plate_gray / 255.0) ** gamma) * 255, 0, 255).astype(np.uint8)
            augmentations.append(bright)

        for aug_img in augmentations:
            # Adaptive thresholding for better OCR performance
            license_plate_thresh = cv2.adaptiveThreshold(
                aug_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=11, C=2
            )

            # OCR recognition
            output = reader.readtext(license_plate_thresh, detail=1)

            for _, text, text_score in output:
                ocr_results.append((text, text_score))

    # Sort OCR results by EasyOCR confidence
    ocr_results = sorted(ocr_results, key=lambda x: x[1], reverse=True)

    # Filter results to keep only alphanumeric (letters and numbers)
    filtered_results = []
    for text, score in ocr_results:
        clean_text = re.sub(r'[^A-Za-z0-9]', '', text)  # Remove non-alphanumeric characters
        if clean_text:
            filtered_results.append((clean_text, score))

    # Display the results
    if filtered_results:
        for text, score in filtered_results[:10]:  # Limit to 10
            print(f"- {text} (Confidence: {score:.2f})")
    else:
        print("License plates detected, but no text extracted.")

# # --- Azure Text Recognition ---
# def azure_text_recognition(img_path):
#     # Submit the image for analysis
#     with open(img_path, "rb") as f:
#         image_data = f.read()
#     # Analyze images to extract text
#     result = client.analyze(
#         image_data=image_data,
#         visual_features=[VisualFeatures.READ]
#      )

#     print("\n--- Azure Computer Vision Recognition Results ---")

#     # Output
#     if result.read is not None:
#        # ---------if any(char.isdigit() for char in line.text) and any(char.isalpha() for char in line.text):        -------------------
#         for line in result.read.blocks[0].lines:
#             print(f"{line.text}")
#     else:
#             print("No text detected.")

# Call Azure API to recognize text from the image
# azure_text_recognition(IMG_PATH)

# Visualization
plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
plt.imshow(license_plate_gray, cmap='gray')

plt.subplot(1, 2, 2)
plt.imshow(license_plate_thresh, cmap='gray')

plt.tight_layout()
plt.show()
