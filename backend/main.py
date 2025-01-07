import os
import io
import re
import sys
import cv2
import util
import easyocr
import requests
import numpy as np
from flask_cors import CORS
from flask import Flask, request, jsonify
from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures

app = Flask(__name__)
CORS(app)

# Custom Vision Configuration
custom_vision_key = "5GyTCCKZr35nIbaaaaaaaaaaaaaaQJ99BAACi5YpzXJ3w3AAAIACOGxTEm"
custom_vision_endpoint = "https://plaaaaaaaaaaaaaaation1/image"

# Azure Configuration
key = "DLWYNyNyFvcmsehgizBk64wb4aaaaaaaaaaaaaa9BAACi5YpzXJ3w3AAAFACOGny2q"
endpoint = "https://taaaaaaaaaaaaaaure.com/"
client = ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

@app.route('/recognize', methods=['POST'])
def recognize_plate():
    # Prepare to capture all outputs
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    img_path = './uploaded_image.jpg'
    logs = ""
    try:
        # Check if image file is provided
        if 'image' not in request.files:
            print("No image file provided")
            logs = buffer.getvalue()
            return jsonify({'error': 'No image file provided', 'logs': logs}), 400

        file = request.files['image']
        if file.filename == '':
            print("No selected file")
            logs = buffer.getvalue()
            return jsonify({'error': 'No selected file', 'logs': logs}), 400

        # Save uploaded file
        file.save(img_path)

        # --- YOLO + EasyOCR Processing ---
        MODEL_CFG_PATH = './model/cfg/yolov3.cfg'
        MODEL_WEIGHTS_PATH = './model/weights/model.weights'

        # Load YOLO model
        net = cv2.dnn.readNetFromDarknet(MODEL_CFG_PATH, MODEL_WEIGHTS_PATH)

        # Load image
        img = cv2.imread(img_path)
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

                # Slight rotation
                for angle in [-5, 5]:
                    M = cv2.getRotationMatrix2D((license_plate.shape[1] // 2, license_plate.shape[0] // 2), angle, 1)
                    rotated = cv2.warpAffine(license_plate_gray, M, (license_plate.shape[1], license_plate.shape[0]))
                    augmentations.append(rotated)

                # Brightness adjustments
                for gamma in [0.5, 1.5]:
                    bright = np.clip(((license_plate_gray / 255.0) ** gamma) * 255, 0, 255).astype(np.uint8)
                    augmentations.append(bright)

                # Adaptive thresholding
                for aug_img in augmentations:
                    license_plate_thresh = cv2.adaptiveThreshold(aug_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=11, C=2)

                    # OCR recognition
                    output = reader.readtext(license_plate_thresh, detail=1)
                    for _, text, text_score in output:
                        clean_text = re.sub(r'[^A-Za-z0-9]', '', text)
                        if clean_text:
                            ocr_results.append((clean_text, text_score))
        
            # Sort OCR results by EasyOCR confidence
            ocr_results = sorted(ocr_results, key=lambda x: x[1], reverse=True)

            if ocr_results:
                for text, score in ocr_results[:10]:
                    print(f"- {text} (Confidence: {score:.2f})")
            else:
                print("License plates detected, but no text extracted.")

        # --- Azure Text Recognition ---
        with open(img_path, 'rb') as img_data:
            headers = {
                'Content-Type': 'application/octet-stream',
                'Prediction-Key': custom_vision_key
            }
            response = requests.post(custom_vision_endpoint, headers=headers, data=img_data)
        
        if response.status_code != 200:
            raise Exception(f"Custom Vision API error: {response.text}")

        detection_results = response.json()

        # Load the image for cropping
        image = cv2.imread(img_path)
        H, W, _ = image.shape

        cropped_plates = []
        for prediction in detection_results.get('predictions', []):
            if prediction['probability'] > 0.1:
                bbox = prediction['boundingBox']
                x1 = int(bbox['left'] * W)
                y1 = int(bbox['top'] * H)
                x2 = x1 + int(bbox['width'] * W)
                y2 = y1 + int(bbox['height'] * H)
                cropped_plates.append(image[y1:y2, x1:x2])

        if not cropped_plates:
            print("No license plates detected.")
            logs = buffer.getvalue()
            return jsonify({'logs': logs}), 200

        # Use Azure Computer Vision to read text
        print("\n--- Azure Computer Vision Results ---")

        # Result
        for i, plate in enumerate(cropped_plates):
            plate_path = f'./cropped_plate_{i}.jpg'
            plate_height, plate_width = plate.shape[:2]

            if plate_height < 50 or plate_width < 50:
                # Calculate scaling factor to make the smallest side at least 60
                scale_factor = 60 / min(plate_height, plate_width)

                # Resize the image
                new_width = int(plate_width * scale_factor)
                new_height = int(plate_height * scale_factor)

                plate = cv2.resize(plate, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            cv2.imwrite(plate_path, plate)

            # Read text from cropped plate
            with open(plate_path, "rb") as plate_img:
                plate_data = plate_img.read()
            result = client.analyze(image_data=plate_data, visual_features=[VisualFeatures.READ])

            if result.read is not None:
                for line in result.read.blocks[0].lines:
                    print(line.text)
            else:
                print("No text detected on cropped plate.")

            # Clean up temporary files
            os.remove(plate_path)

    except Exception as e:
        print(f"Error during processing: {str(e)}")

    finally:
        # Reset standard output
        logs = buffer.getvalue()
        sys.stdout = old_stdout

    # Delete the uploaded image after processing
    try:
        if os.path.exists(img_path):
            os.remove(img_path)
            print(f"Temporary image {img_path} deleted.")
    except Exception as cleanup_error:
        print(f"Error cleaning up file: {cleanup_error}")

    return jsonify({'logs': logs})

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
