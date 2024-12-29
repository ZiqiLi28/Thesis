import io
import re
import sys
import cv2
import util
import easyocr
import numpy as np
from flask import Flask, request, jsonify
from azure.core.credentials import AzureKeyCredential
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures

app = Flask(__name__)

# Azure Configuration
# endpoint = "https://thsis.cognitiveservices.azure.com/"
# key = "4U0j6yDtILKRXL86eYiSCKtXgw97XvwAq6TJW0KEB5t4DR8FFRsdJQQJ99ALACi5YpzXJ3w3AAAFACOGKe6y"
# client = ImageAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

@app.route('/recognize', methods=['POST'])
def recognize_plate():
    # Prepare to capture all outputs
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    if 'image' not in request.files:
        print("No image file provided")
        return jsonify({'error': 'No image file provided', 'logs': buffer.getvalue()}), 400

    file = request.files['image']
    if file.filename == '':
        print("No selected file")
        return jsonify({'error': 'No selected file', 'logs': buffer.getvalue()}), 400

    # Save the uploaded file
    img_path = './uploaded_image.jpg'
    file.save(img_path)
    print(f"Image saved to {img_path}")

    try:
        # YOLO + EasyOCR Processing
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
            for text, score in sorted(ocr_results, key=lambda x: x[1], reverse=True)[:10]:
                print(f"- {text} (Confidence: {score:.2f})")

        # # --- Azure Text Recognition ---
        # with open(img_path, "rb") as f:
        #     image_data = f.read()

        # # Analyze images to extract text
        # result = client.analyze(image_data=image_data, visual_features=[VisualFeatures.READ])

        # print("\n--- Azure Computer Vision Results ---")

        # # Output
        # if result.read is not None:
        # # ---------if any(char.isdigit() for char in line.text) and any(char.isalpha() for char in line.text):        -------------------
        #     for block in result.read.blocks:
        #         for line in block.lines:
        #             print(line.text)
        # else:
        #     print("No text detected.")

    except Exception as e:
        print(f"Error during processing: {str(e)}")
    finally:
        # Reset standard output
        logs = buffer.getvalue()
        sys.stdout = old_stdout
        return jsonify({'logs': logs})

if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=5000)
