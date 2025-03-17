# License Plate Recognition Using YOLO and EasyOCR

This project implements an automatic license plate recognition (ALPR) system using YOLO for license plate detection and EasyOCR for optical character recognition (OCR). Additionally, the Azure Custom Vision API is evaluated for comparison in terms of accuracy and robustness.

## Features

- **License Plate Detection:** YOLO is used for detecting license plates in images.
- **Text Recognition:** EasyOCR extracts characters from detected license plates.
- **Comparison with Azure API:** The Azure Custom Vision API is used as a benchmark to compare performance under different conditions.
- **Scenario-Based Evaluation:** The system is tested in different scenarios, including clear images, multi-angle images, and low-light conditions.

## Installation

### Prerequisites

- Python 3.8+
- Required Python libraries (install using `requirements.txt`)
- Node.js (if using the React front-end)
- Flask (for backend API)

### Setup Instructions

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/license-plate-recognition.git
   cd license-plate-recognition

2. Install dependencies:

   ```sh
   pip install -r requirements.txt

3. Run the backend server:

   ```sh
   python app.py

4. Start the frontend (if applicable):

   ```sh
   cd frontend
   npm install
   npm start

## Usage

1. **Upload an Image**  
   - Provide an image containing a license plate.
  
2. **License Plate Detection & Recognition**  
   - The system detects the license plate using YOLO and extracts the text using EasyOCR.
  
3. **Display Results**  
   - Recognition results are presented, including:
     - Detected license plate text
     - Processing time
     - Confidence scores

---

## Dataset Sources

The training and testing datasets used in this project are obtained from the following sources:

- **[Zemris License Plate Dataset](https://www.zemris.fer.hr/projects/LicensePlates/english/results.shtml)**
- **[Kaggle - European License Plates](https://www.kaggle.com/datasets/abdelhamidzakaria/european-license-plates-dataset)**
- **[License Plate Dataset by RobertLucian](https://github.com/RobertLucian/license-plate-dataset/tree/master/dataset)**
- **[Roboflow License Plates (US/EU)](https://public.roboflow.com/object-detection/license-plates-us-eu/)**
- **[Kaggle - Car Plate Detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection?resource=download&select=images)**
