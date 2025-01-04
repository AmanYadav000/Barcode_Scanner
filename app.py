from flask import Flask, request, jsonify
import cv2
from ultralytics import YOLO
import numpy as np
import pyzxing
import os

# Initialize Flask app
app = Flask(__name__)

# Load a pretrained YOLOv8n model
model = YOLO('/content/barcode.pt')

# Initialize the ZXing decoder
reader = pyzxing.BarCodeReader()

# Function to rotate the image by a given angle
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Generate a rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Perform the rotation
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return rotated

# Function to decode barcodes from a rotated image
def decode_rotated_barcode(region):
    for angle in range(0, 360, 30):  # Rotate in 30-degree intervals
        rotated_region = rotate_image(region, angle)

        # Convert rotated region to a temporary file for ZXing processing
        temp_filename = 'temp_rotated.png'
        cv2.imwrite(temp_filename, rotated_region)

        # Decode barcodes using ZXing
        results = reader.decode(temp_filename)
        if results:
            os.remove(temp_filename)  # Clean up temporary file
            for result in results:
                if result.get('raw', ''):
                    return {
                        'data': result['raw'],
                        'format': result['format'],
                        'angle': angle
                    }
    os.remove(temp_filename)  # Clean up temporary file if no barcode found
    return None

@app.route('/decode-barcode', methods=['POST'])
def decode_barcode():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    # Read the uploaded image
    image_file = request.files['image']
    image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Run inference on the image
    results = model(image)

    # Access bounding boxes for the detected objects
    barcodes = []
    for r in results:
        boxes = r.boxes.xyxy.cpu().numpy()

        # Process each detected region
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            # Extract the barcode region
            barcode_region = image[y1:y2, x1:x2]

            # Attempt to decode barcode with rotation
            decoded_data = decode_rotated_barcode(barcode_region)
            if decoded_data:
                barcodes.append(decoded_data)

    if not barcodes:
        return jsonify({'message': 'No barcodes were successfully decoded.'}), 404

    return jsonify({'barcodes': barcodes}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
