from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO
import pyzxing
import os

# Initialize FastAPI app
app = FastAPI()

# Load a pretrained YOLOv8n model
model = YOLO("barcode.pt")

# Initialize the ZXing decoder
reader = pyzxing.BarCodeReader()

# Function to rotate the image by a given angle
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return rotated

# Function to decode barcodes from a rotated image
def decode_rotated_barcode(region):
    for angle in range(0, 360, 30):  # Rotate in 30-degree intervals
        rotated_region = rotate_image(region, angle)

        # Convert rotated region to a temporary file for ZXing processing
        temp_filename = "temp_rotated.png"
        cv2.imwrite(temp_filename, rotated_region)

        # Decode barcodes using ZXing
        results = reader.decode(temp_filename)
        if results:
            os.remove(temp_filename)  # Clean up temporary file
            for result in results:
                if result.get("raw", ""):
                    return {
                        "data": result["raw"],
                        "format": result["format"],
                        "angle": angle,
                    }
    os.remove(temp_filename)  # Clean up temporary file if no barcode found
    return None

@app.post("/decode-barcode/")
async def decode_barcode(image: UploadFile = File(...)):
    try:
        # Read the uploaded image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

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
            return JSONResponse(
                status_code=404,
                content={"message": "No barcodes were successfully decoded."},
            )

        return {"barcodes": barcodes}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=10000)
