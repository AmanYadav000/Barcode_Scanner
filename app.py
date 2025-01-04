from fastapi import FastAPI, File, UploadFile
from pyzbar.pyzbar import decode
import cv2
import numpy as np
from io import BytesIO
from fastapi.responses import JSONResponse
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Function to process and detect barcodes from the image
def process_barcode(img: BytesIO):
    # Convert the uploaded image into a format suitable for OpenCV (numpy array)
    image = np.array(bytearray(img.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # Decode barcodes in the image using pyzbar
    barcodes = decode(image)
    barcode_dict = {}

    # Loop over detected barcodes and store results in a dictionary
    for barcode in barcodes:
        (x, y, w, h) = barcode.rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

        barcode_data = barcode.data.decode("utf-8")
        barcode_type = barcode.type

        # Append data to the dictionary
        barcode_dict[barcode_data] = barcode_type
        print(f"[INFO] Found {barcode_type} barcode: {barcode_data}")
    
    return barcode_dict

# API endpoint to handle barcode detection from an uploaded image
@app.post("/decode-barcode")
async def decode_barcode(image: UploadFile = File(...)):
    # Read image content
    img = await image.read()

    # Process barcode detection
    barcode_data = process_barcode(BytesIO(img))

    if barcode_data:
        return JSONResponse(content=barcode_data)
    else:
        return JSONResponse(content={"message": "No barcodes detected"}, status_code=404)

# Add this block for running the app locally
if __name__ == "__main__":
    # This is for local development
    uvicorn.run(app, host="0.0.0.0", port=8000)

