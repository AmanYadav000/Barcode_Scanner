import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
from ultralytics import YOLO
import tempfile
import zxing

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the YOLOv8n model
model = YOLO('barcode.pt')

# Function to decode barcode using ZXing
def decode_barcode_with_zxing(image_path):
    # Initialize ZXing Barcode Reader
    reader = zxing.BarCodeReader()
    barcode = reader.decode(image_path)

    if barcode:
        return {
            "data": barcode.raw,
            "type": barcode.format
        }
    return None

@app.get('/')
def index():
    return {"message": "Barcode Detection API - Upload an image to /decode"}

@app.post('/decode')
async def decode_image(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(contents)
            temp_file_path = temp_file.name

        # Load the image using OpenCV
        frame = cv2.imread(temp_file_path)

        # Run YOLO inference
        results = model(frame)

        barcode_detected = False
        barcodes_info = []

        # Access bounding boxes for the detected objects
        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)

                # Extract the barcode region
                barcode_region = frame[y1:y2, x1:x2]

                # Save the region to a temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as region_file:
                    cv2.imwrite(region_file.name, barcode_region)

                    # Decode the barcode using ZXing
                    decoded = decode_barcode_with_zxing(region_file.name)

                    if decoded:
                        barcodes_info.append({
                            "data": decoded["data"],
                            "type": decoded["type"]
                        })
                        barcode_detected = True

        # Response
        if barcode_detected:
            return {
                "message": "Barcodes decoded successfully.",
                "barcodes": barcodes_info
            }
        else:
            return {"message": "No barcodes were successfully decoded."}

    except Exception as e:
        return {"error": str(e)}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
