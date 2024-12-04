from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io
import torch
from ..models.uinet import UIDetectionNet

app = FastAPI()

model = UIDetectionNet()
model.load_state_dict(torch.load('models/trained_model.pth'))
model.eval()

@app.post("/analyze")
async def analyze_ui(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    # Process image and generate JSON
    with torch.no_grad():
        output = model(image)
    
    # Convert model output to JSON format
    ui_elements = {
        "widgets": [
            {
                "type": "button",
                "properties": {
                    "x": 100,
                    "y": 200,
                    "width": 120,
                    "height": 40,
                    "text": "Submit",
                    "style": {
                        "backgroundColor": "#007bff",
                        "color": "#ffffff"  
                    }
                }
            }
        ]
    }
    
    return ui_elements