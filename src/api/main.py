from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
import io
import torch
from torchvision import transforms
import logging
from pathlib import Path
import sys

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.uinet import UIDetectionNet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize model
try:
    model_path = Path(__file__).parent.parent.parent / 'models' / 'trained_model.pth'
    model = UIDetectionNet()
    model.load_state_dict(torch.load(str(model_path), map_location='cpu'))
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

# Define the transform pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.get("/test")
async def test():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze_ui(file: UploadFile = File(...)):
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Apply transformations
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        # Process image
        with torch.no_grad():
            output = model(image_tensor)
            
        # Convert output to desired format
        ui_elements = {
            "widgets": [
                {
                    "type": "button",
                    "properties": {
                        "x": float(output[0][0]),
                        "y": float(output[0][1]),
                        "width": float(output[0][2]),
                        "height": float(output[0][3]),
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
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))