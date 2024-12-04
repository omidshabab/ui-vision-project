# UI Detection Model

## Introduction
This project implements a UI detection model using FastAPI and PyTorch. The model is designed to analyze UI elements from images and return their properties in a structured JSON format. It leverages a custom neural network architecture for feature extraction and classification.

## Files and Folders Structure
```
.
├── data
│   ├── annotations
│   │   └── screenshot1.json
│   └── images
├── models
│   └── uinet.py
├── src
│   ├── api
│   │   └── main.py
│   ├── data
│   │   └── dataset.py
│   └── train.py
├── requirements.txt
└── .gitignore
```

### Description of Files
- **data/annotations/screenshot1.json**: Contains the annotations for UI elements in JSON format.
- **data/images/**: Directory to store images for training and analysis.
- **models/uinet.py**: Defines the `UIDetectionNet` class, which is the neural network architecture used for UI detection.
- **src/api/main.py**: Contains the FastAPI application and the endpoint for analyzing images.
- **src/data/dataset.py**: Implements the `UIDataset` class for loading images and their corresponding annotations.
- **src/train.py**: Script to train the model using the dataset.
- **requirements.txt**: Lists the dependencies required to run the project.

## How to Start Using It

### Prerequisites
Make sure you have Python 3.8 or higher installed. You can create a virtual environment and install the required packages using the following commands:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install the required packages
pip install -r requirements.txt
```

### Training the Model
To train the model, ensure that you have your images in the `data/images` directory and the corresponding annotations in the `data/annotations` directory. Then, run the following command:

```bash
python src/train.py
```

This will start the training process, and the model will be saved to `models/trained_model.pth` after each epoch.

### Running the Server
To start the FastAPI server, run the following command:

```bash
uvicorn src.api.main:app --reload
```

This will start the server at `http://127.0.0.1:8000`. You can access the interactive API documentation at `http://127.0.0.1:8000/docs`.

## API Endpoint
- **POST /analyze**: This endpoint accepts an image file and returns the detected UI elements in JSON format.

### Example Request
You can use tools like Postman or cURL to test the API. Here’s an example using cURL:

```bash
curl -X POST "http://127.0.0.1:8000/analyze" -F "file=@path_to_your_image.jpg"
```

### Example Response
The response will be a JSON object containing the detected UI elements:

```json
{
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
```

## Additional Notes
- Ensure that your images and annotations are correctly formatted to avoid errors during training or analysis.
- You can modify the model architecture in `models/uinet.py` to experiment with different configurations.

Feel free to contribute to the project or raise issues if you encounter any problems!
