"""FastAPI backend for AMSI-Net LULC Recognition System."""

import torch
import time
import base64
import io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Any

from model_loader import get_model_loader
from preprocessing import get_preprocessor
from class_mappings import get_class_label
from explainers import generate_all_explanations

app = FastAPI(
    title="AMSI-Net LULC Recognition API",
    description="Multi-label Remote Sensing Image Classification with AMSI-Net",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ClassPrediction(BaseModel):
    class_index: int
    class_label: str
    confidence: float

class PredictionResponse(BaseModel):
    predicted_labels: List[str]
    all_predictions: List[ClassPrediction]
    explainability_maps: Dict[str, str]
    uncertainty: float
    inference_time_ms: float
    image_info: Dict[str, Any]

class MobileRequest(BaseModel):
    image: str = Field(..., alias="image_b64")

    # Allow aliases in case the mobile app uses different names
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "image": "base64_string_here"
            }
        }
    )

from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi import Request

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    body = await request.body()
    print(f"--- 422 Validation Error ---")
    print(f"Method: {request.method} URL: {request.url}")
    print(f"Body: {body.decode()[:500]}...") # Print first 500 chars
    print(f"Errors: {exc.errors()}")
    print(f"---------------------------")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body_received": body.decode()[:100]},
    )

# Constants
VALID_IMAGE_FORMATS = {"image/jpeg", "image/png", "image/jpg"}
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
CONFIDENCE_THRESHOLD = 0.5

def run_inference(image_bytes: bytes) -> PredictionResponse:
    """Core inference logic reused across endpoints."""
    try:
        preprocessor = get_preprocessor()
        loader = get_model_loader()
        
        # 1. Preprocess
        image_info = preprocessor.get_image_info(image_bytes)
        image_tensor = preprocessor.preprocess(image_bytes)
        image_tensor = image_tensor.to(loader.get_device())
        
        # 2. Inference
        start_time = time.time()
        model = loader.load_model()
        
        with torch.no_grad():
            outputs = model(image_tensor)
            logits = outputs['logits']
            uncertainty = outputs['uncertainty'].item()
            
            # Multi-label probability
            probabilities = torch.sigmoid(logits).squeeze()
            
        inference_time = (time.time() - start_time) * 1000
        
        # 3. Process results
        all_predictions = []
        predicted_labels = []
        
        for i, prob in enumerate(probabilities.tolist()):
            label = get_class_label(i)
            conf = float(prob)
            
            pred_obj = ClassPrediction(
                class_index=i,
                class_label=label,
                confidence=conf
            )
            all_predictions.append(pred_obj)
            
            if conf >= CONFIDENCE_THRESHOLD:
                predicted_labels.append(label)
        
        # If no label exceeds threshold, take the top-1
        if not predicted_labels:
            top_idx = torch.argmax(probabilities).item()
            predicted_labels.append(get_class_label(top_idx))
            
        # 4. Generate explanations
        explain_maps = generate_all_explanations(model, preprocessor, image_tensor, image_bytes)
        
        sorted_preds = sorted(all_predictions, key=lambda x: x.confidence, reverse=True)
        
        # LOGGING: Print detailed output to console/logs
        print(f"\n--- AMSI-Net Inference Results ---")
        print(f"Predicted Labels: {predicted_labels}")
        print(f"Uncertainty Score: {uncertainty:.6f}")
        print(f"Inference Latency: {inference_time:.2f}ms")
        print(f"Top 5 Predictions:")
        for p in sorted_preds[:5]:
            print(f"  - {p.class_label}: {p.confidence:.4f}")
        print(f"----------------------------------\n")
        
        return PredictionResponse(
            predicted_labels=predicted_labels,
            all_predictions=sorted_preds,
            explainability_maps=explain_maps,
            uncertainty=uncertainty,
            inference_time_ms=inference_time,
            image_info=image_info
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def validate_image(file: UploadFile) -> bytes:
    """Validate and read image file."""
    if file.content_type not in VALID_IMAGE_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image format. Supported: JPEG, PNG. Got: {file.content_type}"
        )
    
    content = file.file.read()
    if len(content) > MAX_IMAGE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Image too large. Max size: {MAX_IMAGE_SIZE / 1024 / 1024}MB"
        )
    return content

@app.get("/")
async def root():
    return {"status": "ok", "message": "AMSI-Net API is running"}

@app.get("/health")
async def health():
    loader = get_model_loader()
    return {
        "status": "healthy",
        "device": str(loader.get_device()),
        "model_loaded": loader.model is not None
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """Standard multipart prediction (Web)."""
    image_bytes = validate_image(file)
    return run_inference(image_bytes)

@app.post("/predict_mobile", response_model=PredictionResponse)
async def predict_mobile(request: MobileRequest):
    """Base64 prediction (Mobile)."""
    try:
        # Strip potential base64 prefix (e.g. "data:image/jpeg;base64,")
        base64_str = request.image
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
            
        image_bytes = base64.b64decode(base64_str)
        return run_inference(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")

@app.get("/info")
async def info():
    return {
        "model": "AMSI-Net",
        "task": "Multi-label LULC Recognition",
        "dataset": "MLRSNet (60 classes)",
        "input_size": "224x224",
        "features": ["GradCAM", "GradCAM++", "LIME", "Saliency", "Uncertainty Estimation"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
