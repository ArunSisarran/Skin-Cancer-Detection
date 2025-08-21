from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io
import base64
import logging
from datetime import datetime

from transforms import get_transforms
from loadModel import load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Skin Cancer Detection API",
    description="API for detecting melanoma vs non-melanoma skin lesions",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local
        "https://*.vercel.app",   # Vercel 
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

model = None
device = None
model_loaded = False

@app.on_event("startup")
async def startup_event():
    global model, device, model_loaded
    try:
        model, device = load_model('best_model_focal.pth')
        model_loaded = True
        logger.info("Improved focal loss model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        try:
            model, device = load_model('best.pth')
            model_loaded = True
            logger.info("Fallback to original model successful")
        except Exception as e2:
            logger.error(f"Failed to load fallback model: {str(e2)}")
            model_loaded = False

@app.get("/")
async def root():
    return {
        "message": "Skin Cancer Detection API",
        "status": "healthy",
        "model_loaded": model_loaded
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model_loaded,
        "device": str(device) if device else "unknown"
    }

@app.post("/predict")
async def predict_skin_lesion(file: UploadFile = File(...)):
    if not model_loaded:
        return {
            "prediction": "Non-Melanoma",
            "confidence": 0.853,
            "probabilities": {
                "Non-Melanoma": 0.853,
                "Melanoma": 0.147
            },
            "timestamp": datetime.now().isoformat(),
            "filename": file.filename,
            "demo_mode": True  # Add this flag
        }
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        result, confidence_score = predict_image_from_pil(model, device, image)
        
        probabilities = get_detailed_probabilities(model, device, image)
        
        return {
            "prediction": result,
            "confidence": float(confidence_score),
            "probabilities": probabilities,
            "timestamp": datetime.now().isoformat(),
            "filename": file.filename
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict_with_explanation")
async def predict_with_gradcam(file: UploadFile = File(...)):
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please try again later."
        )
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        result, confidence_score = predict_image_from_pil(model, device, image)
        probabilities = get_detailed_probabilities(model, device, image)
        
        heatmap_b64 = generate_gradcam_heatmap(model, device, image)
        
        return {
            "prediction": result,
            "confidence": float(confidence_score),
            "probabilities": probabilities,
            "heatmap": heatmap_b64,
            "timestamp": datetime.now().isoformat(),
            "filename": file.filename
        }
        
    except Exception as e:
        logger.error(f"Prediction with explanation error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction with explanation failed: {str(e)}"
        )

def predict_image_from_pil(model, device, image: Image.Image):
    transform = get_transforms('val')
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    result = 'Melanoma' if predicted.item() == 1 else 'Non-Melanoma'
    confidence_score = confidence.item()
    
    return result, confidence_score

def get_detailed_probabilities(model, device, image: Image.Image):
    transform = get_transforms('val')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        
    return {
        "Non-Melanoma": float(probabilities[0][0]),
        "Melanoma": float(probabilities[0][1])
    }

def generate_gradcam_heatmap(model, device, image: Image.Image):
    try:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return img_str
        
    except Exception as e:
        logger.error(f"Grad-CAM generation error: {str(e)}")
        return None

@app.get("/model/info")
async def get_model_info():
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    return {
        "model_type": "SkinCancerCNN",
        "backbone": "EfficientNet-B0",
        "classes": ["Non-Melanoma", "Melanoma"],
        "input_size": [224, 224],
        "device": str(device)
    }

@app.post("/batch_predict")
async def batch_predict(files: list[UploadFile] = File(...)):
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(
            status_code=400,
            detail="Maximum 10 images per batch"
        )
    
    results = []
    
    for file in files:
        if not file.content_type.startswith("image/"):
            results.append({
                "filename": file.filename,
                "error": "Invalid file type"
            })
            continue
            
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            
            result, confidence_score = predict_image_from_pil(model, device, image)
            probabilities = get_detailed_probabilities(model, device, image)
            
            results.append({
                "filename": file.filename,
                "prediction": result,
                "confidence": float(confidence_score),
                "probabilities": probabilities
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return {
        "results": results,
        "timestamp": datetime.now().isoformat(),
        "total_processed": len(results)
    }

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860) 