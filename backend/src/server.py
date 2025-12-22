from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import torch
from PIL import Image
import io
from typing import Dict
from features import FeatureExtractor
from train import Classifier

device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
extractor = None

# Lifespan to define on startup behavior
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and feature extractor on startup"""
    global model, extractor
    
    print("Loading model...")
    
    # Load classifier
    checkpoint = torch.load("checkpoints/best_model.pth", map_location=device)
    model = Classifier(input_dim=775)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Load feature extractor
    extractor = FeatureExtractor(device=device)
    extractor.load_normalizers("checkpoints/normalizers.npz")
    
    print(f"✓ Model loaded on {device}")
    print(f"✓ Validation accuracy: {checkpoint['val_acc']:.2f}%")

    yield


# INITIALIZE APP
app = FastAPI(
    title="AI Image Detector API",
    description="Detect AI-generated images using hybrid CLIP + frequency analysis",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# FEATURE EXTRACTION HELPER
def extract_features_from_image(image: Image.Image) -> torch.Tensor:
    """Extract features from PIL Image"""
    
    # Resize to 256x256
    image = image.resize((256, 256), Image.BILINEAR)
    
    # Extract CLIP features
    clip_feat = extractor.extract_clip_features(image)
    
    # Extract frequency features
    freq_feat = extractor.extract_frequency_features(image)
    freq_feat = torch.from_numpy(freq_feat).float()
    
    # Extract compression features
    comp_feat = extractor.extract_compression_features(image)
    comp_feat = torch.from_numpy(comp_feat).float()
    
    # Normalize
    if extractor.is_fitted:
        freq_feat = (freq_feat - torch.from_numpy(extractor.freq_mean).float()) / \
                    torch.from_numpy(extractor.freq_std).float()
        comp_feat = (comp_feat - torch.from_numpy(extractor.comp_mean).float()) / \
                    torch.from_numpy(extractor.comp_std).float()
    
    # Combine all features
    features = torch.cat([clip_feat, freq_feat, comp_feat])
    
    return features

# ENDPOINTS
@app.get("/")
async def root():
    """Root endpoint - API info"""
    return {
        "name": "AI Image Detector API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "POST /detect": "Upload image for detection",
            "GET /health": "Health check",
            "GET /stats": "Model statistics"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": device
    }


@app.get("/stats")
async def get_stats():
    """Get model statistics"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    checkpoint = torch.load("checkpoints/best_model.pth", map_location="cpu")
    
    return {
        "model": {
            "parameters": model.get_num_params(),
            "input_dim": 775,
            "architecture": "CLIP + Frequency + Compression → Linear Classifier"
        },
        "performance": {
            "validation_accuracy": f"{checkpoint['val_acc']:.2f}%",
            "validation_auc": f"{checkpoint.get('val_auc', 0):.4f}",
            "training_epoch": checkpoint.get('epoch', 'unknown')
        },
        "features": {
            "clip_dim": 768,
            "frequency_dim": 4,
            "compression_dim": 3
        }
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)) -> Dict:
    
    # Validate model is loaded
    if model is None or extractor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Must be an image."
        )
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Extract features
        features = extract_features_from_image(image)
        features = features.unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            logits = model(features)
            probs = torch.softmax(logits, dim=1)[0]
            
            pred_class = torch.argmax(probs).item()
            confidence = probs[pred_class].item()
        
        # Format response
        result = {
            "prediction": "fake" if pred_class == 1 else "real",
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                "real": round(probs[0].item() * 100, 2),
                "fake": round(probs[1].item() * 100, 2)
            },
            "metadata": {
                "filename": file.filename,
                "size": f"{image.size[0]}x{image.size[1]}",
                "model_device": device
            }
        }
        
        return result
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)