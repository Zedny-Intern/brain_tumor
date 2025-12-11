"""
Brain Tumor Classification & Segmentation - FastAPI Deployment
===============================================================
This module provides a complete REST API for brain tumor classification
using 5 different deep learning models and automatic tumor segmentation
using a U-Net model when tumors are detected.

Author: Brain Tumor Classification & Segmentation Team
Last Updated: December 2025
"""

import os
import io
import json
import base64
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import cv2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== CONFIGURATION ====================
IMG_SIZE = 224  # Classification model input size
SEGMENTATION_IMG_SIZE = 256  # U-Net segmentation model input size
CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
TUMOR_CLASSES = ['glioma_tumor', 'meningioma_tumor', 'pituitary_tumor']  # Classes that require segmentation
MODELS_DIR = Path(__file__).parent / 'src' / 'models'

# Model file paths - Classification
MODEL_PATHS = {
    'cnn': MODELS_DIR / 'best_brain_tumor_cnn_model.h5',
    'vgg16': MODELS_DIR / 'best_brain_tumor_vgg16_model.h5',
    'vgg19': MODELS_DIR / 'best_brain_tumor_vgg19_model.h5',
    'mobilenet': MODELS_DIR / 'best_brain_tumor_mobilenet_model.h5',
    'resnet': MODELS_DIR / 'best_brain_tumor_resnet_model.h5'
}

# Segmentation model path
SEGMENTATION_MODEL_PATH = MODELS_DIR / 'best_brain_tumor_unet_model.h5'

# ==================== PYDANTIC MODELS ====================
class PredictionResponse(BaseModel):
    """Response model for single prediction"""
    success: bool
    model_name: str
    predicted_class: str
    confidence: float
    all_probabilities: Dict[str, float]
    processing_time_ms: float
    # Segmentation fields (optional)
    has_segmentation: bool = False
    segmented_image_base64: Optional[str] = None
    mask_image_base64: Optional[str] = None
    original_image_base64: Optional[str] = None

class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions from all models"""
    success: bool
    predictions: Dict[str, Dict]
    consensus_prediction: str
    average_confidence: float
    # Segmentation fields (optional)
    has_segmentation: bool = False
    segmented_image_base64: Optional[str] = None
    mask_image_base64: Optional[str] = None
    original_image_base64: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: List[str]
    models_failed: List[str]
    segmentation_model_loaded: bool = False

class SegmentationOnlyResponse(BaseModel):
    """Response model for standalone segmentation"""
    success: bool
    segmented_image_base64: str
    mask_image_base64: str
    original_image_base64: str
    processing_time_ms: float

# ==================== CUSTOM METRICS ====================
def precision_metric(y_true, y_pred):
    """Precision metric for multi-class classification"""
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision

def recall_metric(y_true, y_pred):
    """Recall metric for multi-class classification"""
    true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall

def f1_metric(y_true, y_pred):
    """F1 score metric for multi-class classification"""
    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

# Custom metrics for segmentation (Dice coefficient, IoU)
def dice_coefficient(y_true, y_pred, smooth=1):
    """Dice coefficient for segmentation evaluation"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    """Dice loss for segmentation"""
    return 1 - dice_coefficient(y_true, y_pred)

def iou_metric(y_true, y_pred, smooth=1):
    """Intersection over Union metric"""
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)

# ==================== MODEL MANAGER ====================
class ModelManager:
    """Manages multiple trained models for brain tumor classification and segmentation"""
    
    def __init__(self):
        self.models = {}
        self.model_info = {}
        self.segmentation_model = None
        self.segmentation_model_loaded = False
        
    def load_models(self):
        """Load all available models including segmentation"""
        custom_objects = {
            'precision_metric': precision_metric,
            'recall_metric': recall_metric,
            'f1_metric': f1_metric,
            'dice_coefficient': dice_coefficient,
            'dice_loss': dice_loss,
            'iou_metric': iou_metric
        }
        
        # Load classification models
        for model_name, model_path in MODEL_PATHS.items():
            try:
                if model_path.exists():
                    logger.info(f"Loading {model_name} model from {model_path}")
                    self.models[model_name] = keras.models.load_model(
                        str(model_path),
                        custom_objects=custom_objects
                    )
                    self.model_info[model_name] = {
                        'loaded': True,
                        'path': str(model_path),
                        'size_mb': model_path.stat().st_size / (1024 * 1024)
                    }
                    logger.info(f"✓ {model_name.upper()} loaded successfully")
                else:
                    logger.warning(f"✗ Model file not found: {model_path}")
                    self.model_info[model_name] = {
                        'loaded': False,
                        'error': 'File not found'
                    }
            except Exception as e:
                logger.error(f"✗ Error loading {model_name}: {str(e)}")
                self.model_info[model_name] = {
                    'loaded': False,
                    'error': str(e)
                }
        
        # Load segmentation model (U-Net)
        try:
            if SEGMENTATION_MODEL_PATH.exists():
                logger.info(f"Loading U-Net segmentation model from {SEGMENTATION_MODEL_PATH}")
                self.segmentation_model = keras.models.load_model(
                    str(SEGMENTATION_MODEL_PATH),
                    custom_objects=custom_objects
                )
                self.segmentation_model_loaded = True
                logger.info("✓ U-Net segmentation model loaded successfully")
            else:
                logger.warning(f"✗ Segmentation model not found: {SEGMENTATION_MODEL_PATH}")
        except Exception as e:
            logger.error(f"✗ Error loading segmentation model: {str(e)}")
            self.segmentation_model_loaded = False
        
        loaded_count = len(self.models)
        total_count = len(MODEL_PATHS)
        seg_status = "loaded" if self.segmentation_model_loaded else "not loaded"
        logger.info(f"Loaded {loaded_count}/{total_count} classification models, segmentation model: {seg_status}")
        
    def get_model(self, model_name: str):
        """Get a specific model by name"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded or not found")
        return self.models[model_name]
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names"""
        return list(self.models.keys())
    
    def is_model_available(self, model_name: str) -> bool:
        """Check if a model is available"""
        return model_name in self.models
    
    def get_segmentation_model(self):
        """Get the segmentation model"""
        if not self.segmentation_model_loaded:
            raise ValueError("Segmentation model not loaded")
        return self.segmentation_model

# ==================== IMAGE PREPROCESSING ====================
def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Preprocess uploaded image for classification model prediction
    
    Args:
        image_bytes: Raw image bytes from upload
        
    Returns:
        Preprocessed image array ready for prediction
    """
    try:
        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image = image.resize((IMG_SIZE, IMG_SIZE))
        
        # Convert to numpy array and normalize
        img_array = np.array(image) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

def preprocess_for_segmentation(image_bytes: bytes) -> tuple:
    """
    Preprocess image for U-Net segmentation
    
    Args:
        image_bytes: Raw image bytes from upload
        
    Returns:
        Tuple of (preprocessed array for model, original image for overlay)
    """
    try:
        # Load image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        original_size = image.size
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Save original for overlay
        original_image = np.array(image)
        
        # Resize for segmentation model
        image_resized = image.resize((SEGMENTATION_IMG_SIZE, SEGMENTATION_IMG_SIZE))
        
        # Convert to numpy array and normalize
        img_array = np.array(image_resized) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array, original_image, original_size
        
    except Exception as e:
        logger.error(f"Error preprocessing image for segmentation: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

def create_segmentation_overlay(original_image: np.ndarray, mask: np.ndarray, original_size: tuple) -> tuple:
    """
    Create overlay of segmentation mask on original image
    
    Args:
        original_image: Original image as numpy array
        mask: Predicted segmentation mask
        original_size: Original image size (width, height)
        
    Returns:
        Tuple of (overlay_image, mask_image) as numpy arrays
    """
    # Resize mask to original image size
    mask_resized = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))
    
    # Threshold the mask to binary
    mask_binary = (mask_resized > 0.5).astype(np.uint8) * 255
    
    # Create colored overlay (red for tumor region)
    overlay = original_image.copy()
    
    # Create a colored mask (red/orange for tumor)
    colored_mask = np.zeros_like(original_image)
    colored_mask[:, :, 0] = 255  # Red channel
    colored_mask[:, :, 1] = 100  # Green channel (for orange tint)
    colored_mask[:, :, 2] = 0    # Blue channel
    
    # Apply mask
    mask_3channel = np.stack([mask_resized] * 3, axis=-1)
    
    # Blend overlay with original
    alpha = 0.4  # Transparency
    overlay = np.where(
        mask_3channel > 0.5,
        (1 - alpha) * original_image + alpha * colored_mask,
        original_image
    ).astype(np.uint8)
    
    # Add contour for better visibility
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)  # Green contour
    
    return overlay, mask_binary

def image_to_base64(image: np.ndarray) -> str:
    """Convert numpy array image to base64 string"""
    # Convert to PIL Image
    pil_image = Image.fromarray(image.astype(np.uint8))
    
    # Save to bytes
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    
    # Encode to base64
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# ==================== PREDICTION FUNCTIONS ====================
def predict_single(model, image_array: np.ndarray) -> Dict:
    """
    Make prediction using a single model
    
    Args:
        model: Loaded Keras model
        image_array: Preprocessed image array
        
    Returns:
        Dictionary with prediction results
    """
    import time
    start_time = time.time()
    
    # Make prediction
    predictions = model.predict(image_array, verbose=0)
    
    # Get predicted class and confidence
    predicted_idx = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(predictions[0][predicted_idx])
    
    # Get all probabilities
    all_probabilities = {
        CLASS_NAMES[i]: float(predictions[0][i])
        for i in range(len(CLASS_NAMES))
    }
    
    processing_time = (time.time() - start_time) * 1000  # Convert to ms
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'all_probabilities': all_probabilities,
        'processing_time_ms': processing_time
    }

def perform_segmentation(model_manager: 'ModelManager', image_bytes: bytes) -> Dict:
    """
    Perform tumor segmentation on an image
    
    Args:
        model_manager: ModelManager instance
        image_bytes: Raw image bytes
        
    Returns:
        Dictionary with segmentation results
    """
    import time
    start_time = time.time()
    
    if not model_manager.segmentation_model_loaded:
        return {
            'has_segmentation': False,
            'error': 'Segmentation model not loaded'
        }
    
    try:
        # Preprocess for segmentation
        img_array, original_image, original_size = preprocess_for_segmentation(image_bytes)
        
        # Get segmentation model and predict
        seg_model = model_manager.get_segmentation_model()
        mask_pred = seg_model.predict(img_array, verbose=0)
        
        # Handle different output shapes
        if len(mask_pred.shape) == 4:
            mask = mask_pred[0, :, :, 0]  # Remove batch and channel dimensions
        else:
            mask = mask_pred[0]
        
        # Create overlay
        overlay_image, mask_binary = create_segmentation_overlay(
            original_image, mask, original_size
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            'has_segmentation': True,
            'segmented_image_base64': image_to_base64(overlay_image),
            'mask_image_base64': image_to_base64(mask_binary),
            'original_image_base64': image_to_base64(original_image),
            'processing_time_ms': processing_time
        }
        
    except Exception as e:
        logger.error(f"Segmentation error: {str(e)}")
        return {
            'has_segmentation': False,
            'error': str(e)
        }

def get_consensus_prediction(all_predictions: Dict) -> tuple:
    """
    Get consensus prediction from all models
    
    Args:
        all_predictions: Dictionary of predictions from all models
        
    Returns:
        Tuple of (consensus_class, average_confidence)
    """
    # Count votes for each class
    votes = {}
    confidences = []
    
    for model_pred in all_predictions.values():
        predicted_class = model_pred['predicted_class']
        votes[predicted_class] = votes.get(predicted_class, 0) + 1
        confidences.append(model_pred['confidence'])
    
    # Get class with most votes
    consensus_class = max(votes, key=votes.get)
    average_confidence = np.mean(confidences)
    
    return consensus_class, float(average_confidence)

# ==================== FASTAPI APPLICATION ====================
app = FastAPI(
    title="Brain Tumor Classification & Segmentation API",
    description="REST API for brain tumor classification and automatic tumor segmentation using deep learning models",
    version="2.0.0"
)

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model manager
model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    logger.info("Starting Brain Tumor Classification & Segmentation API...")
    model_manager.load_models()
    logger.info("API ready to serve predictions and segmentation!")

# ==================== API ENDPOINTS ====================

@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Brain Tumor Classification & Segmentation API",
        "version": "2.0.0",
        "description": "Deep learning-based brain tumor classification with automatic tumor segmentation",
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "predict": "/predict/{model_name}",
            "predict_all": "/predict/all",
            "segment": "/segment"
        },
        "features": {
            "classification": "5 models (CNN, VGG16, VGG19, MobileNet, ResNet50)",
            "segmentation": "U-Net model for tumor region highlighting"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    available_models = model_manager.get_available_models()
    failed_models = [
        name for name, info in model_manager.model_info.items()
        if not info.get('loaded', False)
    ]
    
    return HealthResponse(
        status="healthy" if available_models else "unhealthy",
        models_loaded=available_models,
        models_failed=failed_models,
        segmentation_model_loaded=model_manager.segmentation_model_loaded
    )

@app.get("/models")
async def get_models():
    """Get information about all models"""
    return {
        "total_models": len(MODEL_PATHS) + 1,  # +1 for segmentation
        "loaded_models": len(model_manager.models) + (1 if model_manager.segmentation_model_loaded else 0),
        "models": model_manager.model_info,
        "available_models": model_manager.get_available_models(),
        "segmentation_model": {
            "loaded": model_manager.segmentation_model_loaded,
            "name": "U-Net",
            "accuracy": "98%"
        },
        "class_names": CLASS_NAMES
    }

# NOTE: /predict/all MUST come before /predict/{model_name} to avoid routing conflicts
@app.post("/predict/all", response_model=BatchPredictionResponse)
async def predict_all(file: UploadFile = File(...)):
    """
    Predict using all available models and return consensus with segmentation
    
    Args:
        file: Uploaded image file
        
    Returns:
        Predictions from all models with consensus and segmentation (if tumor detected)
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        # Read and preprocess image
        image_bytes = await file.read()
        image_array = preprocess_image(image_bytes)
        
        # Get predictions from all models
        all_predictions = {}
        for model_name in model_manager.get_available_models():
            model = model_manager.get_model(model_name)
            result = predict_single(model, image_array)
            all_predictions[model_name] = result
        
        # Get consensus
        consensus_class, avg_confidence = get_consensus_prediction(all_predictions)
        
        # Perform segmentation if tumor detected
        segmentation_result = {'has_segmentation': False}
        if consensus_class in TUMOR_CLASSES and model_manager.segmentation_model_loaded:
            segmentation_result = perform_segmentation(model_manager, image_bytes)
        
        return BatchPredictionResponse(
            success=True,
            predictions=all_predictions,
            consensus_prediction=consensus_class,
            average_confidence=avg_confidence,
            has_segmentation=segmentation_result.get('has_segmentation', False),
            segmented_image_base64=segmentation_result.get('segmented_image_base64'),
            mask_image_base64=segmentation_result.get('mask_image_base64'),
            original_image_base64=segmentation_result.get('original_image_base64')
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/{model_name}", response_model=PredictionResponse)
async def predict(
    model_name: str,
    file: UploadFile = File(...)
):
    """
    Predict brain tumor type using a specific model with optional segmentation
    
    Args:
        model_name: Name of the model (cnn, vgg16, vgg19, mobilenet, resnet)
        file: Uploaded image file
        
    Returns:
        Prediction results with segmentation if tumor detected
    """
    # Validate model name
    if not model_manager.is_model_available(model_name):
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available models: {model_manager.get_available_models()}"
        )
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        # Read and preprocess image
        image_bytes = await file.read()
        image_array = preprocess_image(image_bytes)
        
        # Get model and make prediction
        model = model_manager.get_model(model_name)
        result = predict_single(model, image_array)
        
        # Perform segmentation if tumor detected
        segmentation_result = {'has_segmentation': False}
        if result['predicted_class'] in TUMOR_CLASSES and model_manager.segmentation_model_loaded:
            segmentation_result = perform_segmentation(model_manager, image_bytes)
        
        return PredictionResponse(
            success=True,
            model_name=model_name,
            **result,
            has_segmentation=segmentation_result.get('has_segmentation', False),
            segmented_image_base64=segmentation_result.get('segmented_image_base64'),
            mask_image_base64=segmentation_result.get('mask_image_base64'),
            original_image_base64=segmentation_result.get('original_image_base64')
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/segment", response_model=SegmentationOnlyResponse)
async def segment_only(file: UploadFile = File(...)):
    """
    Perform tumor segmentation without classification
    
    Args:
        file: Uploaded image file
        
    Returns:
        Segmentation results with overlay and mask images
    """
    if not model_manager.segmentation_model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Segmentation model not loaded"
        )
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail="File must be an image"
        )
    
    try:
        image_bytes = await file.read()
        result = perform_segmentation(model_manager, image_bytes)
        
        if not result.get('has_segmentation'):
            raise HTTPException(
                status_code=500,
                detail=result.get('error', 'Segmentation failed')
            )
        
        return SegmentationOnlyResponse(
            success=True,
            segmented_image_base64=result['segmented_image_base64'],
            mask_image_base64=result['mask_image_base64'],
            original_image_base64=result['original_image_base64'],
            processing_time_ms=result['processing_time_ms']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Segmentation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/classes")
async def get_classes():
    """Get information about tumor classes"""
    return {
        "classes": CLASS_NAMES,
        "tumor_classes": TUMOR_CLASSES,
        "descriptions": {
            "glioma_tumor": "Tumor originating from glial cells - Segmentation available",
            "meningioma_tumor": "Tumor originating from the meninges - Segmentation available",
            "no_tumor": "No tumor detected - healthy brain tissue",
            "pituitary_tumor": "Tumor of the pituitary gland - Segmentation available"
        }
    }

# ==================== MAIN ====================
if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("Brain Tumor Classification & Segmentation API")
    print("=" * 70)
    print("\nStarting server...")
    print("\n📍 API will be available at: http://localhost:8000")
    print("📍 API documentation at: http://localhost:8000/docs")
    print("📍 Alternative docs at: http://localhost:8000/redoc")
    print("\n🧠 Classification: 5 models (CNN, VGG16, VGG19, MobileNet, ResNet50)")
    print("🔬 Segmentation: U-Net model (98% accuracy)")
    print("\n" + "=" * 70)
    
    uvicorn.run(
        "deployment:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
