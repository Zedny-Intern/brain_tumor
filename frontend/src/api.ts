/**
 * API Service for Brain Tumor Classification & Segmentation
 * Handles all communication with the FastAPI backend
 */

import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'multipart/form-data',
    },
});

export interface PredictionResult {
    predicted_class: string;
    confidence: number;
    all_probabilities: Record<string, number>;
    processing_time_ms: number;
}

export interface SegmentationData {
    has_segmentation: boolean;
    segmented_image_base64?: string;
    mask_image_base64?: string;
    original_image_base64?: string;
}

export interface SinglePredictionResponse extends SegmentationData {
    success: boolean;
    model_name: string;
    predicted_class: string;
    confidence: number;
    all_probabilities: Record<string, number>;
    processing_time_ms: number;
}

export interface BatchPredictionResponse extends SegmentationData {
    success: boolean;
    predictions: Record<string, PredictionResult>;
    consensus_prediction: string;
    average_confidence: number;
}

export interface SegmentationOnlyResponse {
    success: boolean;
    segmented_image_base64: string;
    mask_image_base64: string;
    original_image_base64: string;
    processing_time_ms: number;
}

export interface HealthResponse {
    status: string;
    models_loaded: string[];
    models_failed: string[];
    segmentation_model_loaded: boolean;
}

export interface ModelInfo {
    total_models: number;
    loaded_models: number;
    models: Record<string, any>;
    available_models: string[];
    segmentation_model: {
        loaded: boolean;
        name: string;
        accuracy: string;
    };
    class_names: string[];
}

/**
 * Check API health status
 */
export const checkHealth = async (): Promise<HealthResponse> => {
    const response = await api.get('/health');
    return response.data;
};

/**
 * Get information about all models
 */
export const getModels = async (): Promise<ModelInfo> => {
    const response = await api.get('/models');
    return response.data;
};

/**
 * Get tumor class information
 */
export const getClasses = async () => {
    const response = await api.get('/classes');
    return response.data;
};

/**
 * Predict using a specific model
 */
export const predictWithModel = async (
    modelName: string,
    imageFile: File
): Promise<SinglePredictionResponse> => {
    const formData = new FormData();
    formData.append('file', imageFile);

    const response = await api.post(`/predict/${modelName}`, formData);
    return response.data;
};

/**
 * Predict using all available models
 */
export const predictWithAllModels = async (
    imageFile: File
): Promise<BatchPredictionResponse> => {
    const formData = new FormData();
    formData.append('file', imageFile);

    const response = await api.post('/predict/all', formData);
    return response.data;
};

/**
 * Perform segmentation only (without classification)
 */
export const segmentImage = async (
    imageFile: File
): Promise<SegmentationOnlyResponse> => {
    const formData = new FormData();
    formData.append('file', imageFile);

    const response = await api.post('/segment', formData);
    return response.data;
};

export default api;
