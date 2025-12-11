/**
 * ModelComparison Component
 * Display and compare results from all models with segmentation
 */

import React from 'react';
import type { BatchPredictionResponse } from '../api';
import SegmentationResult from './SegmentationResult';
import './ModelComparison.css';

interface ModelComparisonProps {
    batchResult: BatchPredictionResponse;
}

const ModelComparison: React.FC<ModelComparisonProps> = ({ batchResult }) => {
    const modelOrder = ['cnn', 'vgg16', 'vgg19', 'mobilenet', 'resnet'];

    const getModelColor = (index: number) => {
        const colors = [
            'hsl(210, 100%, 50%)',
            'hsl(280, 100%, 60%)',
            'hsl(340, 100%, 55%)',
            'hsl(160, 100%, 50%)',
            'hsl(30, 100%, 55%)',
        ];
        return colors[index % colors.length];
    };

    const isTumor = batchResult.consensus_prediction !== 'no_tumor';

    return (
        <div className="model-comparison-wrapper">
            <div className="model-comparison fade-in">
                <div className="comparison-header">
                    <h2>🏆 Model Comparison</h2>
                    <div className="consensus-badge">
                        <span className="consensus-label">Consensus:</span>
                        <span className="consensus-value">
                            {batchResult.consensus_prediction.replace(/_/g, ' ').toUpperCase()}
                        </span>
                        <span className="consensus-confidence">
                            {(batchResult.average_confidence * 100).toFixed(1)}%
                        </span>
                    </div>
                </div>

                <div className="models-grid">
                    {modelOrder.map((modelName, index) => {
                        const prediction = batchResult.predictions[modelName];
                        if (!prediction) return null;

                        const isConsensus = prediction.predicted_class === batchResult.consensus_prediction;

                        return (
                            <div
                                key={modelName}
                                className={`model-card ${isConsensus ? 'consensus' : ''}`}
                                style={{ animationDelay: `${index * 0.1}s` }}
                            >
                                <div className="model-card-header">
                                    <h3 style={{ color: getModelColor(index) }}>
                                        {modelName.toUpperCase()}
                                    </h3>
                                    {isConsensus && <span className="consensus-icon">✓</span>}
                                </div>

                                <div className="model-prediction">
                                    <div className="prediction-label">Prediction</div>
                                    <div className="prediction-text">
                                        {prediction.predicted_class.replace(/_/g, ' ')}
                                    </div>
                                </div>

                                <div className="model-confidence">
                                    <div className="confidence-bar-container">
                                        <div
                                            className="confidence-bar-fill"
                                            style={{
                                                width: `${prediction.confidence * 100}%`,
                                                background: getModelColor(index),
                                            }}
                                        />
                                    </div>
                                    <div className="confidence-percentage">
                                        {(prediction.confidence * 100).toFixed(1)}%
                                    </div>
                                </div>

                                <div className="processing-info">
                                    ⚡ {prediction.processing_time_ms.toFixed(0)}ms
                                </div>
                            </div>
                        );
                    })}
                </div>

                <div className="detailed-comparison glass-card mt-lg">
                    <h3>Detailed Probability Comparison</h3>
                    <div className="comparison-table">
                        <div className="table-header">
                            <div className="table-cell">Model</div>
                            {Object.keys(batchResult.predictions[modelOrder[0]].all_probabilities).map((className) => (
                                <div key={className} className="table-cell">
                                    {className.replace(/_/g, ' ')}
                                </div>
                            ))}
                        </div>
                        {modelOrder.map((modelName, index) => {
                            const prediction = batchResult.predictions[modelName];
                            if (!prediction) return null;

                            return (
                                <div key={modelName} className="table-row">
                                    <div className="table-cell model-name" style={{ color: getModelColor(index) }}>
                                        {modelName.toUpperCase()}
                                    </div>
                                    {Object.values(prediction.all_probabilities).map((prob, i) => (
                                        <div key={i} className="table-cell">
                                            <span className="probability-chip">
                                                {(prob * 100).toFixed(1)}%
                                            </span>
                                        </div>
                                    ))}
                                </div>
                            );
                        })}
                    </div>
                </div>
            </div>

            {/* Show segmentation if tumor detected and segmentation data available */}
            {isTumor && batchResult.has_segmentation && batchResult.segmented_image_base64 && (
                <SegmentationResult
                    segmentedImageBase64={batchResult.segmented_image_base64}
                    maskImageBase64={batchResult.mask_image_base64}
                    originalImageBase64={batchResult.original_image_base64}
                    predictedClass={batchResult.consensus_prediction}
                />
            )}

            {/* Show message if tumor detected but no segmentation */}
            {isTumor && !batchResult.has_segmentation && (
                <div className="no-segmentation-notice fade-in">
                    <span className="notice-icon">ℹ️</span>
                    <p>Segmentation model not available for this analysis.</p>
                </div>
            )}
        </div>
    );
};

export default ModelComparison;
