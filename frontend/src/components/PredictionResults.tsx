/**
 * PredictionResults Component
 * Display prediction results with confidence visualization and segmentation
 */

import React from 'react';
import type { SinglePredictionResponse } from '../api';
import SegmentationResult from './SegmentationResult';
import './PredictionResults.css';

interface PredictionResultsProps {
    result: SinglePredictionResponse;
}

const PredictionResults: React.FC<PredictionResultsProps> = ({ result }) => {
    const getTumorDescription = (className: string) => {
        const descriptions: Record<string, string> = {
            glioma_tumor: 'Tumor originating from glial cells',
            meningioma_tumor: 'Tumor originating from the meninges',
            no_tumor: 'No tumor detected - healthy brain tissue',
            pituitary_tumor: 'Tumor of the pituitary gland',
        };
        return descriptions[className] || '';
    };

    const getConfidenceColor = (confidence: number) => {
        if (confidence >= 0.9) return 'var(--success)';
        if (confidence >= 0.7) return 'var(--info)';
        if (confidence >= 0.5) return 'var(--warning)';
        return 'var(--error)';
    };

    const sortedProbabilities = Object.entries(result.all_probabilities)
        .sort(([, a], [, b]) => b - a);

    const isTumor = result.predicted_class !== 'no_tumor';

    return (
        <div className="prediction-results-wrapper">
            <div className="prediction-results fade-in-scale">
                <div className="result-header">
                    <div className="model-badge">{result.model_name.toUpperCase()}</div>
                    <div className="processing-time">
                        ⚡ {result.processing_time_ms.toFixed(0)}ms
                    </div>
                </div>

                <div className="main-prediction">
                    <div className="prediction-icon">
                        {result.predicted_class === 'no_tumor' ? '✅' : '🔬'}
                    </div>
                    <h3 className="predicted-class">
                        {result.predicted_class.replace(/_/g, ' ').toUpperCase()}
                    </h3>
                    <p className="prediction-description">
                        {getTumorDescription(result.predicted_class)}
                    </p>

                    <div className="confidence-circle">
                        <svg className="confidence-svg" viewBox="0 0 100 100">
                            <circle
                                cx="50"
                                cy="50"
                                r="45"
                                fill="none"
                                stroke="var(--bg-tertiary)"
                                strokeWidth="10"
                            />
                            <circle
                                cx="50"
                                cy="50"
                                r="45"
                                fill="none"
                                stroke={getConfidenceColor(result.confidence)}
                                strokeWidth="10"
                                strokeDasharray={`${result.confidence * 283} 283`}
                                strokeLinecap="round"
                                transform="rotate(-90 50 50)"
                                className="confidence-progress"
                            />
                        </svg>
                        <div className="confidence-text">
                            <span className="confidence-value">
                                {(result.confidence * 100).toFixed(1)}%
                            </span>
                            <span className="confidence-label">Confidence</span>
                        </div>
                    </div>
                </div>

                <div className="probabilities-section">
                    <h4>All Probabilities</h4>
                    <div className="probability-bars">
                        {sortedProbabilities.map(([className, probability]) => (
                            <div key={className} className="probability-item">
                                <div className="probability-header">
                                    <span className="class-name">
                                        {className.replace(/_/g, ' ')}
                                    </span>
                                    <span className="probability-value">
                                        {(probability * 100).toFixed(1)}%
                                    </span>
                                </div>
                                <div className="probability-bar">
                                    <div
                                        className="probability-fill"
                                        style={{
                                            width: `${probability * 100}%`,
                                            background: className === result.predicted_class
                                                ? getConfidenceColor(probability)
                                                : 'var(--bg-tertiary)',
                                        }}
                                    />
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* Show segmentation if tumor detected and segmentation data available */}
            {isTumor && result.has_segmentation && result.segmented_image_base64 && (
                <SegmentationResult
                    segmentedImageBase64={result.segmented_image_base64}
                    maskImageBase64={result.mask_image_base64}
                    originalImageBase64={result.original_image_base64}
                    predictedClass={result.predicted_class}
                />
            )}

            {/* Show message if tumor detected but no segmentation */}
            {isTumor && !result.has_segmentation && (
                <div className="no-segmentation-notice fade-in">
                    <span className="notice-icon">ℹ️</span>
                    <p>Segmentation model not available for this analysis.</p>
                </div>
            )}
        </div>
    );
};

export default PredictionResults;
