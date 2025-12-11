/**
 * SegmentationResult Component
 * Display tumor segmentation results with overlay visualization
 */

import React, { useState } from 'react';
import './SegmentationResult.css';

interface SegmentationResultProps {
    segmentedImageBase64: string;
    maskImageBase64?: string;
    originalImageBase64?: string;
    predictedClass: string;
}

const SegmentationResult: React.FC<SegmentationResultProps> = ({
    segmentedImageBase64,
    maskImageBase64,
    originalImageBase64,
    predictedClass
}) => {
    const [activeView, setActiveView] = useState<'overlay' | 'mask' | 'original'>('overlay');
    const [isZoomed, setIsZoomed] = useState(false);

    const getImageSrc = () => {
        switch (activeView) {
            case 'overlay':
                return `data:image/png;base64,${segmentedImageBase64}`;
            case 'mask':
                return maskImageBase64
                    ? `data:image/png;base64,${maskImageBase64}`
                    : `data:image/png;base64,${segmentedImageBase64}`;
            case 'original':
                return originalImageBase64
                    ? `data:image/png;base64,${originalImageBase64}`
                    : `data:image/png;base64,${segmentedImageBase64}`;
            default:
                return `data:image/png;base64,${segmentedImageBase64}`;
        }
    };

    const getTumorColor = () => {
        switch (predictedClass) {
            case 'glioma_tumor':
                return 'var(--error)';
            case 'meningioma_tumor':
                return 'var(--warning)';
            case 'pituitary_tumor':
                return 'var(--info)';
            default:
                return 'var(--accent)';
        }
    };

    return (
        <div className="segmentation-result fade-in-scale">
            <div className="segmentation-header">
                <div className="segmentation-title">
                    <span className="segmentation-icon">🔬</span>
                    <h3>Tumor Segmentation</h3>
                </div>
                <div className="segmentation-badge" style={{ backgroundColor: getTumorColor() }}>
                    {predictedClass.replace(/_/g, ' ').toUpperCase()}
                </div>
            </div>

            <div className="segmentation-description">
                <p>
                    The highlighted region shows the detected tumor area.
                    Use the view options below to compare the original image with the segmentation.
                </p>
            </div>

            <div className="view-controls">
                <button
                    className={`view-btn ${activeView === 'overlay' ? 'active' : ''}`}
                    onClick={() => setActiveView('overlay')}
                >
                    <span className="btn-icon">🎯</span>
                    Overlay
                </button>
                <button
                    className={`view-btn ${activeView === 'mask' ? 'active' : ''}`}
                    onClick={() => setActiveView('mask')}
                    disabled={!maskImageBase64}
                >
                    <span className="btn-icon">⬜</span>
                    Mask Only
                </button>
                <button
                    className={`view-btn ${activeView === 'original' ? 'active' : ''}`}
                    onClick={() => setActiveView('original')}
                    disabled={!originalImageBase64}
                >
                    <span className="btn-icon">🖼️</span>
                    Original
                </button>
            </div>

            <div
                className={`segmentation-image-container ${isZoomed ? 'zoomed' : ''}`}
                onClick={() => setIsZoomed(!isZoomed)}
            >
                <img
                    src={getImageSrc()}
                    alt={`${activeView} view of brain MRI`}
                    className="segmentation-image"
                />
                <div className="zoom-hint">
                    {isZoomed ? '🔍 Click to zoom out' : '🔍 Click to zoom in'}
                </div>
            </div>

            <div className="segmentation-legend">
                <div className="legend-item">
                    <div className="legend-color" style={{ backgroundColor: 'rgba(255, 100, 0, 0.6)' }}></div>
                    <span>Detected Tumor Region</span>
                </div>
                <div className="legend-item">
                    <div className="legend-color" style={{ backgroundColor: '#00ff00', border: '2px solid #00ff00' }}></div>
                    <span>Tumor Boundary</span>
                </div>
            </div>

            <div className="segmentation-info">
                <div className="info-card">
                    <span className="info-icon">📊</span>
                    <div className="info-content">
                        <strong>U-Net Model</strong>
                        <span>98% Accuracy</span>
                    </div>
                </div>
                <div className="info-card">
                    <span className="info-icon">🧠</span>
                    <div className="info-content">
                        <strong>Segmentation</strong>
                        <span>Precise tumor localization</span>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default SegmentationResult;
