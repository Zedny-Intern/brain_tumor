/**
 * ImageUploader Component
 * Drag-and-drop image upload with preview
 */

import React, { useState, useRef } from 'react';
import './ImageUploader.css';

interface ImageUploaderProps {
    onImageSelect: (file: File) => void;
    selectedImage: File | null;
}

const ImageUploader: React.FC<ImageUploaderProps> = ({ onImageSelect, selectedImage }) => {
    const [isDragging, setIsDragging] = useState(false);
    const [previewUrl, setPreviewUrl] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = () => {
        setIsDragging(false);
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    };

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const files = e.target.files;
        if (files && files.length > 0) {
            handleFile(files[0]);
        }
    };

    const handleFile = (file: File) => {
        if (file.type.startsWith('image/')) {
            onImageSelect(file);

            // Create preview
            const reader = new FileReader();
            reader.onloadend = () => {
                setPreviewUrl(reader.result as string);
            };
            reader.readAsDataURL(file);
        } else {
            alert('Please select a valid image file');
        }
    };

    const handleClick = () => {
        fileInputRef.current?.click();
    };

    const handleClear = () => {
        onImageSelect(null as any);
        setPreviewUrl(null);
        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }
    };

    return (
        <div className="image-uploader">
            <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                style={{ display: 'none' }}
            />

            {!previewUrl ? (
                <div
                    className={`upload-zone ${isDragging ? 'drag-over' : ''}`}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                    onDrop={handleDrop}
                    onClick={handleClick}
                >
                    <div className="upload-icon">📤</div>
                    <h3>Upload MRI Image</h3>
                    <p>Drag and drop or click to select</p>
                    <p className="upload-hint">Supports: JPG, PNG, JPEG</p>
                </div>
            ) : (
                <div className="preview-container fade-in-scale">
                    <div className="preview-header">
                        <h4>Selected Image</h4>
                        <button onClick={handleClear} className="btn-clear">
                            ✕ Change Image
                        </button>
                    </div>
                    <img src={previewUrl} alt="Preview" className="preview-image" />
                    <p className="file-info">{selectedImage?.name}</p>
                </div>
            )}
        </div>
    );
};

export default ImageUploader;
