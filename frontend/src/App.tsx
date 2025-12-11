/**
 * Main App Component
 * Brain Tumor Classification & Segmentation Application
 */

import { useState } from 'react';
import Header from './components/Header';
import ImageUploader from './components/ImageUploader';
import PredictionResults from './components/PredictionResults';
import ModelComparison from './components/ModelComparison';

import { predictWithModel, predictWithAllModels, type SinglePredictionResponse, type BatchPredictionResponse } from './api';
import './App.css';

function App() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [selectedModel, setSelectedModel] = useState<string>('all');
  const [isLoading, setIsLoading] = useState(false);
  const [singleResult, setSingleResult] = useState<SinglePredictionResponse | null>(null);
  const [batchResult, setBatchResult] = useState<BatchPredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const models = [
    { id: 'all', name: 'All Models (Comparison)', icon: '🏆' },
    { id: 'cnn', name: 'Custom CNN', icon: '🧠' },
    { id: 'vgg16', name: 'VGG16', icon: '🔬' },
    { id: 'vgg19', name: 'VGG19', icon: '🔬' },
    { id: 'mobilenet', name: 'MobileNet', icon: '⚡' },
    { id: 'resnet', name: 'ResNet50', icon: '🎯' },
  ];

  const handlePredict = async () => {
    if (!selectedImage) {
      setError('Please select an image first');
      return;
    }

    setIsLoading(true);
    setError(null);
    setSingleResult(null);
    setBatchResult(null);

    try {
      if (selectedModel === 'all') {
        const result = await predictWithAllModels(selectedImage);
        setBatchResult(result);
      } else {
        const result = await predictWithModel(selectedModel, selectedImage);
        setSingleResult(result);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Prediction failed');
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedImage(null);
    setSingleResult(null);
    setBatchResult(null);
    setError(null);
  };

  return (
    <div className="app">
      <Header />

      <main className="main-content">
        <div className="container">
          {/* Hero Section */}
          <section id="home" className="hero-section section">
            <div className="hero-content">
              <h1 className="fade-in">
                AI-Powered Brain Tumor Classification & Segmentation
              </h1>
              <p className="hero-subtitle fade-in">
                Upload an MRI image and let our advanced deep learning models identify brain tumor types
                with high accuracy, and visualize tumor regions with automatic segmentation.
              </p>
              <div className="hero-stats fade-in">
                <div className="stat-card glass-card">
                  <div className="stat-icon">🤖</div>
                  <div className="stat-number">6</div>
                  <div className="stat-label">AI Models</div>
                </div>
                <div className="stat-card glass-card">
                  <div className="stat-icon">🎯</div>
                  <div className="stat-number">4</div>
                  <div className="stat-label">Tumor Types</div>
                </div>
                <div className="stat-card glass-card">
                  <div className="stat-icon">🔬</div>
                  <div className="stat-number">98%</div>
                  <div className="stat-label">Segmentation Accuracy</div>
                </div>
                <div className="stat-card glass-card">
                  <div className="stat-icon">⚡</div>
                  <div className="stat-number">&lt;2s</div>
                  <div className="stat-label">Analysis Time</div>
                </div>
              </div>
            </div>
          </section>

          {/* Prediction Section */}
          <section id="predict" className="prediction-section section">
            <h2 className="section-title text-center">Start Classification & Segmentation</h2>

            <div className="prediction-container">
              <div className="prediction-controls glass-card">
                <ImageUploader
                  onImageSelect={setSelectedImage}
                  selectedImage={selectedImage}
                />

                <div className="model-selector">
                  <label className="selector-label">Select Model</label>
                  <div className="model-options">
                    {models.map((model) => (
                      <button
                        key={model.id}
                        className={`model-option ${selectedModel === model.id ? 'active' : ''}`}
                        onClick={() => setSelectedModel(model.id)}
                      >
                        <span className="model-icon">{model.icon}</span>
                        <span className="model-name">{model.name}</span>
                      </button>
                    ))}
                  </div>
                </div>

                <div className="action-buttons">
                  <button
                    className="btn btn-primary"
                    onClick={handlePredict}
                    disabled={!selectedImage || isLoading}
                  >
                    {isLoading ? (
                      <>
                        <div className="spinner-small"></div>
                        <span>Processing...</span>
                      </>
                    ) : (
                      <>
                        <span>🔍</span>
                        <span>Analyze Image</span>
                      </>
                    )}
                  </button>

                  {(selectedImage || singleResult || batchResult) && (
                    <button
                      className="btn btn-secondary"
                      onClick={handleReset}
                      disabled={isLoading}
                    >
                      🔄 Start New
                    </button>
                  )}
                </div>

                {error && (
                  <div className="error-message fade-in">
                    <span className="error-icon">⚠️</span>
                    <span>{error}</span>
                  </div>
                )}
              </div>

              {/* Results Display */}
              {isLoading && (
                <div className="loading-container glass-card fade-in">
                  <div className="spinner"></div>
                  <h3>Analyzing MRI Image...</h3>
                  <p>Our AI models are processing your image and performing segmentation</p>
                </div>
              )}

              {singleResult && !isLoading && (
                <PredictionResults result={singleResult} />
              )}

              {batchResult && !isLoading && (
                <ModelComparison batchResult={batchResult} />
              )}
            </div>
          </section>

          {/* About Section */}
          <section id="about" className="about-section section">
            <h2 className="section-title text-center">About Our Models</h2>
            <div className="grid grid-3">
              <div className="feature-card glass-card">
                <div className="feature-icon">🧠</div>
                <h3>5 Classification Models</h3>
                <p>Custom CNN, VGG16, VGG19, MobileNet, and ResNet50 for accurate tumor classification.</p>
              </div>
              <div className="feature-card glass-card">
                <div className="feature-icon">🔬</div>
                <h3>U-Net Segmentation</h3>
                <p>Advanced U-Net model with 98% accuracy for precise tumor region visualization.</p>
              </div>
              <div className="feature-card glass-card">
                <div className="feature-icon">📊</div>
                <h3>Auto-Segmentation</h3>
                <p>Automatic tumor segmentation when a tumor is detected, highlighting affected regions.</p>
              </div>
            </div>
          </section>

          {/* Tumor Types Section */}
          <section id="models" className="tumor-types-section section">
            <h2 className="section-title text-center">Detectable Tumor Types</h2>
            <div className="grid grid-2">
              <div className="tumor-card glass-card">
                <h3>🔴 Glioma Tumor</h3>
                <p>Tumors that originate from glial cells in the brain and spinal cord. Segmentation available.</p>
              </div>
              <div className="tumor-card glass-card">
                <h3>🟠 Meningioma Tumor</h3>
                <p>Tumors that develop in the meninges, the protective layers around the brain. Segmentation available.</p>
              </div>
              <div className="tumor-card glass-card">
                <h3>✅ No Tumor</h3>
                <p>Healthy brain tissue with no signs of tumor presence.</p>
              </div>
              <div className="tumor-card glass-card">
                <h3>🟡 Pituitary Tumor</h3>
                <p>Tumors that form in the pituitary gland at the base of the brain. Segmentation available.</p>
              </div>
            </div>
          </section>
        </div>
      </main>

      {/* Footer */}
      <footer className="footer">
        <div className="container">
          <p>
            ⚠️ <strong>Medical Disclaimer:</strong> This tool is for educational and research purposes only.
            Not intended for medical diagnosis. Always consult qualified healthcare professionals.
          </p>
          <p className="footer-credits">
            Made with ❤️ using TensorFlow, FastAPI, and React | Classification + Segmentation
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
