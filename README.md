# Brain Tumor Classification & Segmentation with Deep Learning 🧠

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive deep learning project for classifying brain tumor MRI images into four categories and **automatically segmenting tumor regions** using multiple neural network architectures including U-Net.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Features](#-features)
- [Installation](#-installation)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Quick Start](#-quick-start)
- [Usage Guide](#-usage-guide)
- [Models](#-models)
- [Evaluation Metrics](#-evaluation-metrics)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 📋 Project Overview

This project implements and compares **5 classification models** and **1 segmentation model** for brain tumor analysis:

| Model | Type | Purpose | Accuracy* |
|-------|------|---------|----------|
| **Custom CNN** | Built from scratch | Classification | TBD |
| **VGG16** | Transfer Learning | Classification | TBD |
| **VGG19** | Transfer Learning | Classification | TBD |
| **MobileNet** | Transfer Learning | Classification | TBD |
| **ResNet50** | Transfer Learning | Classification | TBD |
| **U-Net** | Encoder-Decoder | Segmentation | 98% |

*\*Accuracy will vary based on your dataset*

### 🎯 Tumor Categories
- `glioma_tumor` - Glial cell tumors
- `meningioma_tumor` - Meninges tumors
- `no_tumor` - Healthy brain tissue
- `pituitary_tumor` - Pituitary gland tumors

---

## ✨ Features

### 🌐 **Full-Stack Web Application with Segmentation**
- **FastAPI Backend** - RESTful API for model inference and segmentation
- **React Frontend** - Modern, responsive UI with premium design
- **Real-time Predictions** - Upload MRI images and get instant results
- **Automatic Tumor Segmentation** - U-Net based tumor region highlighting
- **Model Comparison** - Compare predictions from all 5 classification models
- **Interactive Visualizations** - Confidence scores, probability charts, and segmentation overlays

### 🔬 Multiple Architectures
- Custom CNN built from scratch
- 4 pre-trained transfer learning models
- Side-by-side performance comparison

### 📊 Comprehensive Evaluation
- Accuracy, Precision, Recall, F1 Score
- Confusion matrix visualization
- Per-class performance metrics
- Training history plots

### 🎨 Data Visualization
- Random sample display (16 training + 16 testing images)
- Prediction visualizations with confidence scores
- Color-coded correct/incorrect predictions

### ⚙️ Advanced Training
- Data augmentation (rotation, shift, zoom, flip)
- Early stopping to prevent overfitting
- Learning rate reduction on plateau
- Automatic best model saving

### 🛠️ Easy Configuration
- Environment variables for all settings
- Cell-based code structure for notebooks
- Modular and reusable components

---

## 🚀 Quick Start - Web Application

### Prerequisites
- Python 3.8+ with pip
- Node.js 16+ and npm
- (Optional) CUDA-enabled GPU

### 1. Backend Setup

```bash
# Install Python dependencies
pip install -r requirements-deployment.txt

# Start FastAPI server
python deployment.py
```

The API will be available at:
- 🌐 http://localhost:8000
- 📚 API Docs: http://localhost:8000/docs
- 📖 Alternative Docs: http://localhost:8000/redoc

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The web app will open at http://localhost:5173

### 3. Use the Application

1. **Upload Image**: Drag and drop or click to select an MRI image
2. **Select Model**: Choose a specific model or "All Models" for comparison
3. **Analyze**: Click "Analyze Image" to get predictions
4. **View Results**: See detailed predictions with confidence scores



### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-enabled GPU for faster training

### Step 1: Clone or Download

Download this project to your local machine.

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- TensorFlow 2.10+ (deep learning framework)
- NumPy (numerical computing)
- Pandas (data manipulation)
- Matplotlib & Seaborn (visualization)
- scikit-learn (metrics and evaluation)
- python-dotenv (environment management)

### Step 4: Verify Installation

```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0)"
```

Expected output:
```
TensorFlow version: 2.x.x
GPU Available: True/False
```

---

## 📁 Project Structure

```
brain_tumor_classification/
├── 📄 README.md                                    # This file - Complete guide
├── 📄 PROJECT_BRIEF.md                             # Project summary
├── 📄 requirements.txt                             # Python dependencies
├── 📄 requirements-deployment.txt                  # Deployment dependencies
├── 📄 LICENSE                                      # MIT License
├── 📄 .gitignore                                   # Git ignore rules
├── 📄 .env.example                                 # Environment template
├── 📄 .env                                         # Your configuration
│
├── 🚀 deployment.py                                # FastAPI backend server
├── 📓 new-brain-tumor (1).ipynb                     # Updated Jupyter notebook with improvements
├── 📓 brain-tumor-classification (1).ipynb         # Original Jupyter notebook
│
├── 📂 src/                                         # Source code
│   ├── 🧠 brain_tumor_classification_cnn.py           # Custom CNN model
│   ├── 🧠 brain_tumor_classification_vgg16.py         # VGG16 transfer learning
│   ├── 🧠 brain_tumor_classification_vgg19.py         # VGG19 transfer learning
│   ├── 🧠 brain_tumor_classification_mobilenet.py     # MobileNet transfer learning
│   ├── 🧠 brain_tumor_classification_resnet.py        # ResNet50 transfer learning
│   ├── 📊 brain_tumor_models_comparison.py            # Performance comparison
│   └── 📂 models/                                      # Trained model files (.h5)
│       ├── best_brain_tumor_cnn_model.h5
│       ├── best_brain_tumor_vgg16_model.h5
│       ├── best_brain_tumor_vgg19_model.h5
│       ├── best_brain_tumor_mobilenet_model.h5
│       ├── best_brain_tumor_resnet_model.h5
│       └── best_brain_tumor_unet_model.h5              # U-Net segmentation model (98% acc)
│
├── 📂 results/                                     # Visualization results
│   ├── CNN/                                        # CNN model results
│   ├── VGG_16/                                     # VGG16 model results
│   ├── VGG_19/                                     # VGG19 model results
│   ├── MobileNet/                                  # MobileNet model results
│   ├── ResNet/                                     # ResNet model results
│   └── comparison/                                 # Model comparison charts
│
├── 📂 frontend/                                    # React Web Application
│   ├── 📂 src/
│   │   ├── 📄 App.tsx                             # Main application
│   │   ├── 📄 api.ts                              # API service layer
│   │   ├── 📄 index.css                           # Global styles
│   │   └── 📂 components/
│   │       ├── Header.tsx                         # Navigation header
│   │       ├── ImageUploader.tsx                  # Image upload component
│   │       ├── PredictionResults.tsx              # Single model results
│   │       ├── ModelComparison.tsx                # Multi-model comparison
│   │       └── SegmentationResult.tsx             # Tumor segmentation display
│   ├── 📄 package.json                            # Frontend dependencies
│   ├── 📄 .env.example                            # Frontend env template
│   └── 📄 vite.config.ts                          # Vite configuration
│
└── 📂 dataset/                                     # Your dataset (not included)
    ├── Training/
    │   ├── glioma_tumor/
    │   ├── meningioma_tumor/
    │   ├── no_tumor/
    │   └── pituitary_tumor/
    └── Testing/
        ├── glioma_tumor/
        ├── meningioma_tumor/
        ├── no_tumor/
        └── pituitary_tumor/
```


---

## ⚙️ Configuration

### Environment Variables

1. **Copy the example environment file:**
   ```bash
   # Windows
   copy .env.example .env
   
   # macOS/Linux
   cp .env.example .env
   ```

2. **Edit `.env` and update your dataset path:**
   ```bash
   DATASET_PATH=C:\Users\YourName\Documents\brain_tumor_dataset
   ```

### Available Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET_PATH` | `path/to/your/dataset` | **⚠️ REQUIRED** - Path to your dataset |
| `IMG_SIZE` | `224` | Input image size (224x224) |
| `BATCH_SIZE` | `32` | Batch size (reduce if out of memory) |
| `EPOCHS` | `20` | Maximum training epochs |
| `LEARNING_RATE` | `0.001` | Initial learning rate |
| `VALIDATION_SPLIT` | `0.2` | % of training data for validation |
| `EARLY_STOPPING_PATIENCE` | `5` | Epochs to wait before stopping |
| `REDUCE_LR_PATIENCE` | `3` | Epochs to wait before reducing LR |
| `RANDOM_SEED` | `42` | Random seed for reproducibility |

See `.env.example` for all available options.

---

## 🎯 Quick Start

### 1. Prepare Your Dataset

Organize your brain tumor MRI images in this structure:

```
your_dataset/
├── Training/
│   ├── glioma_tumor/       ← Place glioma training images here
│   ├── meningioma_tumor/   ← Place meningioma training images here
│   ├── no_tumor/           ← Place no tumor training images here
│   └── pituitary_tumor/    ← Place pituitary training images here
└── Testing/
    ├── glioma_tumor/       ← Place glioma test images here
    ├── meningioma_tumor/   ← Place meningioma test images here
    ├── no_tumor/           ← Place no tumor test images here
    └── pituitary_tumor/    ← Place pituitary test images here
```

### 2. Update Dataset Path

**Option A: Using `.env` file (Recommended)**
```bash
# Edit .env file
DATASET_PATH=C:\path\to\your_dataset
```

**Option B: Direct in code**
```python
# In Cell 2 of any model file
DATASET_PATH = r'C:\path\to\your_dataset'
```

### 3. Choose Your Environment

**🎓 For Learning (Jupyter Notebook/Colab):**
1. Open Jupyter Notebook or Google Colab
2. Copy code from any model file
3. Split by cell markers (`# ============== CELL X ==============`)
4. Paste into separate cells
5. Run sequentially

**💻 For Quick Testing (Python Script):**
```bash
python brain_tumor_classification_cnn.py
```

### 4. Start with One Model

**Recommended**: Start with **MobileNet** (fastest training):
```bash
python brain_tumor_classification_mobilenet.py
```

Or in Jupyter/Colab, copy cells from `brain_tumor_classification_mobilenet.py`

### 5. Compare All Models

After training all 5 models:
1. Open `brain_tumor_models_comparison.py`
2. Update Cell 2 with your test results
3. Run to see comparison charts

---

## 📖 Usage Guide

### Training a Model

Each model file follows the same structure (10-13 cells):

```python
# Cell 1: Imports and Configuration
# Cell 2: Dataset Loading ⚠️ UPDATE DATASET_PATH HERE
# Cell 3: Data Preprocessing & Augmentation
# Cell 4: Custom Metrics (Precision, Recall, F1)
# Cell 5: Build Model
# Cell 6: Callbacks Configuration
# Cell 7: Compile and Train
# Cell 8: Visualize Training History
# Cell 9: Evaluate on Test Data
# Cell 10: Confusion Matrix & Classification Report
```

### Running in Jupyter Notebook

```python
# Cell 1
import tensorflow as tf
# ... rest of imports ...

# Cell 2
DATASET_PATH = r'C:\your\dataset\path'  # ⚠️ UPDATE THIS
TRAIN_DIR = os.path.join(DATASET_PATH, 'Training')
TEST_DIR = os.path.join(DATASET_PATH, 'Testing')

# ... continue with remaining cells ...
```

### Running as Python Script

```bash
# 1. Update DATASET_PATH in the .py file
# 2. Run the script
python brain_tumor_classification_cnn.py

# Or specify which model
python brain_tumor_classification_vgg16.py
python brain_tumor_classification_vgg19.py
python brain_tumor_classification_mobilenet.py
python brain_tumor_classification_resnet.py
```

### Training Time Estimates

| Model | CPU (approx.) | GPU (approx.) |
|-------|---------------|---------------|
| Custom CNN | 30-40 min | 10-15 min |
| VGG16 | 25-35 min | 8-12 min |
| VGG19 | 25-35 min | 8-12 min |
| MobileNet | 20-30 min | 6-10 min |
| ResNet50 | 25-35 min | 8-12 min |

*Times vary based on dataset size and hardware*

---

## 🤖 Models

### 1. Custom CNN
**Architecture**: 3 convolutional blocks (Updated from original 4 blocks)
```
Conv2D(64) → MaxPool
Conv2D(64) → MaxPool
Conv2D(128) → MaxPool
Flatten → Dropout(0.4) → Dense(128) → Dense(64) → Dense(4)
```
**Updates**: Simplified from 4 convolutional blocks, removed BatchNormalization for efficiency

### 2. VGG16
**Architecture**: Pre-trained VGG16 + Custom head (Functional API)
- Base: VGG16 (frozen, ImageNet weights)
- Head: Flatten → Dense(256) → Dense(4)
- **Updated**: Now uses Keras Functional API for proper model saving/loading

### 3. VGG19
**Architecture**: Pre-trained VGG19 + Custom head (Functional API)
- Base: VGG19 (frozen, ImageNet weights)
- Head: Flatten → Dense(256) → Dense(4)
- **Updated**: Now uses Keras Functional API for proper model saving/loading

### 4. MobileNet
**Architecture**: Pre-trained MobileNet + Custom head (Functional API)
- Base: MobileNet (frozen, ImageNet weights)
- Head: Flatten → Dense(256) → Dense(4)
- **Advantage**: Lightweight, fast training
- **Updated**: Now uses Keras Functional API for proper model saving/loading

### 5. ResNet50
**Architecture**: Pre-trained ResNet50 + Custom head (Functional API)
- Base: ResNet50 (frozen, ImageNet weights)
- Head: Flatten → Dense(256) → Dense(4)
- **Advantage**: Deep network with residual connections
- **Updated**: Now uses Keras Functional API for proper model saving/loading

> **Note**: Model architectures have been updated based on `new-brain-tumor (1).ipynb`. The Functional API approach fixes previous model loading errors and provides better compatibility.

---

## 📊 Evaluation Metrics

All models are evaluated using:

### Classification Metrics
- **Accuracy**: Overall correctness
- **Precision**: Of predicted tumors, how many are correct
- **Recall**: Of actual tumors, how many are detected
- **F1 Score**: Harmonic mean of precision and recall

### Visualizations
- **Training Curves**: Accuracy and loss over epochs
- **Confusion Matrix**: True vs predicted labels heatmap
- **Classification Report**: Per-class precision, recall, F1
- **Sample Predictions**: Visual display with confidence scores

### Model Outputs

After training, each model generates:
```
✅ Saved model file (.h5)
✅ Training history plots
✅ Test metrics (accuracy, precision, recall, F1)
✅ Confusion matrix
✅ Classification report
✅ Sample predictions visualization
```

---

## 🔧 Hyperparameters

### Image Processing
- **IMG_SIZE**: 224×224 pixels
- **Rescaling**: 1/255 (normalize to [0,1])

### Data Augmentation
- **Rotation**: ±20°
- **Width Shift**: 20%
- **Height Shift**: 20%
- **Shear**: 20%
- **Zoom**: 20%
- **Horizontal Flip**: Yes

### Training
- **Batch Size**: 32
- **Epochs**: 20 (with early stopping)
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Loss Function**: Categorical Crossentropy

### Callbacks
- **EarlyStopping**: 
  - Monitor: val_loss
  - Patience: 5 epochs
- **ModelCheckpoint**: 
  - Save best model based on val_loss
- **ReduceLROnPlateau**: 
  - Factor: 0.5
  - Patience: 3 epochs

---

## 🔍 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **Dataset not found** | Update `DATASET_PATH` in `.env` or Cell 2 |
| **Out of memory** | Reduce `BATCH_SIZE` to 16 or 8 |
| **Training too slow** | Use GPU or reduce `EPOCHS` |
| **Poor accuracy** | Check data quality, increase epochs |
| **Model not improving** | Adjust learning rate or augmentation |
| **Import errors** | Run `pip install -r requirements.txt` |
| **GPU not detected** | Install `tensorflow-gpu` or check CUDA install |

### Performance Tips

1. **Use GPU**: 3-5x faster than CPU
   ```bash
   pip install tensorflow-gpu
   ```

2. **Adjust Batch Size**: Based on your GPU memory
   - 16GB GPU: `BATCH_SIZE=64`
   - 8GB GPU: `BATCH_SIZE=32`
   - 4GB GPU: `BATCH_SIZE=16`
   - CPU: `BATCH_SIZE=8`

3. **Start Small**: Test with MobileNet before larger models

4. **Monitor Training**: Watch validation loss to detect overfitting

5. **Save Results**: Record metrics for comparison

---

## 📚 Requirements

See `requirements.txt` for exact versions:

```
tensorflow>=2.10.0,<2.16.0
numpy>=1.21.0,<1.27.0
pandas>=1.3.0,<2.2.0
matplotlib>=3.5.0,<3.9.0
seaborn>=0.12.0,<0.14.0
scikit-learn>=1.0.0,<1.4.0
pillow>=9.0.0,<11.0.0
python-dotenv>=0.19.0,<1.1.0
```

### Optional Dependencies

For Jupyter Notebook:
```bash
pip install jupyter notebook ipykernel
```

For deployment:
```bash
pip install flask streamlit
```


---

## 🔌 API Documentation

The FastAPI backend provides the following endpoints:

### Health & Info Endpoints

#### GET `/health`
Check API health and loaded models status.

```json
{
  "status": "healthy",
  "models_loaded": ["cnn", "vgg16", "vgg19", "mobilenet", "resnet"],
  "models_failed": []
}
```

#### GET `/models`
Get detailed information about all models.

#### GET `/classes`
Get information about tumor class categories.

### Prediction Endpoints

#### POST `/predict/{model_name}`
Predict using a specific model (cnn, vgg16, vgg19, mobilenet, resnet).

**Request:**
- Form Data: `file` (image file)

**Response:**
```json
{
  "success": true,
  "model_name": "vgg16",
  "predicted_class": "glioma_tumor",
  "confidence": 0.95,
  "all_probabilities": {
    "glioma_tumor": 0.95,
    "meningioma_tumor": 0.03,
    "no_tumor": 0.01,
    "pituitary_tumor": 0.01
  },
  "processing_time_ms": 245
}
```

#### POST `/predict/all`
Get predictions from all available models with consensus.

**Request:**
- Form Data: `file` (image file)

**Response:**
```json
{
  "success": true,
  "predictions": {
    "cnn": { /* prediction result */ },
    "vgg16": { /* prediction result */ },
    "vgg19": { /* prediction result */ },
    "mobilenet": { /* prediction result */ },
    "resnet": { /* prediction result */ }
  },
  "consensus_prediction": "glioma_tumor",
  "average_confidence": 0.92
}
```

### Interactive API Documentation

Visit these URLs when the server is running:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

---

## 🤝 Contributing


Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Ideas for Contribution
- Add more pre-trained models (EfficientNet, DenseNet)
- Implement data preprocessing improvements
- Add web interface for predictions
- Improve documentation
- Add unit tests

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Medical Disclaimer

⚠️ **IMPORTANT**: This software is for **educational and research purposes only**.

- NOT intended for medical diagnosis
- NOT a substitute for professional medical advice
- Always consult qualified healthcare professionals
- Authors assume NO liability for medical decisions

---

## 🎓 Learning Resources

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras Applications](https://keras.io/api/applications/)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/)

---

## 📞 Support

### Quick Links
- 📖 [Project Brief](PROJECT_BRIEF.md) - Overview and specifications
- 🔧 [Environment Template](.env.example) - Configuration options
- 📋 [Requirements](requirements.txt) - Dependencies

### Getting Help
1. Check [Troubleshooting](#-troubleshooting) section
2. Review code comments in model files
3. Verify dataset structure matches requirements
4. Ensure all dependencies are installed

---

## 🎯 Project Status

- ✅ Custom CNN implementation
- ✅ VGG16 transfer learning
- ✅ VGG19 transfer learning
- ✅ MobileNet transfer learning
- ✅ ResNet50 transfer learning
- ✅ Model comparison tool
- ✅ Comprehensive documentation
- ✅ Environment configuration
- ✅ Installation automation
- ✅ **FastAPI Backend Deployment**
- ✅ **React Frontend Web Application**
- ✅ **RESTful API with Interactive Docs**
- ✅ **Real-time Model Inference**
- ✅ **Multi-Model Comparison Interface**

---

## 🌟 Acknowledgments

- TensorFlow/Keras team for the deep learning framework
- Pre-trained model weights from ImageNet
- Medical imaging research community

---

**Made with ❤️ for Educational Purposes**

**Happy Training & Predicting! 🚀**

*Last Updated: December 2025*
