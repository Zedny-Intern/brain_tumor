# Environment Setup Guide

## Problem
Your current environment uses Python 3.13, which is very new and some packages don't have pre-built binaries yet, causing compilation errors.

## Solution: Create Dedicated Conda Environment

### Option 1: Automated Setup (Recommended)

Simply run the setup script:
```bash
setup_environment.bat
```

This will:
1. Create a new conda environment named `brain_tumor_env` with Python 3.11
2. Install all required packages via conda (faster, pre-built)
3. Install TensorFlow and FastAPI dependencies

### Option 2: Manual Setup

If you prefer to do it manually:

```bash
# Create environment with Python 3.11
conda create -n brain_tumor_env python=3.11 -y

# Activate it
conda activate brain_tumor_env

# Install packages via conda (recommended for these)
conda install -c conda-forge numpy pandas pillow matplotlib seaborn scikit-learn -y

# Install deep learning and web frameworks via pip
pip install tensorflow fastapi uvicorn[standard] pydantic python-multipart
```

## Using the Environment

### Every time you work on this project:

**Terminal 1 - Backend:**
```bash
conda activate brain_tumor_env
python deployment.py
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

## Verifying Installation

After setup, test with:
```bash
conda activate brain_tumor_env
python test_backend.py
```

## Why Python 3.11?

- ✅ Excellent compatibility with TensorFlow 2.x
- ✅ All packages have pre-built wheels (no compilation needed)
- ✅ Stable and well-tested
- ✅ Recommended for machine learning projects

## Deactivating Environment

When you're done:
```bash
conda deactivate
```

## Removing Environment (if needed)

If you ever want to start fresh:
```bash
conda deactivate
conda env remove -n brain_tumor_env
```

Then re-run the setup script.

---

**Note**: Your base conda environment will remain unchanged. This creates a separate, isolated environment just for this project.
