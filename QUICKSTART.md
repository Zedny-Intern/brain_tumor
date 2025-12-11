# Brain Tumor Classification & Segmentation - Quick Start Guide

This guide helps you get up and running quickly with the brain tumor classification and segmentation web application.

---

## Prerequisites

- ✅ Python 3.8+ installed
- ✅ Node.js 16+ and npm installed
- ✅ All 5 classification models + 1 segmentation model in `src/models/` directory

---

## Setup Steps

### 1. Install Backend Dependencies

```bash
# From the project root directory
pip install -r requirements-deployment.txt
```

### 2. Install Frontend Dependencies  

```bash
# Navigate to frontend directory
cd frontend

# Install npm packages
npm install

# **IMPORTANT**: Create .env file for API connection
Copy-Item .env.example -Destination .env

# **VERIFY**: Check the .env file content is correct
Get-Content .env
# Should show: VITE_API_URL=http://localhost:8000

# Return to root
cd ..
```

> **⚠️ Critical**: The `.env` file is **required** for the frontend to connect to the backend API. If you skip this step, you'll see a blank page.

> **✅ Verification**: After creating the `.env` file, verify it contains **exactly**: `VITE_API_URL=http://localhost:8000`

---

### 3. Start the Backend Server

```bash
# From project root
python deployment.py
```

**Expected Output**:
```
======================================================================
Brain Tumor Classification API
======================================================================

Starting server...

📍 API will be available at: http://localhost:8000
📍 API documentation at: http://localhost:8000/docs
📍 Alternative docs at: http://localhost:8000/redoc

======================================================================
INFO: Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
INFO: Loading cnn model from C:\...\best_brain_tumor_cnn_model.h5
INFO: ✓ CNN loaded successfully
INFO: ✓ VGG16 loaded successfully
INFO: ✓ VGG19 loaded successfully
INFO: ✓ MOBILENET loaded successfully
INFO: ✓ RESNET loaded successfully
INFO: ✓ U-Net segmentation model loaded successfully
INFO: Loaded 5/5 classification models, segmentation model: loaded
INFO: API ready to serve predictions and segmentation!
```

> **✅ Success Indicator**: All 5 models should show "loaded successfully"

---

### 4. Start the Frontend (New Terminal)

**Open a NEW terminal window** and run:

```bash
# Navigate to frontend  
cd frontend

# Start dev server
npm run dev
```

**Expected Output**:
```
VITE v6.4.1  ready in 1069 ms

➜  Local:   http://localhost:5173/
➜  Network: use --host to expose
➜  press h + enter to show help
```

> **✅ Success Indicator**: Should see "ready" message with no errors

**Frontend will be running at:** http://localhost:5173

---

### 5. Use the Application

1. Open **http://localhost:5173** in your browser
2. You should see the **full UI** (hero section, upload area, model selection)
3. Upload an MRI image (drag & drop or click to browse)
4. Select a model or choose "All Models" for comparison
5. Click "Analyze Image"
6. View results with confidence scores and predictions
7. **If a tumor is detected**, view the automatic segmentation showing the tumor region

---

## Troubleshooting

### ❌ Blank Page / Empty Screen

**Problem**: Opening http://localhost:5173 shows a blank/white page

**Causes**:
1. Missing `.env` file in frontend directory
2. Corrupted or incorrect `.env` file content

**Solution**:

**Step 1 - Check if .env file exists and is correct**:
```bash
cd frontend
Get-Content .env
```

**Expected output**:
```
VITE_API_URL=http://localhost:8000
```

**If the file is missing or has wrong content**:
```bash
# Remove corrupted file (if exists)
Remove-Item .env -ErrorAction SilentlyContinue

# Create fresh .env file
"VITE_API_URL=http://localhost:8000" | Out-File -FilePath .env -Encoding ASCII -NoNewline

# Verify it's correct
Get-Content .env

# Restart frontend
# Press Ctrl+C in the npm run dev terminal, then:
npm run dev
```

**Step 2 - Hard refresh browser**:
- Press `Ctrl + Shift + R` (Windows)
- Or `Cmd + Shift + R` (Mac)

---

### ❌ CSS @import Error

**Problem**: Frontend shows error: `@import must precede all other statements`

**Cause**: Google Fonts import was in wrong position in CSS file

**Solution**: Already fixed! The import is now at the top of `index.css`. If you still see this:
1. Stop frontend (Ctrl+C)
2. Run `npm run dev` again
3. Hard refresh browser (Ctrl+Shift+R or Cmd+Shift+R)

---

### ❌ Can't Connect to API

**Problem**: Frontend loads but can't make predictions

**Symptoms**: 
- Network errors in browser console
- "Failed to fetch" errors
- CORS errors

**Solutions**:

1. **Ensure backend is running**:
   - Check that `python deployment.py` terminal is still active
   - Should show "API ready to serve predictions!"

2. **Verify .env file**:
   ```bash
   cd frontend
   type .env  # Windows
   cat .env   # Mac/Linux
   ```
   Should contain: `VITE_API_URL=http://localhost:8000`

3. **Check CORS settings** in `deployment.py` (already configured correctly)

4. **Restart both servers**:
   - Stop both terminals (Ctrl+C)
   - Start backend first, then frontend

---

### ❌ Models Not Loading

**Problem**: Backend shows "Model file not found" or loading errors

**Solution**:
- Ensure all .h5 files exist in `src/models/` directory:
  - `best_brain_tumor_cnn_model.h5`
  - `best_brain_tumor_vgg16_model.h5`
  - `best_brain_tumor_vgg19_model.h5`
  - `best_brain_tumor_mobilenet_model.h5`
  - `best_brain_tumor_resnet_model.h5`
  - `best_brain_tumor_unet_model.h5` (segmentation model)

- If models are missing, you need to train them first using the scripts in `src/`

---

### ❌ Port 8000 Already in Use

**Problem**: `Address already in use` error when starting backend

**Solution**:
```bash
# Windows: Find and kill process on port 8000
netstat -ano | findstr :8000
taskkill /PID <PID_NUMBER> /F

# Mac/Linux:
lsof -ti:8000 | xargs kill -9
```

Or change the port in `deployment.py` (line 437):
```python
port=8001,  # Change from 8000 to 8001
```

---

### ❌ npm install Fails

**Problem**: Frontend dependencies won't install

**Solutions**:
1. Update Node.js to version 16 or higher
2. Try: `npm install --legacy-peer-deps`
3. Delete `node_modules` and `package-lock.json`, then run `npm install` again

---

### Model Architecture Updates

This project has been updated with improved model architectures from `new-brain-tumor (1).ipynb`:

- **CNN**: Simplified from 4 to 3 convolutional blocks for better efficiency
- **Transfer Learning Models**: Now use Keras Functional API (fixes model loading issues)
- **Classification Head**: Simplified dense layers for faster training

**⚠️ Important**: If you have existing `.h5` model files from older versions, they may be incompatible with the updated code. You'll need to retrain models using the updated scripts in `src/`.

---

## Production Deployment

### Backend
```bash
# Install production server
pip install gunicorn

# Run with gunicorn (4 workers)
gunicorn deployment:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000
```

### Frontend
```bash
cd frontend

# Build for production
npm run build

# Serve the built files
npm run preview
```

The production build will be in `frontend/dist/` directory.

---

## Quick Verification Checklist

Before using the application, verify:

- [ ] Backend terminal shows "Loaded 5/5 models successfully"
- [ ] Frontend terminal shows "ready" with no errors
- [ ] `.env` file exists in `frontend/` directory
- [ ] Opening http://localhost:5173 shows **full UI** (not blank)
- [ ] Browser console has no errors (press F12 to check)
- [ ] Can see upload area, model selection, and buttons

If all checkboxes are ticked, you're ready to classify brain tumors! 🧠🚀

---

## Need Help?

- 📖 Read the full [README.md](README.md) for detailed documentation
- 📚 Check [API Documentation](http://localhost:8000/docs) (when server running)
- 🔍 Review code comments in source files
- 📊 See model architectures in [README.md](README.md#-models)

---

**Happy Classifying! 🧠🚀**

*Last Updated: December 3, 2025*
