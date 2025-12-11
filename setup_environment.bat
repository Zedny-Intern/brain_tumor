@echo off
REM Brain Tumor Classification - Environment Setup Script
REM This script creates a conda environment with all required dependencies

echo ========================================
echo Brain Tumor Classification Setup
echo ========================================
echo.

echo Creating conda environment 'brain_tumor_env' with Python 3.11...
call conda create -n brain_tumor_env python=3.11 -y

echo.
echo Activating environment...
call conda activate brain_tumor_env

echo.
echo Installing core packages via conda (faster, pre-built)...
call conda install -c conda-forge numpy pandas pillow matplotlib seaborn scikit-learn -y

echo.
echo Installing TensorFlow...
call pip install tensorflow

echo.
echo Installing FastAPI and deployment dependencies...
call pip install fastapi uvicorn[standard] pydantic python-multipart

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To use this environment:
echo   1. conda activate brain_tumor_env
echo   2. python deployment.py
echo.
echo In a new terminal for frontend:
echo   1. cd frontend
echo   2. npm run dev
echo.
pause
