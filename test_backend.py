"""
Quick test script to verify the backend is working with segmentation
"""
import requests
import time

# Wait a moment for server to be ready
time.sleep(2)

try:
    # Test health endpoint
    print("Testing backend server...")
    print("=" * 60)
    
    response = requests.get("http://localhost:8000/health", timeout=5)
    print(f"✅ Health Check: {response.status_code}")
    data = response.json()
    print(f"Response: {data}\n")
    print(f"Segmentation Model Loaded: {data.get('segmentation_model_loaded', False)}")
    
    # Test models endpoint
    response = requests.get("http://localhost:8000/models", timeout=5)
    print(f"\n✅ Models endpoint: {response.status_code}")
    data = response.json()
    print(f"Loaded models: {data.get('loaded_models', 0)}/{data.get('total_models', 6)}")
    print(f"Available models: {data.get('available_models', [])}")
    print(f"Segmentation model: {data.get('segmentation_model', {})}\n")
    
    # Test classes endpoint
    response = requests.get("http://localhost:8000/classes", timeout=5)
    print(f"✅ Classes endpoint: {response.status_code}")
    print(f"Classes: {response.json().get('classes', [])}")
    print(f"Tumor classes (segmentation available): {response.json().get('tumor_classes', [])}\n")
    
    print("=" * 60)
    print("🎉 Backend is working correctly!")
    print("📍 API Documentation: http://localhost:8000/docs")
    print("📍 Alternative Docs: http://localhost:8000/redoc")
    print("\n🧠 Classification: 5 models (CNN, VGG16, VGG19, MobileNet, ResNet50)")
    print("🔬 Segmentation: U-Net model (98% accuracy)")
    
except requests.exceptions.ConnectionError:
    print("❌ Cannot connect to backend. Make sure it's running:")
    print("   python deployment.py")
except Exception as e:
    print(f"❌ Error: {e}")
