"""
Brain Tumor Classification - MobileNet Transfer Learning Model
This code is structured as cells for Jupyter Notebook/Google Colab
Each cell is separated by: # ============== CELL X ==============
"""

# ============== CELL 1: IMPORTS AND CONFIGURATION ==============
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNet
from tensorflow.keras import backend as K

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
NUM_CLASSES = len(CLASS_NAMES)

# Set random seed
np.random.seed(42)
tf.random.set_seed(42)

print("MobileNet Transfer Learning Model")


# ============== CELL 2: DATASET LOADING ==============
# ⚠️ CHANGE THIS PATH TO YOUR DATASET LOCATION ⚠️
DATASET_PATH = 'path/to/your/dataset'  # <-- CHANGE THIS
TRAIN_DIR = os.path.join(DATASET_PATH, 'Training')
TEST_DIR = os.path.join(DATASET_PATH, 'Testing')


# ============== CELL 3: DATA PREPROCESSING AND AUGMENTATION ==============
# Training data generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

# Validation and test data generator
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Create generators
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True,
    seed=42
)

test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)


# ============== CELL 4: CUSTOM METRICS ==============
def precision_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall_metric(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1_metric(y_true, y_pred):
    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# ============== CELL 5: BUILD MOBILENET MODEL ==============
def build_mobilenet_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    """
    Build MobileNet model with custom classification head
    Updated to use Functional API (fixes model loading errors)
    Based on new-brain-tumor (1).ipynb
    """
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Flatten, Dense
    
    # 1. Define explicit Input layer
    input_tensor = Input(shape=input_shape)
    
    # 2. Load pre-trained MobileNet
    base_model = MobileNet(
        weights='imagenet',
        include_top=False,
        input_tensor=input_tensor  # Connect Input here
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # 3. Build complete model using Functional API
    x = base_model.output
    x = Flatten()(x)  # Connect Flatten directly to the tensor 'x'
    x = Dense(256, activation='relu')(x)
    output_tensor = Dense(num_classes, activation='softmax')(x)
    
    # 4. Define the final Model
    model = Model(inputs=base_model.input, outputs=output_tensor)
    
    return model

# Build MobileNet model
mobilenet_model = build_mobilenet_model()
mobilenet_model.summary()


# ============== CELL 6: CALLBACKS CONFIGURATION ==============
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    'best_brain_tumor_mobilenet_model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

callbacks = [early_stopping, model_checkpoint, reduce_lr]


# ============== CELL 7: COMPILE AND TRAIN MOBILENET ==============
mobilenet_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', precision_metric, recall_metric, f1_metric]
)

print("Training MobileNet model...\n")

mobilenet_history = mobilenet_model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

print("\nMobileNet Training completed!")


# ============== CELL 8: VISUALIZE MOBILENET TRAINING HISTORY ==============
def plot_training_history(history, model_name):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{model_name} - Training History', fontsize=16, fontweight='bold')
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision & Recall
    axes[1, 0].plot(history.history['precision_metric'], label='Train Precision', linewidth=2)
    axes[1, 0].plot(history.history['val_precision_metric'], label='Val Precision', linewidth=2)
    axes[1, 0].plot(history.history['recall_metric'], label='Train Recall', linewidth=2)
    axes[1, 0].plot(history.history['val_recall_metric'], label='Val Recall', linewidth=2)
    axes[1, 0].set_title('Precision and Recall', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # F1 Score
    axes[1, 1].plot(history.history['f1_metric'], label='Train F1 Score', linewidth=2)
    axes[1, 1].plot(history.history['val_f1_metric'], label='Val F1 Score', linewidth=2)
    axes[1, 1].set_title('F1 Score', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

plot_training_history(mobilenet_history, 'MobileNet')


# ============== CELL 9: EVALUATE MOBILENET ON TEST DATA ==============
best_mobilenet = keras.models.load_model(
    'best_brain_tumor_mobilenet_model.h5',
    custom_objects={
        'precision_metric': precision_metric,
        'recall_metric': recall_metric,
        'f1_metric': f1_metric
    }
)

test_loss, test_acc, test_precision, test_recall, test_f1 = best_mobilenet.evaluate(test_generator)

print("\n" + "="*50)
print("TEST RESULTS - MOBILENET MODEL")
print("="*50)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
print("="*50)


# ============== CELL 10: MOBILENET CONFUSION MATRIX ==============
test_generator.reset()
y_pred_mobilenet = best_mobilenet.predict(test_generator, verbose=1)
y_pred_classes_mobilenet = np.argmax(y_pred_mobilenet, axis=1)
y_true_mobilenet = test_generator.classes

cm_mobilenet = confusion_matrix(y_true_mobilenet, y_pred_classes_mobilenet)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_mobilenet, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - MobileNet', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.show()

print("\nClassification Report:")
print("="*70)
print(classification_report(y_true_mobilenet, y_pred_classes_mobilenet, target_names=CLASS_NAMES))

print("\n✅ MobileNet Model - Complete!")
