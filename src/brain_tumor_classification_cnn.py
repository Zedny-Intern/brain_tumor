"""
Brain Tumor Classification - Custom CNN Model
This code is structured as cells for Jupyter Notebook/Google Colab
Each cell is separated by: # ============== CELL X ==============
"""

# ============== CELL 1: IMPORTS AND CONFIGURATION ==============
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
NUM_CLASSES = len(CLASS_NAMES)

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))


# ============== CELL 2: DATASET LOADING ==============
# ⚠️ CHANGE THIS PATH TO YOUR DATASET LOCATION ⚠️
DATASET_PATH = 'path/to/your/dataset'  # <-- CHANGE THIS
TRAIN_DIR = os.path.join(DATASET_PATH, 'Training')
TEST_DIR = os.path.join(DATASET_PATH, 'Testing')

# Verify dataset structure
print(f"Training Directory: {TRAIN_DIR}")
print(f"Testing Directory: {TEST_DIR}")

# Count images in each class
for split, directory in [('Training', TRAIN_DIR), ('Testing', TEST_DIR)]:
    print(f"\n{split} Dataset:")
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(directory, class_name)
        if os.path.exists(class_dir):
            num_images = len(os.listdir(class_dir))
            print(f"  {class_name}: {num_images} images")
        else:
            print(f"  {class_name}: Directory not found!")


# ============== CELL 3: VISUALIZE RANDOM IMAGES FROM TRAINING ==============
def visualize_random_images(data_dir, class_names, num_images=16, title="Random Images"):
    """Visualize random images from the dataset"""
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    fig.suptitle(title, fontsize=20, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            # Randomly select a class
            class_name = np.random.choice(class_names)
            class_dir = os.path.join(data_dir, class_name)
            
            # Randomly select an image from that class
            if os.path.exists(class_dir):
                images = os.listdir(class_dir)
                if images:
                    random_image = np.random.choice(images)
                    img_path = os.path.join(class_dir, random_image)
                    
                    # Load and display image
                    img = plt.imread(img_path)
                    ax.imshow(img)
                    ax.set_title(f"{class_name}", fontsize=12, fontweight='bold')
                    ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Visualize 16 random training images
visualize_random_images(TRAIN_DIR, CLASS_NAMES, num_images=16, 
                        title="16 Random Training Images")


# ============== CELL 4: VISUALIZE RANDOM IMAGES FROM TESTING ==============
# Visualize 16 random testing images
visualize_random_images(TEST_DIR, CLASS_NAMES, num_images=16, 
                        title="16 Random Testing Images")


# ============== CELL 5: DATA PREPROCESSING AND AUGMENTATION ==============
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
    validation_split=0.2  # 20% for validation
)

# Validation and test data generator (only rescaling)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Create training generator
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

# Create validation generator
validation_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True,
    seed=42
)

# Create test generator
test_generator = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")
print(f"Test samples: {test_generator.samples}")
print(f"Class indices: {train_generator.class_indices}")


# ============== CELL 6: CUSTOM METRICS (PRECISION, RECALL, F1) ==============
def precision_metric(y_true, y_pred):
    """Precision metric for multi-class classification"""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall_metric(y_true, y_pred):
    """Recall metric for multi-class classification"""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1_metric(y_true, y_pred):
    """F1 score metric for multi-class classification"""
    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

print("Custom metrics defined: Precision, Recall, F1 Score")


# ============== CELL 7: BUILD CUSTOM CNN MODEL ==============
def build_custom_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    """
    Build a custom CNN model for brain tumor classification
    Updated architecture based on new-brain-tumor (1).ipynb
    - Simplified to 3 convolutional blocks (was 4)
    - Removed BatchNormalization layers
    - Simplified dense layers
    """
    model = keras.Sequential([
        # Input layer
        layers.Input(shape=input_shape),
        
        # First convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and dense layers
        layers.Flatten(),
        layers.Dropout(0.4),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        
        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Build the model
cnn_model = build_custom_cnn()
cnn_model.summary()


# ============== CELL 8: CALLBACKS CONFIGURATION ==============
# Early stopping: stop training if val_loss doesn't improve for 5 epochs
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Model checkpoint: save the best model
model_checkpoint = ModelCheckpoint(
    'best_brain_tumor_cnn_model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Reduce learning rate: reduce LR when val_loss plateaus
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

callbacks = [early_stopping, model_checkpoint, reduce_lr]

print("Callbacks configured: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau")


# ============== CELL 9: COMPILE AND TRAIN THE MODEL ==============
# Compile the model
cnn_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', precision_metric, recall_metric, f1_metric]
)

print("Model compiled with Adam optimizer")
print("Loss: categorical_crossentropy")
print("Metrics: accuracy, precision, recall, f1_score\n")

# Train the model
history = cnn_model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

print("\nTraining completed!")


# ============== CELL 10: VISUALIZE TRAINING HISTORY ==============
def plot_training_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[0, 1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot precision and recall
    axes[1, 0].plot(history.history['precision_metric'], label='Train Precision', linewidth=2)
    axes[1, 0].plot(history.history['val_precision_metric'], label='Val Precision', linewidth=2)
    axes[1, 0].plot(history.history['recall_metric'], label='Train Recall', linewidth=2)
    axes[1, 0].plot(history.history['val_recall_metric'], label='Val Recall', linewidth=2)
    axes[1, 0].set_title('Precision and Recall', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot F1 score
    axes[1, 1].plot(history.history['f1_metric'], label='Train F1 Score', linewidth=2)
    axes[1, 1].plot(history.history['val_f1_metric'], label='Val F1 Score', linewidth=2)
    axes[1, 1].set_title('F1 Score', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('F1 Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

plot_training_history(history)


# ============== CELL 11: EVALUATE ON TEST DATA ==============
# Load the best model
best_model = keras.models.load_model(
    'best_brain_tumor_cnn_model.h5',
    custom_objects={
        'precision_metric': precision_metric,
        'recall_metric': recall_metric,
        'f1_metric': f1_metric
    }
)

# Evaluate on test data
test_loss, test_acc, test_precision, test_recall, test_f1 = best_model.evaluate(test_generator)

print("\n" + "="*50)
print("TEST RESULTS - CUSTOM CNN MODEL")
print("="*50)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
print("="*50)


# ============== CELL 12: CONFUSION MATRIX AND CLASSIFICATION REPORT ==============
# Get predictions
test_generator.reset()
y_pred = best_model.predict(test_generator, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Custom CNN', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.show()

# Classification report
print("\nClassification Report:")
print("="*70)
print(classification_report(y_true, y_pred_classes, target_names=CLASS_NAMES))


# ============== CELL 13: VISUALIZE SAMPLE PREDICTIONS ==============
def visualize_predictions(model, generator, class_names, num_images=16):
    """Visualize model predictions on test images"""
    generator.reset()
    x_batch, y_batch = next(generator)
    predictions = model.predict(x_batch)
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle('Sample Predictions - Custom CNN', fontsize=20, fontweight='bold')
    
    for i, ax in enumerate(axes.flat):
        if i < num_images and i < len(x_batch):
            # Display image
            ax.imshow(x_batch[i])
            
            # Get true and predicted labels
            true_label = class_names[np.argmax(y_batch[i])]
            pred_label = class_names[np.argmax(predictions[i])]
            confidence = np.max(predictions[i]) * 100
            
            # Color: green if correct, red if incorrect
            color = 'green' if true_label == pred_label else 'red'
            
            # Title
            title = f"True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%"
            ax.set_title(title, fontsize=10, fontweight='bold', color=color)
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

visualize_predictions(best_model, test_generator, CLASS_NAMES, num_images=16)

print("\n✅ Custom CNN Model - Complete!")
