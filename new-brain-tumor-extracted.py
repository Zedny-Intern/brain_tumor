#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# In[2]:


# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
NUM_CLASSES = len(CLASS_NAMES)

# In[3]:


# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# In[4]:


print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# In[5]:


train_path = '/kaggle/input/brain-tumor-mri-dataset/Training'
test_path = '/kaggle/input/brain-tumor-mri-dataset/Testing'

# In[6]:


train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # 20% for validation
)

# In[7]:


val_test_datagen = ImageDataGenerator(rescale=1./255)

# In[8]:


train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

# In[9]:


validation_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True,
    seed=42
)

# In[10]:


test_generator = val_test_datagen.flow_from_directory(
    test_path,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# In[11]:


print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")
print(f"Test samples: {test_generator.samples}")
print(f"Class indices: {train_generator.class_indices}")

# In[12]:


def precision_metric(y_true, y_pred):
    """Precision metric for multi-class classification"""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# In[13]:


def recall_metric(y_true, y_pred):
    """Recall metric for multi-class classification"""
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

# In[14]:


def f1_metric(y_true, y_pred):
    """F1 score metric for multi-class classification"""
    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

# In[15]:


def build_custom_cnn(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
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


# In[16]:


early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# In[17]:


model_checkpoint = ModelCheckpoint(
    'best_brain_tumor_cnn_model.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# In[18]:


reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

callbacks = [early_stopping, model_checkpoint, reduce_lr]

# In[19]:


cnn_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', precision_metric, recall_metric, f1_metric]
)

print("Model compiled with Adam optimizer")
print("Loss: categorical_crossentropy")
print("Metrics: accuracy, precision, recall, f1_score\n")

# In[20]:


history = cnn_model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

print("\nTraining completed!")

# In[21]:


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

# In[22]:


best_model = keras.models.load_model(
    'best_brain_tumor_cnn_model.h5',
    custom_objects={
        'precision_metric': precision_metric,
        'recall_metric': recall_metric,
        'f1_metric': f1_metric
    }
)

# In[23]:


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

# In[24]:


test_generator.reset()
y_pred = best_model.predict(test_generator, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = test_generator.classes

# In[25]:


cm = confusion_matrix(y_true, y_pred_classes)

# In[26]:


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

# In[27]:


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


# In[28]:


# from tensorflow.keras.applications import MobileNet
# def build_mobilenet_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
#     """Build MobileNet model with custom classification head"""
#     # Load pre-trained MobileNet without top layers
#     base_model = MobileNet(
#         weights='imagenet',
#         include_top=False,
#         input_shape=input_shape
#     )
    
#     # Freeze base model layers
#     base_model.trainable = False
    
#     # Build complete model
#     model = keras.Sequential([
#         base_model,
#         # layers.GlobalAveragePooling2D(),
#         layers.Flatten(),
#         layers.Dense(256, activation='relu'),
#         layers.Dense(num_classes, activation='softmax')
#     ])
    
#     return model
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model # <--- Added Import
from tensorflow.keras.layers import Input, Flatten, Dense # <--- Added Imports

def build_mobilenet_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    """Build MobileNet model with custom classification head"""
    
    # 1. Define explicit Input layer
    input_tensor = Input(shape=input_shape) 
    
    # 2. Load pre-trained MobileNet
    base_model = MobileNet(
        weights='imagenet',
        include_top=False,
        input_tensor=input_tensor # <--- Connect Input here
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # 3. Build complete model using Functional API
    x = base_model.output # Get output tensor
    x = Flatten()(x)     # Connect Flatten directly to the tensor 'x'
    x = Dense(256, activation='relu')(x)
    output_tensor = Dense(num_classes, activation='softmax')(x)
    
    # 4. Define the final Model
    model = Model(inputs=base_model.input, outputs=output_tensor) 
    
    return model
# Build MobileNet model
mobilenet_model = build_mobilenet_model()
mobilenet_model.summary()

# In[29]:


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

# In[30]:


mobilenet_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', precision_metric, recall_metric, f1_metric]
)

# In[31]:


print("Training MobileNet model...\n")

mobilenet_history = mobilenet_model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

print("\nMobileNet Training completed!")

# In[32]:


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

# In[33]:


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


# In[34]:


# from tensorflow.keras.applications import ResNet50
# def build_resnet_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
#     """Build ResNet50 model with custom classification head"""
#     # Load pre-trained ResNet50 without top layers
#     base_model = ResNet50(
#         weights='imagenet',
#         include_top=False,
#         input_shape=input_shape
#     )
    
#     # Freeze base model layers
#     base_model.trainable = False
    
#     # Build complete model
#     model = keras.Sequential([
#         base_model,
#         # layers.GlobalAveragePooling2D(),
#         layers.Flatten(),
#         layers.Dense(256, activation='relu'),
#         layers.Dense(num_classes, activation='softmax')
#     ])
    
#     return model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense

def build_resnet_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    """Build ResNet50 model with custom classification head"""
    
    # 1. Define explicit Input layer
    input_tensor = Input(shape=input_shape)
    
    # 2. Load pre-trained ResNet50
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_tensor=input_tensor # <--- Connect Input here
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # 3. Build complete model using Functional API
    x = base_model.output
    x = Flatten()(x)     # Connect Flatten directly to the tensor 'x'
    x = Dense(256, activation='relu')(x)
    output_tensor = Dense(num_classes, activation='softmax')(x)
    
    # 4. Define the final Model
    model = Model(inputs=base_model.input, outputs=output_tensor) 
    
    return model
# Build ResNet50 model
resnet_model = build_resnet_model()
resnet_model.summary()

# In[35]:


early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    'best_brain_tumor_resnet_model.h5',
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

# In[36]:


resnet_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', precision_metric, recall_metric, f1_metric]
)

# In[37]:


print("Training ResNet50 model...\n")

resnet_history = resnet_model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

print("\nResNet50 Training completed!")

# In[38]:


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

plot_training_history(resnet_history, 'ResNet50')

# In[39]:


best_resnet = keras.models.load_model(
    'best_brain_tumor_resnet_model.h5',
    custom_objects={
        'precision_metric': precision_metric,
        'recall_metric': recall_metric,
        'f1_metric': f1_metric
    }
)

test_loss, test_acc, test_precision, test_recall, test_f1 = best_resnet.evaluate(test_generator)

print("\n" + "="*50)
print("TEST RESULTS - RESNET50 MODEL")
print("="*50)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
print("="*50)

# In[40]:


# ============== CELL 10: RESNET50 CONFUSION MATRIX ==============
test_generator.reset()
y_pred_resnet = best_resnet.predict(test_generator, verbose=1)
y_pred_classes_resnet = np.argmax(y_pred_resnet, axis=1)
y_true_resnet = test_generator.classes

cm_resnet = confusion_matrix(y_true_resnet, y_pred_classes_resnet)

# In[41]:


plt.figure(figsize=(10, 8))
sns.heatmap(cm_resnet, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - ResNet50', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.show()

print("\nClassification Report:")
print("="*70)
print(classification_report(y_true_resnet, y_pred_classes_resnet, target_names=CLASS_NAMES))
print("\n✅ ResNet50 Model - Complete!")

# In[42]:


# from tensorflow.keras.applications import VGG16
# def build_vgg16_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
#     """Build VGG16 model with custom classification head"""
#     # Load pre-trained VGG16 without top layers
#     base_model = VGG16(
#         weights='imagenet',
#         include_top=False,
#         input_shape=input_shape
#     )
    
#     # Freeze base model layers
#     base_model.trainable = False
    
#     # Build complete model
#     model = keras.Sequential([
#         base_model,
#         keras.layers.Flatten(),
#         layers.Dense(256, activation='relu'),
#         layers.Dense(num_classes, activation='softmax')
#     ])
    
#     return model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense

def build_vgg16_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    """Build VGG16 model with custom classification head"""
    
    # 1. Define explicit Input layer
    input_tensor = Input(shape=input_shape)
    
    # 2. Load pre-trained VGG16
    base_model = VGG16(
        weights='imagenet',
        include_top=False,
        input_tensor=input_tensor # <--- Connect Input here
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # 3. Build complete model using Functional API
    x = base_model.output
    x = Flatten()(x)     # Connect Flatten directly to the tensor 'x'
    x = Dense(256, activation='relu')(x)
    output_tensor = Dense(num_classes, activation='softmax')(x)
    
    # 4. Define the final Model
    model = Model(inputs=base_model.input, outputs=output_tensor) 
    
    return model
# Build VGG16 model
vgg16_model = build_vgg16_model()
vgg16_model.summary()

# In[43]:


early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    'best_brain_tumor_vgg16_model.h5',
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

# In[44]:


vgg16_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', precision_metric, recall_metric, f1_metric]
)

# In[45]:


print("Training VGG16 model...\n")

vgg16_history = vgg16_model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

print("\nVGG16 Training completed!")

# In[46]:


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

plot_training_history(vgg16_history, 'VGG16')

# In[47]:


best_vgg16 = keras.models.load_model(
    'best_brain_tumor_vgg16_model.h5',
    custom_objects={
        'precision_metric': precision_metric,
        'recall_metric': recall_metric,
        'f1_metric': f1_metric
    }
)

# In[48]:


test_loss, test_acc, test_precision, test_recall, test_f1 = best_vgg16.evaluate(test_generator)

print("\n" + "="*50)
print("TEST RESULTS - VGG16 MODEL")
print("="*50)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
print("="*50)

# In[49]:


test_generator.reset()
y_pred_vgg16 = best_vgg16.predict(test_generator, verbose=1)
y_pred_classes_vgg16 = np.argmax(y_pred_vgg16, axis=1)
y_true_vgg16 = test_generator.classes

cm_vgg16 = confusion_matrix(y_true_vgg16, y_pred_classes_vgg16)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_vgg16, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - VGG16', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.show()

print("\nClassification Report:")
print("="*70)
print(classification_report(y_true_vgg16, y_pred_classes_vgg16, target_names=CLASS_NAMES))

print("\n✅ VGG16 Model - Complete!")

# In[50]:


# from tensorflow.keras.applications import VGG19
# def build_vgg19_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
#     """Build VGG19 model with custom classification head"""
#     # Load pre-trained VGG19 without top layers
#     base_model = VGG19(
#         weights='imagenet',
#         include_top=False,
#         input_shape=input_shape
#     )
    
#     # Freeze base model layers
#     base_model.trainable = False
    
#     # Build complete model
#     model = keras.Sequential([
#         base_model,
#         keras.layers.Flatten(),
#         layers.Dense(256, activation='relu'),
#         layers.Dense(num_classes, activation='softmax')
#     ])
    
#     return model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense

def build_vgg19_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES):
    """Build VGG19 model with custom classification head"""
    
    # 1. Define explicit Input layer
    input_tensor = Input(shape=input_shape)
    
    # 2. Load pre-trained VGG19
    base_model = VGG19(
        weights='imagenet',
        include_top=False,
        input_tensor=input_tensor # <--- Connect Input here
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # 3. Build complete model using Functional API
    x = base_model.output
    x = Flatten()(x)     # Connect Flatten directly to the tensor 'x'
    x = Dense(256, activation='relu')(x)
    output_tensor = Dense(num_classes, activation='softmax')(x)
    
    # 4. Define the final Model
    model = Model(inputs=base_model.input, outputs=output_tensor) 
    
    return model
# Build VGG19 model
vgg19_model = build_vgg19_model()
vgg19_model.summary()


# In[51]:


early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    'best_brain_tumor_vgg19_model.h5',
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

# In[52]:


# ============== CELL 7: COMPILE AND TRAIN VGG19 ==============
vgg19_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy', precision_metric, recall_metric, f1_metric]
)

# In[53]:


print("Training VGG19 model...\n")

vgg19_history = vgg19_model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=1
)

print("\nVGG19 Training completed!")

# In[54]:


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

plot_training_history(vgg19_history, 'VGG19')

# In[55]:


best_vgg19 = keras.models.load_model(
    'best_brain_tumor_vgg19_model.h5',
    custom_objects={
        'precision_metric': precision_metric,
        'recall_metric': recall_metric,
        'f1_metric': f1_metric
    }
)

# In[56]:


test_loss, test_acc, test_precision, test_recall, test_f1 = best_vgg19.evaluate(test_generator)

print("\n" + "="*50)
print("TEST RESULTS - VGG19 MODEL")
print("="*50)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
print("="*50)

# In[57]:


test_generator.reset()
y_pred_vgg19 = best_vgg19.predict(test_generator, verbose=1)
y_pred_classes_vgg19 = np.argmax(y_pred_vgg19, axis=1)
y_true_vgg19 = test_generator.classes

cm_vgg19 = confusion_matrix(y_true_vgg19, y_pred_classes_vgg19)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_vgg19, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - VGG19', fontsize=16, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.show()

print("\nClassification Report:")
print("="*70)
print(classification_report(y_true_vgg19, y_pred_classes_vgg19, target_names=CLASS_NAMES))

print("\n✅ VGG19 Model - Complete!")

# In[58]:


results = {
    'Model': ['Custom CNN', 'VGG16', 'VGG19', 'MobileNet', 'ResNet50'],
    'Accuracy': [0.9359, 0.9115, 0.8940, 0.9397, 0.7765],  # <-- UPDATE THESE
    'Precision': [0.9353, 0.9154, 0.8955, 0.9402, 0.7826],  # <-- UPDATE THESE
    'Recall': [0.9329, 0.9061, 0.8887, 0.9398, 0.7576],  # <-- UPDATE THESE
    'F1 Score': [0.9340, 0.9106, 0.8920, 0.9400, 0.7696],  # <-- UPDATE THESE
}
 # Create DataFrame
import pandas as pd
df_results = pd.DataFrame(results)
print("\nModel Performance Comparison:")
print("="*80)
print(df_results.to_string(index=False))
print("="*80)

# In[59]:


sns.set_style("whitegrid")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Brain Tumor Classification - Models Comparison', 
             fontsize=18, fontweight='bold', y=1.00)

# Color palette
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8']

# Plot 1: Accuracy Comparison
ax1 = axes[0, 0]
bars1 = ax1.bar(df_results['Model'], df_results['Accuracy'], color=colors, alpha=0.8)
ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold', pad=15)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_ylim([0, 1.0])
ax1.grid(axis='y', alpha=0.3)
# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 2: Precision Comparison
ax2 = axes[0, 1]
bars2 = ax2.bar(df_results['Model'], df_results['Precision'], color=colors, alpha=0.8)
ax2.set_title('Precision Comparison', fontsize=14, fontweight='bold', pad=15)
ax2.set_ylabel('Precision', fontsize=12)
ax2.set_ylim([0, 1.0])
ax2.grid(axis='y', alpha=0.3)
# Add value labels on bars
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 3: Recall Comparison
ax3 = axes[1, 0]
bars3 = ax3.bar(df_results['Model'], df_results['Recall'], color=colors, alpha=0.8)
ax3.set_title('Recall Comparison', fontsize=14, fontweight='bold', pad=15)
ax3.set_ylabel('Recall', fontsize=12)
ax3.set_ylim([0, 1.0])
ax3.grid(axis='y', alpha=0.3)
# Add value labels on bars
for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 4: F1 Score Comparison
ax4 = axes[1, 1]
bars4 = ax4.bar(df_results['Model'], df_results['F1 Score'], color=colors, alpha=0.8)
ax4.set_title('F1 Score Comparison', fontsize=14, fontweight='bold', pad=15)
ax4.set_ylabel('F1 Score', fontsize=12)
ax4.set_ylim([0, 1.0])
ax4.grid(axis='y', alpha=0.3)
# Add value labels on bars
for bar in bars4:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# In[60]:


fig, ax = plt.subplots(figsize=(14, 8))

# Set the width of bars and positions
x = np.arange(len(df_results['Model']))
width = 0.2

# Create bars
bars1 = ax.bar(x - 1.5*width, df_results['Accuracy'], width, label='Accuracy', color='#FF6B6B', alpha=0.8)
bars2 = ax.bar(x - 0.5*width, df_results['Precision'], width, label='Precision', color='#4ECDC4', alpha=0.8)
bars3 = ax.bar(x + 0.5*width, df_results['Recall'], width, label='Recall', color='#45B7D1', alpha=0.8)
bars4 = ax.bar(x + 1.5*width, df_results['F1 Score'], width, label='F1 Score', color='#FFA07A', alpha=0.8)

# Add labels and title
ax.set_xlabel('Models', fontsize=14, fontweight='bold')
ax.set_ylabel('Score', fontsize=14, fontweight='bold')
ax.set_title('Brain Tumor Classification - All Metrics Comparison', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(df_results['Model'], fontsize=11)
ax.legend(fontsize=12, loc='lower right')
ax.set_ylim([0, 1.1])
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()

# In[61]:


print("\n" + "="*80)
print("BEST MODELS FOR EACH METRIC")
print("="*80)

best_accuracy_idx = df_results['Accuracy'].idxmax()
best_precision_idx = df_results['Precision'].idxmax()
best_recall_idx = df_results['Recall'].idxmax()
best_f1_idx = df_results['F1 Score'].idxmax()

print(f"Best Accuracy:  {df_results.loc[best_accuracy_idx, 'Model']:12s} - {df_results.loc[best_accuracy_idx, 'Accuracy']:.4f}")
print(f"Best Precision: {df_results.loc[best_precision_idx, 'Model']:12s} - {df_results.loc[best_precision_idx, 'Precision']:.4f}")
print(f"Best Recall:    {df_results.loc[best_recall_idx, 'Model']:12s} - {df_results.loc[best_recall_idx, 'Recall']:.4f}")
print(f"Best F1 Score:  {df_results.loc[best_f1_idx, 'Model']:12s} - {df_results.loc[best_f1_idx, 'F1 Score']:.4f}")
print("="*80)

# Overall best model (based on F1 score)
overall_best_idx = df_results['F1 Score'].idxmax()
print(f"\n🏆 OVERALL BEST MODEL: {df_results.loc[overall_best_idx, 'Model']}")
print(f"   F1 Score: {df_results.loc[overall_best_idx, 'F1 Score']:.4f}")
print("="*80)

print("\n✅ Models Comparison - Complete!")
