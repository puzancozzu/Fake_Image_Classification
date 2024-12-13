import tensorflow as tf
from keras import layers,Model
from keras.callbacks import ModelCheckpoint, EarlyStopping # type: ignore
import os
from tensorflow.keras.models import load_model # type: ignore
import numpy as np
import cv2 
import keras
import matplotlib.pyplot as plt # type: ignore

from smash_img import smash_n_reconstruct
from filters import apply_all_filters

@tf.function
def hard_tanh(x):
    return tf.maximum(tf.minimum(x, 1), -1)

class featureExtractionLayer(layers.Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv = layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu')
        self.bn = layers.BatchNormalization()
        self.activation = layers.Lambda(hard_tanh)
        
    
    def call(self, input):
        x = self.conv(input)
        x = self.bn(x)
        x = self.activation(x)
        return x

input1 = layers.Input(shape=(256,256,1),name="rich_texture")
input2 = layers.Input(shape=(256,256,1),name="poor_texture")

l1 = featureExtractionLayer(name="feature_extraction_layer_rich_texture")(input1)
l2 = featureExtractionLayer(name="feature_extraction_layer_poor_texture")(input2)

contrast = layers.subtract((l1,l2))

x = layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu')(contrast)
x = layers.BatchNormalization()(x)
for i in range(3):
    x = layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu')(x)
    x = layers.BatchNormalization()(x)
x = layers.BatchNormalization()(x)

for i in range(4):
    x = layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu')(x)
    x = layers.BatchNormalization()(x)
x = layers.AveragePooling2D(3,3)(x)

for i in range(2):
    x = layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu')(x)
    x = layers.BatchNormalization()(x)
x = layers.AveragePooling2D(3,3)(x)

for i in range(2):
    x = layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu')(x)
    x = layers.BatchNormalization()(x)
x = layers.GlobalAveragePooling2D()(x)

x = layers.Flatten()(x)
x = layers.Dense(1,activation='sigmoid')(x)

model = Model(inputs=(input1,input2), outputs=x, name="rich_texture_poor_texture_contrast")
model.compile(
                optimizer='adam',
                loss='BinaryCrossentropy',
                metrics=['binary_accuracy']
            )

def preprocess_image(path):
    # Convert path to string if it's a tensor
    path = path.numpy().decode('utf-8') if tf.is_tensor(path) else path
    
    try:
        # Perform preprocessing
        rt, pt = smash_n_reconstruct(path)

        plt.subplot(1,2,1)
        plt.imshow(rt)
        plt.title('rich texture - AI generated image')
        plt.subplot(1,2,2)
        plt.imshow(pt)
        plt.title('poor texture - AI generated')
        plt.show()
        
        rt = apply_all_filters(rt)
        pt = apply_all_filters(pt)
        plt.subplot(1,2,1)
        plt.imshow(rt, cmap = 'gray')
        plt.title('rich texture - AI generated image')
        plt.subplot(1,2,2)
        plt.imshow(pt, cmap = 'gray')
        plt.title('poor texture - AI generated')
        plt.show()

        rt, pt = smash_n_reconstruct(path)
        
        # Apply filters and ensure consistent shape
        frt = tf.cast(tf.expand_dims(apply_all_filters(rt), axis=-1), dtype=tf.float64)
        fpt = tf.cast(tf.expand_dims(apply_all_filters(pt), axis=-1), dtype=tf.float64)
        
        # Ensure shape
        frt = tf.ensure_shape(frt, [256, 256, 1])
        fpt = tf.ensure_shape(fpt, [256, 256, 1])
        
        # Add batch dimension, so shape becomes [1, 256, 256, 1]
        frt = tf.expand_dims(frt, axis=0)
        fpt = tf.expand_dims(fpt, axis=0)
        
        return frt, fpt
        
    except Exception as e:
        print(f"Error processing {path}: {e}")
        # Return dummy tensors to avoid breaking the pipeline
        dummy = tf.zeros([256, 256, 1], dtype=tf.float32)
        return dummy, dummy


# Register the custom layer with Keras
keras.utils.get_custom_objects()['featureExtractionLayer'] = featureExtractionLayer

# Load the trained model from the .h5 file
model = load_model('d:/Project/model_checkpoint_all_purpose.keras')

# Example test data
test_image_path = 'd:/Project/test.webp'

# Display image
image = cv2.imread(test_image_path)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

# preprocess image
test_frt, test_fpt = preprocess_image(test_image_path)

# Make a prediction
predictions = model.predict([test_frt, test_fpt])
print(predictions)

if predictions > 0.5:
    print("Fake")
else:
    print("Real")