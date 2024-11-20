import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import string
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

image_directory = './archive'
image_generator = data_generator.flow_from_directory(
    image_directory,
    target_size=(224, 224),
    batch_size=32,
    class_mode=None,
    shuffle=False
)


base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
encoder_model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

features = {}

for i, image_batch in enumerate(image_generator):
    batch_features = encoder_model.predict(image_batch)
    for j, feature in enumerate(batch_features):
      img_fname = image_generator.filenames[i * image_generator.batch_size + j]
      features[img_fname] = feature.flatten()
    if i >= len(image_generator) - 1:
        break

with open('/Users/david/Desktop/AI-ML-assignmentfeatures.pkl', 'wb') as f:
    pickle.dump(features, f)

print("Feature extraction complete. Features saved to 'features.pkl'.")

import string

def clean_captions(caption_file):
    captions = {}
    with open(caption_file, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                # Skip empty lines
                continue
            
            # Split line into img_id and caption
            parts = line.split(' ', 1)
            
            if len(parts) < 2:
                continue
            
            img_id, caption = parts
            img_id = img_id.split('.')[0]  
            
            # Clean the caption: lowercase, remove punctuation, add start and end tokens
            caption = caption.lower()
            caption = caption.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
            caption = "start " + caption + " end"  # Add start and end tokens
            
            if img_id not in captions:
                captions[img_id] = []
            captions[img_id].append(caption)
    
    return captions

caption_file = './archive/captions.txt'
captions = clean_captions(caption_file)

            
            
            