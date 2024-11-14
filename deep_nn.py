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

image_directory = '/Users/david/Desktop/AI-ML-assignment'
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

def clean_captions(caption_file):
    captions = {}
    with open(caption_file, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                #if line is empty i.e not true skip the line
                continue
            img_id, caption = line.split(' ', 1)
            img_id = img_id.split('.')[0]
            #clean caption .i.e remove lowercase, remove punctuation, and start and end token 
            caption = caption.lowercase()
            caption = caption.translate(str.maketrans('','',string.punctuation))
            caption = "start" + caption + "end"
            
            if img_id not in captions:
                captions[img_id] = []
            captions[img_id].append(caption)
    return captions

caption_file = '/Users/david/Desktop/AI-ML-assignment/archive/captions.txt'
captions = clean_captions(caption_file)
            
            
            