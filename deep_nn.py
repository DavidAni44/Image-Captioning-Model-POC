#resources used https://www.tensorflow.org/text/tutorials/image_captioning, https://blog.paperspace.com/image-captioning-with-ai/, https://blog.paperspace.com/image-captioning-with-tensorflow/, Github copilot


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

with open('../features.pkl', 'wb') as f:
    pickle.dump(features, f)

print("Feature extraction complete. Features saved to 'features.pkl'.")


def clean_captions(caption_file: str) -> dict[str, list[str]]:
    captions = {}
    with open(caption_file, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                # Skip empty lines
                continue
            
            # Split using comma instead of space
            parts = line.split(',', 1)
            
            if len(parts) < 2:
                continue
            
            img_id, caption = parts
            img_id = img_id.split('.')[0]  # Remove file extension
            
            # Clean the caption: lowercase, remove punctuation, add start and end tokens
            caption = caption.lower()
            caption = caption.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
            caption = "start " + caption + " end"  # Add start and end tokens
            
            # Add to captions dictionary
            if img_id not in captions:
                captions[img_id] = []
            captions[img_id].append(caption)
    
    return captions

def create_tokenizer(captions: dict[str, list[str]], max_vocab_size: int = 5000) -> Tokenizer:
    caption_list = []
    for caption_group in captions.values():
        for caption in caption_group:
            caption_list.append(caption)
    
    # Create tokenizer
    tokenizer = Tokenizer(num_words=max_vocab_size, oov_token='<unk>')
    tokenizer.fit_on_texts(caption_list)
    
    return tokenizer
#turn words into numbers so it can be passed into the model 
def caption_to_sequence(tokenizer: Tokenizer, captions: dict[str, list[str]]) -> dict[str, list[np.ndarray]]:
    sequences = {}
    for img_id, caption_group in captions.items():
        sequences[img_id] = tokenizer.texts_to_sequences(caption_group)
    
    return sequences

#padding the sequences so they are all the same length
def pad_caption_sequences(sequences: dict[str, list[np.ndarray]], max_length: int) -> dict[str, np.ndarray]:
    padded_sequences = {}
    for img_id, sequence_group in sequences.items():
        padded_sequences[img_id] = pad_sequences(sequence_group, maxlen=max_length, padding='post')
    return padded_sequences

def data_gen(captions, features, tokenizer, max_length, vocab_size, batch_size=32):
    #create batch variables
    n = 0
    X1, X2, y = list(), list(), list()
    while True:
        for img_id, caption_list in captions.items():
            for caption in caption_list:
                for i in range(1, len(caption)):
                    #input-output pairs
                    in_seq, out_seq = caption[:i], caption[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=vocab_size)[0]
                    #append to batch
                    X1.append(features[img_id])
                    X2.append(in_seq)
                    y.append(out_seq)
                    n += 1
                    #yield the batch no return needed
                    if n == batch_size:
                        yield ([np.array(X1), np.array(X2)], np.array(y))
                        X1, X2, y = list(), list(), list()
                        n = 0




# Test with the caption file
caption_file = './archive/captions.txt'
try:
    captions = clean_captions(caption_file)
    print("Captions cleaned successfully.")
except Exception as e:
    print("Error:", e)

            
            
            