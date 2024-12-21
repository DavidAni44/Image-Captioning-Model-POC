#resources used https://www.tensorflow.org/text/tutorials/image_captioning, https://blog.paperspace.com/image-captioning-with-ai/, https://blog.paperspace.com/image-captioning-with-tensorflow/, Github copilot


import numpy as np
import pandas as pd
import tensorflow as tf
import os
import pickle
import string
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Dropout, Add
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def extract_features(image_directory, save_path):
    # Load pre-trained VGG16
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    encoder_model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

    features = {}
    for img_name in os.listdir(image_directory):
        img_path = os.path.join(image_directory, img_name)
        try:
            img = load_img(img_path, target_size=(224, 224))
            img = img_to_array(img)
            img = preprocess_input(img)
            img = np.expand_dims(img, axis=0)  # Add batch dimension

            feature = encoder_model.predict(img)
            feature = feature.flatten()
            feature = np.expand_dims(feature, axis=0)  # Add batch dimension
            feature = Dense(4096, activation='relu')(feature)  # Reduce dimensionality to 4096
            features[os.path.splitext(img_name)[0]] = feature.numpy().flatten()
        except Exception as e:
            print(f"Error processing image {img_name}: {e}")

    with open(save_path, 'wb') as f:
        pickle.dump(features, f)
    print(f"Feature extraction complete. Features saved to '{save_path}'.")

# Extract features and save them
image_directory = './archive/images'
save_path = './features.pkl'

#extract_features(image_directory, save_path)


def clean_captions(caption_file):
    captions = {}
    with open(caption_file, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',', 1)
            if len(parts) < 2:
                continue
            img_id, caption = parts
            img_id = img_id.split('.')[0]
            caption = caption.lower()
            caption = caption.translate(str.maketrans('', '', string.punctuation))
            caption = "start " + caption + " end"
            if img_id not in captions:
                captions[img_id] = []
            captions[img_id].append(caption)
    return captions

# Load and clean captions
caption_file = './archive/captions.txt'
captions = clean_captions(caption_file)

def create_tokenizer(captions, max_vocab_size=5000):
    caption_list = [caption for caption_group in captions.values() for caption in caption_group]
    tokenizer = Tokenizer(num_words=max_vocab_size, oov_token='<unk>')
    tokenizer.fit_on_texts(caption_list)
    return tokenizer

tokenizer = create_tokenizer(captions)

#turn words into numbers so it can be passed into the model 
def caption_to_sequence(tokenizer, captions, max_length):
    sequences = {}
    for img_id, caption_group in captions.items():
        sequences[img_id] = []
        for caption in caption_group:
            # Convert caption to sequence of integers
            seq = tokenizer.texts_to_sequences([caption])[0]
            # Pad the sequence
            seq = pad_sequences([seq], maxlen=max_length, padding='post')[0]
            sequences[img_id].append(seq)
    return sequences

# Calculate max_length based on the captions
all_captions = [caption for caption_group in captions.values() for caption in caption_group]
max_length = max(len(tokenizer.texts_to_sequences([caption])[0]) for caption in all_captions)

# Example usage
sequences = caption_to_sequence(tokenizer, captions, max_length)

# Padding the sequences so they are all the same length
def pad_caption_sequences(sequences, max_length):
    padded_sequences = {}
    for img_id, seq_list in sequences.items():
        padded_sequences[img_id] = pad_sequences(seq_list, maxlen=max_length, padding='post')
    return padded_sequences

# Example usage
padded_sequences = pad_caption_sequences(sequences, max_length)



def data_gen(captions, features, tokenizer, max_length, vocab_size, batch_size=32):
    n = 0
    X1, X2, y = [], [], []
    feature_keys = set(features.keys())
    while True:
        for img_id, caption_list in captions.items():
            if img_id not in feature_keys:
                print(f"Warning: {img_id} not found in features. Available keys: {list(feature_keys)[:5]}...")  # Print first 5 keys for debugging
                continue
            for caption in caption_list:
                # Convert caption to sequence of integers
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    # Create input-output pairs from tokenized captions
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length, padding='post')[0]
                    out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=vocab_size)[0]
                    
                    # Append to batch
                    X1.append(features[img_id])
                    X2.append(in_seq)
                    y.append(out_seq)
                    n += 1
                    if n == batch_size:
                        yield (np.array(X1), np.array(X2)), np.array(y)
                        X1, X2, y = [], [], []
                        n = 0

                        
def build_image_captioning_model(vocab_size, max_length, feature_vector_size=4096, embedding_dim=256, lstm_units=256):
    image_input = Input(shape=(feature_vector_size,))
    image_dense = Dense(embedding_dim, activation='relu')(image_input)

    caption_input = Input(shape=(max_length,))
    caption_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)(caption_input)
    caption_lstm = LSTM(lstm_units)(caption_embedding)

    decoder_input = Add()([image_dense, caption_lstm])
    decoder_output = Dense(vocab_size, activation='softmax')(decoder_input)

    model = Model(inputs=[image_input, caption_input], outputs=decoder_output)
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model

with open('./features.pkl', 'rb') as f:
    features = pickle.load(f)
    
features = {os.path.splitext(key)[0]: value for key, value in features.items()}

# Define parameters
batch_size = 32
vocab_size = len(tokenizer.word_index) + 1  # Define vocab_size
max_length = max(len(seq) for seq_list in sequences.values() for seq in seq_list)  # Calculate max_length

# Build the model
model = build_image_captioning_model(vocab_size, max_length)

# Calculate steps per epoch
steps_per_epoch = sum(len(caption_list) for caption_list in captions.values()) // batch_size

# Initialize the generator
generator = data_gen(captions, features, tokenizer, max_length, vocab_size, batch_size)


output_signature = (
    (tf.TensorSpec(shape=(None, 4096), dtype=tf.float32), tf.TensorSpec(shape=(None, max_length), dtype=tf.float32)),
    tf.TensorSpec(shape=(None, vocab_size), dtype=tf.float32)
)


dataset = tf.data.Dataset.from_generator(
    lambda: generator,
    output_signature=output_signature
)

model.fit(generator, steps_per_epoch=steps_per_epoch, epochs=50, verbose=1)

model.save('image_captioning_model.h5')
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
    

def extract_single_feature(image_path, encoder_model):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    feature = encoder_model.predict(img)
    feature = feature.flatten()
    feature = np.expand_dims(feature, axis=0)  # Add batch dimension
    feature = Dense(4096, activation='relu')(feature)  # Reduce dimensionality to 4096
    return feature.numpy().flatten()


def generate_caption(model, tokenizer, image_feature, max_length):
    in_text = 'start'
    
    # Ensure image_feature has the correct shape (1, 4096)
    if image_feature.ndim == 1:  # If it's (4096,)
        image_feature = np.expand_dims(image_feature, axis=0)

    for _ in range(max_length):
        # Convert text to sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # Pad the sequence to max_length
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')[0]
        
        # Ensure sequence has the correct shape (1, max_length)
        sequence = np.expand_dims(sequence, axis=0)

        # Debugging: Print shapes before prediction
        print(f"Image Feature Shape: {image_feature.shape}")  # Should be (1, 4096)
        print(f"Sequence Shape: {sequence.shape}")            # Should be (1, max_length)

        # Predict next word
        yhat = model.predict([image_feature, sequence], verbose=0)
        yhat = np.argmax(yhat)  # Get the word index with the highest probability
        
        # Map word index to word
        word = tokenizer.index_word.get(yhat)
        if word is None or word == 'end':
            break
        in_text += ' ' + word

    return in_text.replace('start', '').strip()



base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
encoder_model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

test_image_path = 'test images\kitty.jpg'
image_feature = extract_single_feature(test_image_path, encoder_model)
caption = generate_caption(model, tokenizer, image_feature, max_length)
print("Generated Caption:", caption)