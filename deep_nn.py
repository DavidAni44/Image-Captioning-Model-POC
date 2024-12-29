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
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu

def extract_features(image_directory, save_path):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    pooling_layer = tf.keras.layers.GlobalAveragePooling2D()  # Added pooling layer
    encoder_model = Model(inputs=base_model.input, outputs=pooling_layer(base_model.output))

    features = {}
    for img_name in os.listdir(image_directory):
        img_path = os.path.join(image_directory, img_name)
        try:
            img = load_img(img_path, target_size=(224, 224))
            img = img_to_array(img)
            img = preprocess_input(img)
            img = np.expand_dims(img, axis=0)

            feature = encoder_model.predict(img).flatten()  # Flatten the pooled output
            features[os.path.splitext(img_name)[0]] = feature
        except Exception as e:
            print(f"Error processing image {img_name}: {e}")

    with open(save_path, 'wb') as f:
        pickle.dump(features, f)
    print(f"Feature extraction complete. Features saved to '{save_path}'.")

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

def create_tokenizer(captions, max_vocab_size=5000):
    caption_list = [caption for caption_group in captions.values() for caption in caption_group]
    tokenizer = Tokenizer(num_words=max_vocab_size, oov_token='<unk>')
    tokenizer.fit_on_texts(caption_list)
    return tokenizer

def caption_to_sequence(tokenizer, captions, max_length):
    sequences = {}
    for img_id, caption_group in captions.items():
        sequences[img_id] = []
        for caption in caption_group:
            seq = tokenizer.texts_to_sequences([caption])[0]
            seq = pad_sequences([seq], maxlen=max_length, padding='post')[0]
            sequences[img_id].append(seq)
    return sequences

def data_gen(captions, features, tokenizer, max_length, vocab_size, batch_size=32):
    feature_keys = set(features.keys())
    while True:
        X1, X2, y = [], [], []
        for img_id, caption_list in captions.items():
            if img_id not in feature_keys:
                print(f"Warning: {img_id} not found in features. Available keys: {list(feature_keys)[:5]}...")
                continue
            for caption in caption_list:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length, padding='post')[0]
                    out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(features[img_id])
                    X2.append(in_seq)
                    y.append(out_seq)
                    if len(X1) == batch_size:
                        yield (np.array(X1), np.array(X2)), np.array(y)
                        X1, X2, y = [], [], []

def build_image_captioning_model(vocab_size, max_length, feature_vector_size=512, embedding_dim=256, lstm_units=256):
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

def extract_single_feature(image_path, encoder_model):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return encoder_model.predict(img).flatten()

def generate_caption(model, tokenizer, image_feature, max_length):
    in_text = 'start'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post')[0]
        sequence = np.expand_dims(sequence, axis=0)
        yhat = model.predict([np.array([image_feature]), sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None or word == 'end':
            break
        in_text += ' ' + word
    return in_text.replace('start', '').strip()

def evaluate_bleu_score(model, tokenizer, features, captions, max_length):
    references = []
    candidates = []

    for img_id, ground_truth_captions in captions.items():
        if img_id not in features:
            print(f"Warning: Image ID {img_id} not found in features.")
            continue

        image_feature = features[img_id]
        generated_caption = generate_caption(model, tokenizer, image_feature, max_length)

        references.append([caption.split() for caption in ground_truth_captions])
        candidates.append(generated_caption.split())

    bleu1 = corpus_bleu(references, candidates, weights=(1.0, 0, 0, 0))
    bleu2 = corpus_bleu(references, candidates, weights=(0.5, 0.5, 0, 0))

    print(f"BLEU-1 Score: {bleu1:.4f}")
    print(f"BLEU-2 Score: {bleu2:.4f}")
    return bleu1, bleu2

image_directory = './archive/images'
save_path = './features.pkl'
#extract_features(image_directory, save_path)

caption_file = './archive/captions.txt'
captions = clean_captions(caption_file)
tokenizer = create_tokenizer(captions)
vocab_size = len(tokenizer.word_index) + 1

all_captions = [caption for caption_group in captions.values() for caption in caption_group]
max_length = max(len(tokenizer.texts_to_sequences([caption])[0]) for caption in all_captions)

with open('./features.pkl', 'rb') as f:
    features = pickle.load(f)

features = {os.path.splitext(key)[0]: value for key, value in features.items()}

batch_size = 32
steps_per_epoch = sum(len(caption_list) for caption_list in captions.values()) // batch_size
generator = data_gen(captions, features, tokenizer, max_length, vocab_size, batch_size)

dataset = tf.data.Dataset.from_generator(
    lambda: data_gen(captions, features, tokenizer, max_length, vocab_size, batch_size),
    output_signature=(
        (tf.TensorSpec(shape=(None, 512), dtype=tf.float32), tf.TensorSpec(shape=(None, max_length), dtype=tf.float32)),
        tf.TensorSpec(shape=(None, vocab_size), dtype=tf.float32)
    )
)

model = build_image_captioning_model(vocab_size, max_length)
model.fit(dataset, steps_per_epoch=steps_per_epoch, epochs=20, verbose=1)

model.save('image_captioning_model.keras')
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
encoder_model = Model(inputs=base_model.input, outputs=tf.keras.layers.GlobalAveragePooling2D()(base_model.output))

test_image_path = r'test images/kitty.jpg'
image_feature = extract_single_feature(test_image_path, encoder_model)
caption = generate_caption(model, tokenizer, image_feature, max_length)
print("Generated Caption:", caption)

# Evaluate BLEU Score
evaluate_bleu_score(model, tokenizer, features, captions, max_length)
