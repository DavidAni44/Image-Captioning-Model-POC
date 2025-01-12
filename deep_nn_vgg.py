import numpy as np
import tensorflow as tf
import os
import pickle
import string
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, LSTM, Dropout, Add, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

# Feature Extraction with VGG16
def extract_features(image_directory, save_path, batch_size=32):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    encoder_model = Model(inputs=base_model.input, outputs=Flatten()(base_model.output))

    features = {}
    img_names = os.listdir(image_directory)
    for i in range(0, len(img_names), batch_size):
        batch_names = img_names[i:i+batch_size]
        batch_images = []
        for img_name in batch_names:
            img_path = os.path.join(image_directory, img_name)
            try:
                img = load_img(img_path, target_size=(224, 224))
                img = img_to_array(img)
                img = preprocess_input(img)
                batch_images.append(img)
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                continue
        if batch_images:
            batch_images = np.array(batch_images)
            batch_features = encoder_model.predict(batch_images)
            for j, img_name in enumerate(batch_names):
                features[os.path.splitext(img_name)[0]] = batch_features[j]

    with open(save_path, 'wb') as f:
        pickle.dump(features, f)

def extract_single_feature(image_path):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    encoder_model = Model(inputs=base_model.input, outputs=Flatten()(base_model.output))
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    feature = encoder_model.predict(img).flatten()
    return feature

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

def data_gen(captions, features, tokenizer, max_length, vocab_size, batch_size=32):
    feature_keys = set(features.keys())
    while True:
        X1, X2, y = [], [], []
        for img_id, caption_list in captions.items():
            if img_id not in feature_keys:
                continue
            for caption in caption_list:
                seq = tokenizer.texts_to_sequences([caption])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length, padding='post')[0]
                    X1.append(features[img_id])
                    X2.append(in_seq)
                    y.append(out_seq)
                    if len(X1) == batch_size:
                        yield (np.array(X1), np.array(X2)), np.array(y)
                        X1, X2, y = [], [], []

def build_image_captioning_model(vocab_size, max_length, feature_vector_size=25088, embedding_dim=256, lstm_units=256):
    image_input = Input(shape=(feature_vector_size,))
    image_dense = Dense(embedding_dim, activation='relu')(image_input)
    caption_input = Input(shape=(max_length,))
    caption_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True)(caption_input)
    caption_lstm = LSTM(lstm_units)(caption_embedding)
    decoder_input = Add()([image_dense, caption_lstm])
    decoder_output = Dense(vocab_size, activation='softmax')(decoder_input)
    model = Model(inputs=[image_input, caption_input], outputs=decoder_output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def generate_caption_beam_search(model, tokenizer, image_feature, max_length, beam_width=10):
    sequences = [[list(), 1.0]]  # (sequence, score)
    for _ in range(max_length):
        all_candidates = []
        for seq, score in sequences:
            sequence = pad_sequences([seq], maxlen=max_length, padding='post')
            predictions = model.predict([np.array([image_feature]), sequence], verbose=0)[0]
            top_indices = np.argsort(predictions)[-beam_width:]
            for idx in top_indices:
                candidate = seq + [idx]
                candidate_score = score * predictions[idx]
                all_candidates.append((candidate, candidate_score))
        ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
        sequences = ordered[:beam_width]
    final_sequence = sequences[0][0]
    words = [tokenizer.index_word.get(idx, '') for idx in final_sequence if idx > 0]
    caption = ' '.join(words)
    return caption.replace('start', '').replace('end', '').strip()

# BLEU
def evaluate_bleu_score(model, tokenizer, features, captions, max_length):
    references, candidates = [], []
    for img_id, ground_truth_captions in captions.items():
        if img_id not in features:
            continue
        image_feature = features[img_id]
        generated_caption = generate_caption_beam_search(model, tokenizer, image_feature, max_length)
        references.append([caption.split() for caption in ground_truth_captions])
        candidates.append(generated_caption.split())
    bleu1 = corpus_bleu(references, candidates, weights=(1.0, 0, 0, 0))
    bleu2 = corpus_bleu(references, candidates, weights=(0.5, 0.5, 0, 0))
    print(f"BLEU-1 Score: {bleu1:.4f}")
    print(f"BLEU-2 Score: {bleu2:.4f}")

# Split data into train, validation, and test sets
def split_data(captions, test_size=0.2, val_size=0.1):
    image_ids = list(captions.keys())
    train_ids, temp_ids = train_test_split(image_ids, test_size=(test_size + val_size), random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=test_size / (test_size + val_size), random_state=42)
    return (
        {img_id: captions[img_id] for img_id in train_ids},
        {img_id: captions[img_id] for img_id in val_ids},
        {img_id: captions[img_id] for img_id in test_ids},
    )

# Execution Pipeline
image_directory = './archive/images'
caption_file = './archive/captions.txt'
save_path = './features_vgg16.pkl'

extract_features(image_directory, save_path)

# Load and clean captions
captions = clean_captions(caption_file)

# Split captions into training, validation, and test sets
train_captions, val_captions, test_captions = split_data(captions, test_size=0.1, val_size=0.1)

# Tokenize captions
tokenizer = create_tokenizer(train_captions)
vocab_size = len(tokenizer.word_index) + 1
all_captions = [caption for caption_group in train_captions.values() for caption in caption_group]
max_length = max(len(tokenizer.texts_to_sequences([caption])[0]) for caption in all_captions)

# Load precomputed features
with open(save_path, 'rb') as f:
    features = pickle.load(f)

# Build and train the model
model = build_image_captioning_model(vocab_size, max_length)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

batch_size = 32
steps_per_epoch = sum(len(caption_list) for caption_list in train_captions.values()) // batch_size
validation_steps = sum(len(caption_list) for caption_list in val_captions.values()) // batch_size

model.fit(
    data_gen(train_captions, features, tokenizer, max_length, vocab_size, batch_size),
    steps_per_epoch=steps_per_epoch,
    validation_data=data_gen(val_captions, features, tokenizer, max_length, vocab_size, batch_size),
    validation_steps=validation_steps,
    epochs=50,
    callbacks=[early_stopping],
    verbose=1
)

model.save('image_captioning_model_vgg16.keras')
with open('tokenizer_vgg16.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

# Generate caption for a test image
test_image_path = './test_images/mountain.jpg'
image_feature = extract_single_feature(test_image_path)
caption = generate_caption_beam_search(model, tokenizer, image_feature, max_length)
print("Generated Caption:", caption)

# Evaluate BLEU scores on test set
evaluate_bleu_score(model, tokenizer, features, test_captions, max_length)
