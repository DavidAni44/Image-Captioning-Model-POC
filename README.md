
# Image Captioning with ResNet and LSTM

## Overview

This project implements an image captioning model using a deep learning encoder-decoder architecture. The encoder uses a pre-trained ResNet50 model to extract high-level features from images, while the decoder uses an LSTM network to generate natural language descriptions of those images. The goal is to bridge computer vision and natural language processing, automatically generating meaningful captions for images.

The model is trained and tested on the Flickr8k dataset, which consists of 8,000 images with five human-annotated captions each. The solution is designed to be accessible to both technical and non-technical audiences, and is suitable for deployment or further development in a professional setting.

---

## Key Features

- **Encoder**: Utilizes ResNet50, a powerful CNN trained on ImageNet, to generate 2048-dimensional feature vectors.
- **Decoder**: LSTM-based network that uses embeddings and context to generate captions sequentially.
- **Beam Search**: Generates more accurate captions by exploring multiple possible word sequences.
- **BLEU Score Evaluation**: Assesses the quality of generated captions compared to human-annotated references.
- **Early Stopping**: Reduces overfitting by halting training when performance on validation data stops improving.
- **Data Splitting**: Properly separates data into training, validation, and testing subsets to ensure robust evaluation.

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- NLTK (for BLEU score)
- Scikit-learn
- ResNet50 (pre-trained model)
- VGG (pre-trained model)

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/DavidAni44/Image-Classification-model.poc.git
cd Image-Classification-model.poc
```

### 2. Set up a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the training script
```bash
python deep_nn.py
```

---

## Folder Structure

```
├── test_images/          # Folder with unseen images for caption generation
├── features_resnet.pkl   # Saved image feature vectors
├── tokenizer_resnet.pkl  # Trained tokenizer
├── image_captioning_model_resnet.keras  # Trained model
├── deep_nn.py            # Main training and evaluation script
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

---

## Example Output

**Input Image:**
A photo of a car

**Generated Caption:**
`a black car is parked on the street`

**BLEU-1 Score:** 0.51  
**BLEU-2 Score:** 0.34

---

## Future Improvements

- Integrate attention mechanisms for more accurate captioning.
- Use larger datasets like Flickr30k or MSCOCO.
- Experiment with transformer-based models like ViT + GPT.
- Deploy as a web service using Flask or FastAPI.

---

## License

This project is open-source and available under the MIT License.

---

## Acknowledgements

- Flickr8k Dataset by University of Illinois at Urbana-Champaign
- ResNet50 pre-trained model by Kaiming He et al.
