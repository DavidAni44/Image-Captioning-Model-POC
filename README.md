
# Image Captioning with ResNet and LSTM

## Overview

This project implements an image captioning model using a deep learning encoder-decoder architecture, combining computer vision and natural language processing to generate meaningful textual descriptions for images.

Two versions of the model are implemented:

ResNet50 Encoder: Utilizes a pre-trained ResNet50 CNN to extract rich, high-level visual features from images.

VGG16 Encoder: An alternate version using VGG16, offering a simpler but effective approach to feature extraction.

In both cases, the decoder is an LSTM (Long Short-Term Memory) network trained to generate coherent and grammatically sound image captions.

The model is trained and evaluated on the Flickr8k dataset, which contains 8,000 images, each annotated with five human-generated captions.

This solution is designed with clarity and modularity, making it approachable for both technical users (developers, ML practitioners) and non-technical stakeholders (product managers, analysts). It’s well-suited for real-world use cases such as:

Content accessibility (alt-text generation)

Automated media tagging

Image-based search enhancement

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
├── tokenizer_vgg16.pkl   # Saved image feature vectors
├── tokenizer_resnet.pkl  # Trained tokenizer
├── image_captioning_model.keras  # Trained model
├── image_captioning_model.h5  # Trained model
├── deep_nn.py            # Main training and evaluation script
├── deep_nn_vgg.py            # Training and evaluation script using vgg
├── deep_nn.ipynb            # Neural network captured on jupyter notebook
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
