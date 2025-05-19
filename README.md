# Email Spam Detector using RNN/LSTM

This project implements an email spam detector using Recurrent Neural Networks (RNN) with LSTM layers. The model is built using TensorFlow/Keras and achieves high accuracy in classifying emails as spam or non-spam.

## Features

- Text preprocessing and tokenization
- LSTM-based neural network architecture
- Interactive GUI for testing emails
- Model performance visualization
- Easy-to-use interface

## Requirements

- Python 3.11
- TensorFlow
- NumPy
- Pandas
- scikit-learn
- Tkinter (for GUI)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/BNgocSon666/email-spam-rnn-.git
```

2. Install required packages:
```bash
pip install tensorflow numpy pandas scikit-learn
```

## Usage

1. Train the model:
```bash
python spam_detector.py
```

2. Test emails using the GUI:
```bash
python test_spam_detector.py
```

## Model Architecture

The model uses:
- Embedding layer for text vectorization
- Two LSTM layers (64 and 32 units)
- Dense layers for classification
- Binary cross-entropy loss and Adam optimizer

## Performance

The model achieves approximately 98% accuracy on the test set.
