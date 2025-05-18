import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import re
import string

# Load và tiền xử lý dữ liệu
def load_data():
    # Đọc file CSV từ thư mục hiện tại
    file_path = "spam.csv"
    df = pd.read_csv(file_path)
    # Đổi tên cột để phù hợp với code
    df = df.rename(columns={'Category': 'label', 'Message': 'text'})
    return df

def preprocess_text(text):
    # Chuyển text về chữ thường
    text = text.lower()
    # Loại bỏ dấu câu
    text = ''.join([char for char in text if char not in string.punctuation])
    # Loại bỏ số
    text = re.sub('[0-9]+', '', text)
    return text

# Xây dựng mô hình RNN
def build_model(vocab_size, embedding_dim=100, max_length=200):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    return model

def main():
    # Tải dữ liệu
    print("Loading data...")
    df = load_data()
    
    # Tiền xử lý văn bản
    print("Preprocessing texts...")
    texts = df['text'].apply(preprocess_text).values
    labels = (df['label'] == 'spam').astype(int).values
    
    # Tokenize văn bản
    max_words = 10000  # Số lượng từ tối đa trong từ điển
    max_length = 200   # Độ dài tối đa của mỗi email
    
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded_sequences = pad_sequences(sequences, maxlen=max_length)
    
    # Chia dữ liệu thành tập train và test
    X_train, X_test, y_train, y_test = train_test_split(
        padded_sequences, labels, test_size=0.2, random_state=42
    )
    
    # Xây dựng và huấn luyện mô hình
    print("Building and training model...")
    model = build_model(vocab_size=max_words)
    
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
      # Đánh giá mô hình
    print("\nEvaluating model...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {accuracy:.4f}")
    
    # Vẽ biểu đồ loss và accuracy
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    # Vẽ confusion matrix
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Lưu mô hình
    model.save('spam_detector_model.h5')
    
    # Lưu tokenizer
    import pickle
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("\nModel saved as 'spam_detector_model.h5'")
    print("Tokenizer saved as 'tokenizer.pickle'")
    print("Training history plot saved as 'training_history.png'")
    print("Confusion matrix plot saved as 'confusion_matrix.png'")

if __name__ == "__main__":
    main()
