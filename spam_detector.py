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
    # Đọc file CSV tiếng Việt từ thư mục hiện tại
    file_path = "vi_dataset.csv"
    df = pd.read_csv(file_path)
    # Các cột đã được đặt tên đúng (labels, texts_vi)
    df = df.rename(columns={'labels': 'label', 'texts_vi': 'text'})
    return df

def preprocess_text(text):
    # Xử lý giá trị NaN
    if pd.isna(text):
        return ""
    # Chuyển về string nếu không phải
    text = str(text)
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
    print("Đang tải dữ liệu tiếng Việt...")
    df = load_data()
    
    # Tiền xử lý văn bản
    print("Đang tiền xử lý văn bản tiếng Việt...")
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
    print("Đang xây dựng và huấn luyện mô hình...")
    model = build_model(vocab_size=max_words)
    
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Đánh giá mô hình
    print("\nĐang đánh giá mô hình...")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nĐộ chính xác kiểm thử: {accuracy:.4f}")
    
    # Vẽ biểu đồ loss và accuracy
    plt.figure(figsize=(12, 4))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Mất mát khi huấn luyện')
    plt.plot(history.history['val_loss'], label='Mất mát khi kiểm định')
    plt.title('Mất mát của mô hình')
    plt.xlabel('Epoch')
    plt.ylabel('Mất mát')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Độ chính xác khi huấn luyện')
    plt.plot(history.history['val_accuracy'], label='Độ chính xác khi kiểm định')
    plt.title('Độ chính xác của mô hình')
    plt.xlabel('Epoch')
    plt.ylabel('Độ chính xác')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    # Vẽ confusion matrix
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Ma trận nhầm lẫn')
    plt.ylabel('Nhãn thực tế')
    plt.xlabel('Nhãn dự đoán')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Lưu mô hình
    model.save('spam_detector_model.h5')
    
    # Lưu tokenizer
    import pickle
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("\nMô hình đã được lưu thành 'spam_detector_model.h5'")
    print("Tokenizer đã được lưu thành 'tokenizer.pickle'")
    print("Biểu đồ huấn luyện đã được lưu thành 'training_history.png'")
    print("Ma trận nhầm lẫn đã được lưu thành 'confusion_matrix.png'")

if __name__ == "__main__":
    main()
