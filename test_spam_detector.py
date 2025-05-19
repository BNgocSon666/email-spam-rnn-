import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tkinter as tk
from tkinter import ttk, scrolledtext
import pickle
import re
import string

def preprocess_text(text):
    # Handle NaN values
    if pd.isna(text):
        return ""
    # Convert to string if not already
    text = str(text)
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    # Remove numbers
    text = re.sub('[0-9]+', '', text)
    return text

def predict_spam(model, text, tokenizer, max_length=200):
    # Preprocess the text
    text = preprocess_text(text)
    
    # Convert text to sequences of numbers
    sequences = tokenizer.texts_to_sequences([text])
    
    # Pad sequences to fixed length
    padded_seq = pad_sequences(sequences, maxlen=max_length)
    
    # Make prediction
    prediction = model.predict(padded_seq)[0][0]
    
    return {
        'text': text,
        'is_spam': prediction > 0.5,
        'confidence': float(prediction)
    }

# Load model
model = load_model('spam_detector_model.h5')

# Load the saved tokenizer
try:
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
except:
    print("Không thể tải tokenizer đã lưu, sử dụng tokenizer mới")
    tokenizer = Tokenizer()

class ScrollableFrame(ttk.Frame):
    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)
        canvas = tk.Canvas(self)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

class SpamDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Phát hiện Email Spam")
        self.root.geometry("600x700")

        # Load model and tokenizer
        self.load_model()

        # Create main UI
        self.create_input_section()
        self.create_result_section()
        self.create_examples_section()

    def create_input_section(self):
        input_frame = ttk.Frame(self.root, padding="10")
        input_frame.pack(fill=tk.X)
        
        ttk.Label(input_frame, text="Nhập nội dung email:").pack()
        
        self.text_input = scrolledtext.ScrolledText(input_frame, height=10, width=60)
        self.text_input.pack(pady=10)

        ttk.Button(input_frame, text="Kiểm tra Spam", command=self.check_spam).pack()

    def create_result_section(self):
        self.result_frame = ttk.Frame(self.root, padding="10")
        self.result_frame.pack(fill=tk.BOTH)
        self.result_label = ttk.Label(self.result_frame, text="")
        self.result_label.pack()

    def create_examples_section(self):
        examples_frame = ttk.LabelFrame(self.root, text="Email Mẫu", padding="10")
        examples_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        scrollable_frame = ScrollableFrame(examples_frame)
        scrollable_frame.pack(fill=tk.BOTH, expand=True)

        example_emails = [
            "Chúc mừng! Bạn đã trúng 1 tỷ đồng từ chương trình khuyến mãi!",
            "Chào bạn, cuộc họp nhóm được lên lịch lúc nào vậy?",
            "KHẨN CẤP: Tài khoản của bạn đã bị xâm phạm. Nhấp vào đây để xác minh.",
            "Đây là tài liệu bạn yêu cầu cho dự án.",
            "HOT: Cơ hội đầu tư với lợi nhuận 500% mỗi tháng!",
            "Báo cáo tiến độ tuần cho thấy đã hoàn thành 80% công việc đã lên kế hoạch.",
            "Kiếm tiền nhanh! Cơ hội làm việc tại nhà!",
            "Nhắc nhở cuộc họp: Cập nhật trạng thái dự án lúc 14:00",
            "Gói hàng của bạn đã được giao",
            "NGƯỜI CHIẾN THẮNG!! Bạn đã được chọn nhận giải thưởng đặc biệt!"
        ]

        for email in example_emails:
            email_frame = ttk.Frame(scrollable_frame.scrollable_frame)
            email_frame.pack(fill=tk.X, pady=2)
            
            ttk.Button(
                email_frame, 
                text="Thử", 
                command=lambda e=email: self.load_example(e)
            ).pack(side=tk.LEFT, padx=(0, 5))
            
            ttk.Label(
                email_frame, 
                text=email, 
                wraplength=500
            ).pack(side=tk.LEFT, fill=tk.X)

    def load_model(self):
        try:
            self.model = load_model('spam_detector_model.h5')
            with open('tokenizer.pickle', 'rb') as handle:
                self.tokenizer = pickle.load(handle)
        except Exception as e:
            print(f"Lỗi khi tải mô hình: {e}")
            self.model = None
            self.tokenizer = None

    def preprocess_text(self, text):
        if pd.isna(text):
            return ""
        text = str(text)
        text = text.lower()
        text = ''.join([char for char in text if char not in string.punctuation])
        text = re.sub('[0-9]+', '', text)
        return text

    def check_spam(self):
        if not self.model or not self.tokenizer:
            self.show_result("Lỗi: Không thể tải mô hình")
            return

        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            self.show_result("Vui lòng nhập nội dung email")
            return

        processed_text = self.preprocess_text(text)
        sequences = self.tokenizer.texts_to_sequences([processed_text])
        padded = pad_sequences(sequences, maxlen=200)
        
        prediction = self.model.predict(padded)[0][0]
        confidence = prediction if prediction >= 0.5 else 1 - prediction
        
        result = "SPAM" if prediction >= 0.5 else "KHÔNG PHẢI SPAM"
        self.show_result(f"Kết quả: {result}\nĐộ tin cậy: {confidence:.2%}")

    def show_result(self, message):
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        ttk.Label(self.result_frame, text=message).pack()

    def load_example(self, example_text):
        self.text_input.delete("1.0", tk.END)
        self.text_input.insert("1.0", example_text)
        self.check_spam()

def main():
    root = tk.Tk()
    app = SpamDetectorGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
