import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tkinter as tk
from tkinter import ttk, scrolledtext
import pickle
import re
import string

def preprocess_text(text):    # Convert text to lowercase
    text = text.lower()
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    # Remove numbers
    text = re.sub('[0-9]+', '', text)
    return text

def predict_spam(model, text, tokenizer, max_length=200):    # Preprocess the text
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
import pickle
try:
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
except:
    print("Could not load saved tokenizer, using new one")
    tokenizer = Tokenizer()

class SpamDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Spam Email Detector")
        self.root.geometry("600x700")

        # Style
        style = ttk.Style()
        style.configure('TButton', padding=5)
        style.configure('TLabel', padding=5)

        # Input area
        input_frame = ttk.Frame(root, padding="10")
        input_frame.pack(fill=tk.X)
        
        ttk.Label(input_frame, text="Enter email text:").pack()
        
        self.text_input = scrolledtext.ScrolledText(input_frame, height=10, width=60)
        self.text_input.pack(pady=10)

        ttk.Button(input_frame, text="Check Spam", command=self.check_spam).pack()

        # Results area
        self.result_frame = ttk.Frame(root, padding="10")
        self.result_frame.pack(fill=tk.BOTH, expand=True)

        # Example emails
        examples_frame = ttk.LabelFrame(root, text="Example Emails", padding="10")
        examples_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        example_emails = [
            "Congratulations! You've won $1 million in our promotion!",
            "Hi, when is our team meeting scheduled?",
            "URGENT: Your account has been compromised. Click here to verify.",
            "Here are the documents you requested for the project.",
            "HOT: Investment opportunity with 500% monthly returns!",
            "Weekly progress report shows 80% completion of planned tasks."
        ]

        for email in example_emails:
            ttk.Button(examples_frame, text="Try", 
                      command=lambda e=email: self.load_example(e)).pack(anchor='w')
            ttk.Label(examples_frame, text=email, wraplength=500).pack(anchor='w', pady=2)

    def load_example(self, email):
        self.text_input.delete(1.0, tk.END)
        self.text_input.insert(tk.END, email)

    def check_spam(self):
        email = self.text_input.get(1.0, tk.END).strip()
        if not email:
            return

        # Clear previous results
        for widget in self.result_frame.winfo_children():
            widget.destroy()

        result = predict_spam(model, email, tokenizer)
        
        # Create result display
        if result['is_spam']:
            result_text = "SPAM DETECTED!"
            color = "#ff4444"
        else:
            result_text = "NOT SPAM"
            color = "#44aa44"

        ttk.Label(self.result_frame, 
                 text=result_text,
                 font=('Helvetica', 16, 'bold')).pack(pady=10)

        confidence_frame = ttk.Frame(self.result_frame)
        confidence_frame.pack(fill=tk.X, pady=5)
        
        confidence_label = ttk.Label(confidence_frame, 
                                   text=f"Confidence: {result['confidence']:.1%}")
        confidence_label.pack()

        # Create confidence bar
        canvas = tk.Canvas(confidence_frame, height=20, width=200)
        canvas.pack(pady=5)
        canvas.create_rectangle(0, 0, 200, 20, fill='#dddddd')
        canvas.create_rectangle(0, 0, 200 * result['confidence'], 20, fill=color)

# Initialize the model and tokenizer
model = load_model('spam_detector_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Create and run GUI
root = tk.Tk()
app = SpamDetectorGUI(root)
root.mainloop()
