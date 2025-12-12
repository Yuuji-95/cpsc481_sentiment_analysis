import os
import tkinter as tk
from tkinter import ttk, messagebox

import pickle

try:
    import joblib
except ImportError:
    joblib = None


# ---------- Load model + vectorizer ----------

# Base directory is the project root (one level above src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "data", "sentiment_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "data", "tfidf_vectorizer.pkl")


def load_object(path):
    """Try joblib first, then pickle."""
    if joblib is not None:
        try:
            return joblib.load(path)
        except Exception:
            pass  # fall back to pickle

    with open(path, "rb") as f:
        return pickle.load(f)


try:
    model = load_object(MODEL_PATH)
    vectorizer = load_object(VECTORIZER_PATH)
except Exception as e:
    raise RuntimeError(
        f"Could not load model/vectorizer. "
        f"Make sure you ran train_model.py first.\nDetails: {e}"
    )

# ---------- Tkinter GUI ----------

class SentimentApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Movie Review Sentiment Analyzer")
        self.geometry("650x450")
        self.minsize(650, 450)

        # Overall padding
        self.container = ttk.Frame(self, padding=15)
        self.container.pack(fill="both", expand=True)

        # Title
        title_label = ttk.Label(
            self.container,
            text="IMDB Movie Review Sentiment Analysis",
            font=("Segoe UI", 16, "bold")
        )
        title_label.pack(pady=(0, 10))

        # Instructions
        instructions = ttk.Label(
            self.container,
            text="Type or paste a movie review below, then click 'Analyze Sentiment'.",
            font=("Segoe UI", 10)
        )
        instructions.pack(pady=(0, 10))

        # Text box
        self.text_box = tk.Text(self.container, height=10, wrap="word", font=("Segoe UI", 10))
        self.text_box.pack(fill="both", expand=True)

        # Buttons
        button_frame = ttk.Frame(self.container)
        button_frame.pack(pady=10)

        analyze_btn = ttk.Button(button_frame, text="Analyze Sentiment", command=self.analyze_sentiment)
        analyze_btn.grid(row=0, column=0, padx=5)

        clear_btn = ttk.Button(button_frame, text="Clear", command=self.clear_text)
        clear_btn.grid(row=0, column=1, padx=5)

        # Result label
        self.result_var = tk.StringVar(value="Prediction will appear here.")
        self.result_label = ttk.Label(
            self.container,
            textvariable=self.result_var,
            font=("Segoe UI", 12, "bold")
        )
        self.result_label.pack(pady=(5, 0))

    def clear_text(self):
        self.text_box.delete("1.0", tk.END)
        self.result_var.set("Prediction will appear here.")

    def analyze_sentiment(self):
        review = self.text_box.get("1.0", tk.END).strip()

        if not review:
            messagebox.showwarning("No text", "Please enter a movie review first.")
            return

        # Vectorize
        try:
            X = vectorizer.transform([review])
            pred = model.predict(X)[0]

            # If model supports probabilities, show confidence
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                confidence = max(proba)
                confidence_pct = confidence * 100
                self.result_var.set(
                    f"Sentiment: {pred}  (Confidence: {confidence_pct:.1f}%)"
                )
            else:
                self.result_var.set(f"Sentiment: {pred}")
        except Exception as e:
            messagebox.showerror("Error", f"Error making prediction:\n{e}")


if __name__ == "__main__":
    app = SentimentApp()
    app.mainloop()
