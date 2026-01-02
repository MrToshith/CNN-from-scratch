import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
import os


# =========================
# LOAD MODEL (robust path)
# =========================
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "models", "cnn_model.keras")
model = tf.keras.models.load_model(model_path)


# =========================
# APP CONFIG
# =========================
CANVAS_SIZE = 280
BRUSH_SIZE = 15


class DigitApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognizer")

        self.canvas = tk.Canvas(
            root,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="black"
        )
        self.canvas.pack()

        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

        btn_frame = tk.Frame(root)
        btn_frame.pack()

        tk.Button(btn_frame, text="Predict", command=self.predict).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Clear", command=self.clear).pack(side=tk.LEFT)

        self.result_label = tk.Label(root, text="", font=("Arial", 16))
        self.result_label.pack()

    def paint(self, event):
        x1 = event.x - BRUSH_SIZE
        y1 = event.y - BRUSH_SIZE
        x2 = event.x + BRUSH_SIZE
        y2 = event.y + BRUSH_SIZE

        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        self.draw.ellipse([x1, y1, x2, y2], fill=255)

    def clear(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (CANVAS_SIZE, CANVAS_SIZE), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="")

    def predict(self):
        img = self.image.resize((28, 28))
        img = np.array(img) / 255.0
        img = img.reshape(1, 28, 28, 1)

        pred = model.predict(img, verbose=0)[0]
        digit = np.argmax(pred)
        confidence = np.max(pred) * 100

        self.result_label.config(
            text=f"Predicted: {digit}  |  Confidence: {confidence:.2f}%"
        )


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitApp(root)
    root.mainloop()
