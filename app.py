import gradio as gr
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image

# Load the Keras model
model = keras.models.load_model('model.keras')

CLASS_NAMES = [
    'Lionel Messi',
    'Cristiano Ronaldo',
    'Kylian Mbappe',
    'Erling Haaland',
    'Neymar Jr'
]

def predict(image):
    # Preprocess exactly as during training for ResNet50
    img = image.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    # ResNet50 preprocessing: BGR + mean subtraction
    mean = np.array([103.939, 116.779, 123.68])
    img_bgr = img_array[..., ::-1]          # RGB -> BGR
    img_bgr -= mean
    input_data = np.expand_dims(img_bgr, axis=0)

    output = model.predict(input_data, verbose=0)
    pred_idx = int(np.argmax(output[0]))
    confidence = float(np.max(output[0]))
    return {CLASS_NAMES[i]: float(output[0][i]) for i in range(len(CLASS_NAMES))}

iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='pil', label="Upload a player's photo"),
    outputs=gr.Label(num_top_classes=4),
    title="⚽ Football Player Face Recognition",
    description="Upload a photo of Messi, Ronaldo, Mbappé or Neymar."
)

iface.launch(server_name="0.0.0.0", server_port=7860)
