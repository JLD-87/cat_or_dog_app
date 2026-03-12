import os
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np

#load the prediction model
def load_model(model_path):
    # verify if file is found, and if it is a .h5 file
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    if not model_path.lower().endswith(".h5"):
        raise ValueError("The model doesn't have h5 extension.")
    
    # load the model
    try:
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


#image is taken from streamlit file uploader, only usable in a single session. It is formatted here for use in the model.
def prepare_image_data(image_loaded):
    # Load the image and resize it
    img_array = image.img_to_array(image_loaded)

    # Normalize pixel values to [0, 1]
    img_array /= 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array