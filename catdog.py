import os
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import sys

def load_model(model_path):
    # verify is file is found, and if it is a .h5 file
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


image_resize=128
def prepare_image_data_with_path(image_path, target_size=(image_resize,image_resize)):
    # Load the image and resize it
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)

    # Normalize pixel values to [0, 1]
    img_array /= 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def prepare_image_data(image_loaded):
    # Load the image and resize it
    img_array = image.img_to_array(image_loaded)

    # Normalize pixel values to [0, 1]
    img_array /= 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

if __name__ == "__main__":
    model_file = "model.h5"  # Replace with your model path
    
    try:
        model = load_model(model_file)
        
        # Example: if model expects (image_resize, image_resize) input
        sample_input = np.random.rand(image_resize, image_resize)  # Replace with real data
        prepared_data = prepare_image_data("./images/testdog.jpg", (image_resize, image_resize))
        
        # Make prediction
        prediction = model.predict(prepared_data)
        print("Prediction output:", prediction)
        if prediction[0][0] > 0.5:
            print("The image is classified as a dog.")
        else:
            print("The image is classified as a cat.")
        
    except Exception as err:
        print(f"Error: {err}", file=sys.stderr)