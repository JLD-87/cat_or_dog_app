import streamlit as st
import os
import catdog
import numpy as np
import sys
from PIL import Image

image_resize=128
model_file = "model.h5"

st.title("Is it a cat or a dog?")
st.markdown("Upload an image of a cat or a dog, and the model will try to classify it for you after pressing the button!")
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

pressed = st.button("Cat or dog?")

if uploaded_file is not None:
    image_uploaded = Image.open(uploaded_file)
    resized_image = image_uploaded.resize((image_resize, image_resize))
    st.image(image_uploaded, caption=uploaded_file.name)


if pressed:
    #random_number = np.random.randint(1,38)
    #st.image(os.path.join(os.getcwd(),"static", f"{random_number}.jpg"))
    
    try:
        
        model = catdog.load_model(model_file)
        
        # Example: if model expects (image_resize, image_resize) input
        #sample_input = np.random.rand(image_resize, image_resize)  # Replace with real data
        #prepared_data = catdog.prepare_image_data(f"./static/{random_number}.jpg", (image_resize, image_resize))
        prepared_data = catdog.prepare_image_data(resized_image)
        
        # Make prediction
        prediction = model.predict(prepared_data)
        st.caption(f"Prediction output:{prediction}" )
        if prediction[0][0] > 0.7:
            st.caption("The image is classified as a dog.")
        elif prediction[0][0] < 0.3:
            st.caption("The image is classified as a cat.")
        else:
            st.caption("I'm not sure if it is a cat or a dog.")
        
    except Exception as err:
        print(f"Error: {err}", file=sys.stderr)
