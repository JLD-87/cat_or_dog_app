import streamlit as st
import catdog
import sys
from PIL import Image

image_resize=128
model_file = "model.h5"
model = catdog.load_model(model_file)

#Sidebar layout
st.sidebar.title ("About this app")
st.sidebar.markdown("""
This is a simple image classification app that uses a pre-trained model to classify images as either cats or dogs.

The model was trained on a dataset of  25k images of cats and dogs, and it can predict the class of an uploaded image with a certain level of confidence. 

Please upload an image of a cat or a dog, and click the button to see the prediction results.
""")

#Frontend : title, description, file uploader, button, 
# then image display, prediction display

st.title("Is it a cat or a dog?")
st.markdown("Upload an image of a cat or a dog, and the model will try to classify it for you after pressing the button!")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
pressed = st.button("Cat or dog?")

#load the image to screen if uploaded, save it in this session as resized_image
if uploaded_file is not None:
    image_uploaded = Image.open(uploaded_file)
    resized_image = image_uploaded.resize((image_resize, image_resize)) #model expects 128*128 images
    st.image(image_uploaded, caption=uploaded_file.name)

if pressed:
    try:
        #prepare the uploaded image
        prepared_data = catdog.prepare_image_data(resized_image)
        
        # Make prediction and display results
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
