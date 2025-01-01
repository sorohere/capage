import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2
import streamlit as st
from construct.utils import set_cuda, image_transformation, plot_attention
from construct.architecture import Encoder_Decoder_Model, Vocabulary, Image_encoder, Attention_Based_Decoder, AttentionLayer
import numpy as np
from PIL import Image

# Set the device for computation
device = set_cuda()

# Load the pre-trained model and vocabulary
model = torch.load('model/model.pt', map_location=device)
vocab = torch.load("model/vocab.pth", map_location=device)

# Configure Streamlit app with a dark theme and page layout
st.set_page_config(
    page_title="capage",
    layout="centered",  # Center content for better aesthetics
    initial_sidebar_state="expanded"
)

# CSS styles for dark theme and improved visuals
st.markdown("""
    <style>
        body {
            background-color: #121212;
            color: #E0E0E0;
        }
        .title {
            text-align: center;
            font-size: 36px;
            color: #BB86FC;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .header {
            text-align: center;
            font-size: 18px;
            color: #CF6679;
            margin-bottom: 20px;
        }
        .stButton button {
            background-color: #03DAC6;
            color: #000;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 8px;
        }
        .stButton button:hover {
            background-color: #018786;
            color: #FFF;
        }
        .matrix-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 20px;
        }
        .textbox {
            width: 480px;
            height: 100px;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #BB86FC;
            font-size: 16px;
            background-color: #1F1B24;
            color: #E0E0E0;
        }
    </style>
""", unsafe_allow_html=True)

# Display title and instructions
st.markdown('<div class="title">capage: captionimage</div>', unsafe_allow_html=True)
st.markdown('<div class="header">Upload an image and watch the captioning model work its magic!</div>', unsafe_allow_html=True)

# Upload image section
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Load and preprocess the uploaded image
    image = Image.open(uploaded_image)
    image = np.array(image)
    image = cv2.resize(image, (480, int(480 * image.shape[0] / image.shape[1])))  # Adjust height proportionally

    # Arrange the image and caption in a 1x2 matrix layout
    st.markdown('<div class="matrix-container">', unsafe_allow_html=True)

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", width=480)

    # Button to generate captions
    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            # Generate caption and attention maps
            attentions, caption = model.predict(image, vocab)
            caption_text = ' '.join(caption[1:-1])

            # Display the generated caption in a text box
            st.markdown(f'<textarea class="textbox" readonly>{caption_text}</textarea>', unsafe_allow_html=True)

            # Option to display attention maps
            if st.checkbox("Show Attention Maps"):
                st.write("### Attention Maps")
                try:
                    # Create a placeholder for attention maps
                    attention_placeholder = st.empty()
                    with attention_placeholder:
                        plot_attention(image, caption, attentions, is_streamlit=True)
                except Exception as e:
                    st.error(f"Error displaying attention maps: {str(e)}")
                    st.error("Attention vectors shape: " + str([att.shape for att in attentions]))
                    st.error("Caption length: " + str(len(caption)))

    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Upload an image to get started!")
