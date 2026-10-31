
import streamlit as st
import torch
from PIL import Image
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor
)

st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Plant Leaf Disease Detection")

st.write(
    "Upload a bean leaf image to identify whether "
    "it has Angular Leaf Spot, Bean Rust, or is Healthy."
)

MODEL_ID = "erum111/plant-leaf-disease-detection"

@st.cache_resource
def load_model():
    processor = ViTImageProcessor.from_pretrained(MODEL_ID)

    model = ViTForImageClassification.from_pretrained(
        MODEL_ID
    )

    model.eval()

    return processor, model

try:
    processor, model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a bean leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded leaf image",
        use_container_width=True
    )

    if st.button("Detect Plant Disease"):

        with st.spinner("Analyzing leaf image..."):

            inputs = processor(
                images=image,
                return_tensors="pt"
            )

            with torch.no_grad():
                outputs = model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=1
            )[0]

            predicted_class = probabilities.argmax().item()

            disease = model.config.id2label[
                predicted_class
            ]

            confidence = probabilities[
                predicted_class
            ].item() * 100

        st.subheader("Prediction Result")

        st.success(
            f"Predicted class: {disease.replace('_', ' ').title()}"
        )

        st.metric(
            "Prediction confidence",
            f"{confidence:.2f}%"
        )

        st.write("Class probabilities:")

        for index, probability in enumerate(probabilities):
            label = model.config.id2label[index]

            st.write(
                f"{label.replace('_', ' ').title()}: "
                f"{probability.item() * 100:.2f}%"
            )

        st.info(
            "This tool is an educational screening aid. "
            "Confirm suspected disease with an agricultural expert."
        )
