
import streamlit as st
import torch
from PIL import Image
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor
)

# Page configuration
st.set_page_config(
    page_title="Plant Leaf Disease Detection",
    page_icon="🌱",
    layout="centered"
)

# Application heading
st.title("🌱 Plant Leaf Disease Detection")

st.write(
    "Upload a bean leaf image to detect "
    "Angular Leaf Spot, Bean Rust, or Healthy leaves."
)

# Hugging Face model repository
MODEL_ID = "erum111/plant-leaf-disease-detection"

# Load model and processor
@st.cache_resource
def load_model():

    processor = ViTImageProcessor.from_pretrained(MODEL_ID)

    model = ViTForImageClassification.from_pretrained(MODEL_ID)

    model.eval()

    return processor, model


try:
    processor, model = load_model()
    st.success("AI model loaded successfully!")

except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


# Upload image
uploaded_file = st.file_uploader(
    "Choose a bean leaf image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Bean Leaf",
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

            disease = model.config.id2label[predicted_class]

            confidence = probabilities[predicted_class].item() * 100

        st.subheader("Detection Result")

        st.success(
            f"Predicted Disease: {disease.replace('_', ' ').title()}"
        )

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        st.subheader("Class Probabilities")

        for index, probability in enumerate(probabilities):

            label = model.config.id2label[index]

            st.write(
                f"{label.replace('_', ' ').title()}: "
                f"{probability.item() * 100:.2f}%"
            )

            st.progress(probability.item())

        st.info(
            "This application is an educational screening tool. "
            "Consult an agricultural expert for confirmation."
        )

