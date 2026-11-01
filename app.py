
import streamlit as st
import torch
from PIL import Image
from transformers import (
    ViTForImageClassification,
    ViTImageProcessor,
    pipeline
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

MODEL_ID = "erum111/plant-leaf-disease-detection"

# Load the image verification model
@st.cache_resource
def load_image_checker():

    device = 0 if torch.cuda.is_available() else -1

    return pipeline(
        "zero-shot-image-classification",
        model="openai/clip-vit-base-patch32",
        device=device
    )

# Check whether the uploaded image is a bean leaf
def is_bean_leaf(image):

    checker = load_image_checker()

    candidates = [
        "a close-up photograph of a bean leaf",
        "a photograph of a bean plant leaf",
        "a photograph of a computer keyboard",
        "a photograph of a mobile phone",
        "a photograph of a person",
        "a photograph of an animal",
        "a photograph of a car",
        "a photograph of food",
        "a photograph of a building",
        "a photograph of an unrelated object"
    ]

    results = checker(
        image,
        candidate_labels=candidates
    )

    leaf_labels = [
        "a close-up photograph of a bean leaf",
        "a photograph of a bean plant leaf"
    ]

    leaf_score = max(
        item["score"]
        for item in results
        if item["label"] in leaf_labels
    )

    non_leaf_score = max(
        item["score"]
        for item in results
        if item["label"] not in leaf_labels
    )

    return (
        leaf_score >= 0.20
        and leaf_score > non_leaf_score + 0.03
    )

# Load the trained ViT model
@st.cache_resource
def load_model():

    processor = ViTImageProcessor.from_pretrained(
        MODEL_ID
    )

    model = ViTForImageClassification.from_pretrained(
        MODEL_ID
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return processor, model, device


# Upload image
uploaded_file = st.file_uploader(
    "Choose a bean leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

        if st.button("Detect Plant Disease"):

            with st.spinner(
                "Checking whether the image contains a bean leaf..."
            ):

                valid_leaf = is_bean_leaf(image)

            if not valid_leaf:

                st.error("❌ Invalid Image!")

                st.warning(
                    "This image does not appear to be a bean leaf. "
                    "Please upload a clear bean leaf photograph."
                )

            else:

                st.success("Bean leaf image detected!")

                try:
                    processor, model, device = load_model()

                    with st.spinner(
                        "Analyzing bean leaf for disease..."
                    ):

                        inputs = processor(
                            images=image,
                            return_tensors="pt"
                        )

                        inputs = {
                            key: value.to(device)
                            for key, value in inputs.items()
                        }

                        with torch.no_grad():

                            outputs = model(**inputs)

                            probabilities = torch.softmax(
                                outputs.logits,
                                dim=1
                            )[0]

                        predicted_class = (
                            probabilities.argmax().item()
                        )

                        disease = model.config.id2label[
                            predicted_class
                        ]

                        confidence = (
                            probabilities[predicted_class].item()
                            * 100
                        )

                    st.subheader("Detection Result")

                    st.success(
                        "Predicted Disease: "
                        + disease.replace("_", " ").title()
                    )

                    st.metric(
                        "Prediction Confidence",
                        f"{confidence:.2f}%"
                    )

                    st.subheader("Class Probabilities")

                    for index, probability in enumerate(
                        probabilities
                    ):

                        label = model.config.id2label[index]

                        st.write(
                            f"{label.replace('_', ' ').title()}: "
                            f"{probability.item() * 100:.2f}%"
                        )

                        st.progress(probability.item())

                    st.info(
                        "This application is an educational screening "
                        "tool. Consult an agricultural expert for "
                        "confirmation."
                    )

                except Exception as e:

                    st.error(
                        f"Disease prediction failed: {e}"
                    )

    except Exception as e:

        st.error(f"Image processing failed: {e}")
