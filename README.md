
# 🌱 Plant Leaf Disease Detection

## Project Description
An AI-powered application that identifies bean leaf diseases using a fine-tuned Vision Transformer (ViT).

## Disease Classes
- Angular Leaf Spot
- Bean Rust
- Healthy

## Model
google/vit-base-patch16-224-in21k

Fine-tuned model:
https://huggingface.co/erum111/plant-leaf-disease-detection

## Dataset
Beans plant disease dataset from Hugging Face.

## Test Accuracy
92.19%

## Technologies
- Python
- PyTorch
- Hugging Face Transformers
- Streamlit
- Vision Transformer

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Disclaimer
This application is an educational screening tool and does not replace professional agricultural diagnosis.
