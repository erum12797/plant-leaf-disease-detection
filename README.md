
# Plant Leaf Disease Detection 🌱

## Project Description
An AI-powered application that detects bean leaf diseases using a fine-tuned Vision Transformer (ViT).

## Disease Classes
- Angular Leaf Spot
- Bean Rust
- Healthy

## Dataset
Beans Plant Disease Dataset.

Training: 1034 images  
Validation: 133 images  
Testing: 128 images

## Pretrained Model
google/vit-base-patch16-224-in21k

## Fine-tuned Model
https://huggingface.co/erum111/plant-leaf-disease-detection

## Model Performance
Test Accuracy: 92.19%

## Technologies Used
- Python
- PyTorch
- Hugging Face Transformers
- Vision Transformer
- Streamlit

## Run Locally

Install dependencies:

pip install -r requirements.txt

Run application:

streamlit run app.py

## Disclaimer
This application is an educational screening tool. 
Consult an agricultural expert for disease confirmation.





