import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import json
import os

# Set page config
st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the pre-trained model
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    num_classes = 15
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load('Model/crop_disease_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Class labels
CLASS_LABELS = {
    0: 'Tomato - Healthy',
    1: 'Tomato - Leaf Mold',
    2: 'Tomato - Yellow Leaf Curl Virus',
    3: 'Tomato - Septoria Leaf Spot',
    4: 'Potato - Healthy',
    5: 'Potato - Late Blight',
    6: 'Potato - Early Blight',
    7: 'Corn - Healthy',
    8: 'Corn - Northern Leaf Blight',
    9: 'Corn - Common Rust',
    10: 'Corn - Gray Leaf Spot',
    11: 'Rice - Healthy',
    12: 'Rice - Blast',
    13: 'Rice - Bacterial Leaf Blight',
    14: 'Rice - Brown Spot'
}

# Load disease info
with open('disease_info.json', 'r') as f:
    DISEASE_INFO = json.load(f)

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = image.convert('RGB')
    img = transform(img).unsqueeze(0)
    return img

# Prediction function
def predict_disease(image):
    img = preprocess_image(image)
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
        predicted_class = predicted.item()
    return CLASS_LABELS[predicted_class], DISEASE_INFO.get(CLASS_LABELS[predicted_class], {})

# Main app
def main():
    st.title("🌱 Crop Disease Detection")
    st.markdown("Upload an image of a plant leaf to detect potential diseases and get treatment recommendations.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        if st.button('Analyze'):
            with st.spinner('Analyzing the image...'):
                prediction, disease_info = predict_disease(image)
                
                st.success("Analysis Complete!")
                
                # Display results in an appealing layout
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.subheader("Prediction Result")
                    st.markdown(f"**{prediction}**")
                    st.image(image, use_column_width=True)
                
                with col2:
                    st.subheader("Disease Information")
                    
                    # Use expanders for each section
                    with st.expander("Symptoms", expanded=True):
                        st.info(disease_info['symptoms'])
                    
                    with st.expander("Treatment Recommendations"):
                        st.warning(disease_info['treatment'])
                    
                    with st.expander("Did You Know?"):
                        st.success(disease_info['fun_fact'])

if __name__ == '__main__':
    main()