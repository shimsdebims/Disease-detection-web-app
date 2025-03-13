from flask import Flask, render_template, request, redirect, url_for
import os
import torch
from torchvision import models, transforms
from PIL import Image
import json

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'Uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained PyTorch model
MODEL_PATH = 'Model/crop_disease_model.pth'

# Define the model architecture
model = models.resnet18(pretrained=False)  # Use pretrained=False since we're loading custom weights
num_classes = 15  # Adjust based on your dataset
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # Adjust the final layer

# Load the state dictionary (weights) into the model
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

# Class labels (update based on your dataset)
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

# Load disease information from JSON
with open('disease_info.json', 'r') as f:
    DISEASE_INFO = json.load(f)

# Preprocess image for model input
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0)
    return img

# Predict disease
def predict_disease(img_path):
    img = preprocess_image(img_path)
    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)
        predicted_class = predicted.item()
    return CLASS_LABELS[predicted_class], DISEASE_INFO.get(CLASS_LABELS[predicted_class], {})

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        prediction, disease_info = predict_disease(file_path)
        return render_template('results.html', prediction=prediction, disease_info=disease_info, image_url=file_path)

if __name__ == '__main__':
    app.run(debug=True)