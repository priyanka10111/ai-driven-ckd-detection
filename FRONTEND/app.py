import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, redirect, url_for, flash, request, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from torchvision import transforms
from PIL import Image

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Image transformations
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the model class (same as the one used during training)
class MobileNetModel(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetModel, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        num_features = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.mobilenet(x)

# Load the trained model
model = MobileNetModel(num_classes=2)
model.load_state_dict(torch.load("mobilenet_irrelevent.pt"))
model = model.to(device)
model.eval()


def predict_image(image_path):
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = image_transform(image).unsqueeze(0)  # Add batch dimension
    image = image.to(device)

    # Perform the prediction
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()

# Helper function to map the prediction to label
def map_prediction_to_label(prediction):
    label_mapping = {0: "irrelevent", 1: "relevent"}
    return label_mapping.get(prediction, "Unknown")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.String(1), nullable=False)
    mobile = db.Column(db.String(15), nullable=False)

# Define the MobileNet + LSTM model as used before
class MobileNetLSTMModel(nn.Module):
    def __init__(self, num_classes, hidden_size=512, lstm_layers=2):
        super(MobileNetLSTMModel, self).__init__()
        # Use MobileNetV2 as a feature extractor
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        
        # Remove the final classification layer of MobileNetV2 and use it as a feature extractor
        self.mobilenet.classifier = nn.Sequential(*list(self.mobilenet.classifier.children())[:-1])
        
        # LSTM to model sequential dependencies over features from MobileNet
        self.lstm = nn.LSTM(input_size=1280, hidden_size=hidden_size, num_layers=lstm_layers, batch_first=True)
        
        # Fully connected layer to output the final classification
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # Get features from MobileNetV2
        features = self.mobilenet(x)
        
        # LSTM expects input of shape (batch_size, sequence_length, feature_size)
        features = features.unsqueeze(1)  # Add a dimension for sequence length (1 in this case)
        
        # Pass the features through the LSTM
        lstm_out, (hn, cn) = self.lstm(features)
        
        # Get the output of the last time step (the last hidden state)
        out = self.fc(hn[-1])
        return out
    

# Load the trained model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_prediction = MobileNetLSTMModel(num_classes=5).to(device)  # Set num_classes appropriately
model_prediction.load_state_dict(torch.load("mobilenet_lstm_model.pt", map_location=torch.device('cpu')))
model_prediction.eval()

class_names = ['G4', 'G1', 'G2', 'G3', 'Normal'] 


# Preprocess the image for prediction
def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # Add batch dimension
    return image


# Predict a single image
def predict_image_model(image_path):
    image = preprocess_image(image_path).to(device)
    with torch.no_grad():
        output = model_prediction(image)
    _, predicted = torch.max(output, 1)
    return predicted.item()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')

        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            session['user_id'] = user.id  # Store user ID in session
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password.', 'danger')
    return render_template('auth.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        age = request.form.get('age')
        gender = request.form.get('gender')
        mobile = request.form.get('mobile')
        
        if len(mobile) != 10 or not mobile.isdigit():
            flash('Mobile number must be exactly 10 digits.', 'danger')
            return render_template('auth.html')

        if User.query.filter_by(email=email).first():
            flash('Email address already in use. Please choose a different one.', 'danger')
            return render_template('auth.html')
        
        if User.query.filter_by(username=username).first():
            flash('Username is already taken. Please choose a different one.', 'danger')
            return render_template('auth.html')

        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return render_template('auth.html')
        
        if len(password) < 8:
            flash('Password must be at least 8 characters long.', 'danger')
            return render_template('auth.html')

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        new_user = User(username=username, email=email, password=hashed_password, age=age, gender=gender, mobile=mobile)
        
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('auth.html')

@app.route('/home')
def home():
    return render_template('home.html')

# @app.route('/prediction', methods=['GET', 'POST'])
# def prediction():
#     predicted_class = None  # Initialize the predicted_class variable

#     if request.method == 'POST':
#         if 'file' not in request.files:
#             flash('No file part', 'danger')
#             return redirect(request.url)

#         myfile = request.files['file']

#         if myfile.filename == '':
#             flash('No selected file', 'danger')
#             return redirect(request.url)
        

#         fn = myfile.filename
#         mypath = os.path.join(r'static/saved_images', fn)
#         myfile.save(mypath)

#         # Predict image relevance
#         prediction = predict_image(mypath)
#         print(11222222222222222,prediction)
#         predicted_label = map_prediction_to_label(prediction)
#         print(11111111111111111111,predicted_label)

#         if predicted_label == "relevent":
#             # Pass the image to the RandomForestClassifier for deforestation/forest prediction
#             predicted_class_index = predict_image_model(mypath)
#             predicted_class = class_names[predicted_class_index]   
#             # Print the predicted class index and name
#             print(f"Predicted class index: {predicted_class_index}")
#             print(f"Predicted class name: {predicted_class}")

#         else:
#             predicted_class = "Not relevant"

#     # Pass the predicted_class and image file name to the template
#     return render_template('prediction.html', predicted_class=predicted_class, image_path=fn if 'fn' in locals() else None)

from PIL import Image
import os

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_tif_to_jpg(tif_image_path):
    with Image.open(tif_image_path) as img:
        img.convert("RGB").save(tif_image_path.replace(".tif", ".jpg"), "JPEG")

@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    predicted_class = None 

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)

        myfile = request.files['file']

        if myfile.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)

        if not allowed_file(myfile.filename):
            flash('Invalid file type. Allowed types: png, jpg, jpeg, tif, tiff.', 'danger')
            return redirect(request.url)
        fn = myfile.filename
        mypath = os.path.join(r'static/saved_images', fn)
        myfile.save(mypath)
        if fn.lower().endswith(('.tif', '.tiff')):
            jpg_image_path = mypath.replace(".tif", ".jpg").replace(".tiff", ".jpg")
            convert_tif_to_jpg(mypath)
            fn = jpg_image_path.split(os.sep)[-1]
        prediction = predict_image(mypath)
        predicted_label = map_prediction_to_label(prediction)

        

        if predicted_label == "relevent":
            predicted_class_index = predict_image_model(mypath)
            predicted_class = class_names[predicted_class_index]
        else:
            predicted_class = "Not relevant"

        

    return render_template('prediction.html', predicted_class=predicted_class, image_path=fn if 'fn' in locals() else None)




if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create the database tables
    app.run(debug=True)