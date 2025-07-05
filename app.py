import streamlit as st
import pandas as pd
import os
from PIL import Image
import cv2
import torch
from datetime import datetime
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import tempfile


# YOLO and char segmentation model dependencies
from ultralytics import YOLO
from torchvision import transforms
from utils import segment_characters, show_results, crop_image, detect_Np  # replace with your actual functions

# Set Streamlit page config
st.set_page_config(page_title="License Plate Lookup", layout="centered")

# --- Paths ---
LOG_PATH = r'E:\computer vision\PA2_CNN1\PA2\newversion\license_plate_log.csv'
IMAGE_DIR = r'E:\computer vision\PA2_CNN1\PA2\newversion\Images'
YOLO_MODEL_PATH = r'E:\computer vision\PA2_CNN1\PA2\best.torchscript'

######################################
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
    transforms.Resize((28, 28)),  # Smaller size = faster training
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset_path = r'E:\computer vision\PA2_CNN1\PA2\CNN_lp_dataset\train'
train_dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
idx_to_class = {v: k for k, v in train_dataset.class_to_idx.items()}
######################################


######################################
class CharCNN(nn.Module):
    def __init__(self, num_classes=36):  # 0-9 and A-Z = 36 classes
        super(CharCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # Input = 1 channel
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # [1,28,28] → [32,14,14]
        x = self.pool(F.relu(self.conv2(x)))  # [32,14,14] → [64,7,7]
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
model = CharCNN()
model.load_state_dict(torch.load(r'E:\computer vision\PA2_CNN1\PA2\char_cnn.pth'))
#######################################

# --- Load the CSV ---
@st.cache_data
def load_log():
    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH, names=["Plate Number", "Image Label", "Timestamp"])
        return df
    else:
        return pd.DataFrame(columns=["Plate Number", "Image Label", "Timestamp"])

df = load_log()

# --- Load detection model ---
@st.cache_resource
def load_models():
    yolo_model = YOLO(YOLO_MODEL_PATH, task = 'detect')
    characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    return yolo_model, characters

yolo_model, characters = load_models()

st.title("🚓 License Plate Lookup - Safe City Surveillance")

# Tabs for functionality
tab1, tab2 = st.tabs(["🔍 Search by Plate Number", "📤 Upload Image"])

# --- Tab 1: Search by Plate ---
with tab1:
    plate_query = st.text_input("Enter License Plate Number:")

    if plate_query:
        filtered = df[df['Plate Number'].str.contains(plate_query.upper(), na=False)]

        if not filtered.empty:
            st.success(f"Found {len(filtered)} result(s):")

            for _, row in filtered.iterrows():
                st.markdown(f"**Plate Number:** {row['Plate Number']}")
                st.markdown(f"**Timestamp:** {row['Timestamp']}")
                img_path = os.path.join(IMAGE_DIR, row['Image Label'])

                if os.path.exists(img_path):
                    st.image(Image.open(img_path), caption=row['Image Label'], width=300)
                else:
                    st.warning(f"Image not found: {img_path}")
                st.markdown("---")
        else:
            st.error("No matches found.")

# --- Tab 2: Upload Image ---
with tab2:
    uploaded_file = st.file_uploader("Upload an image of a car", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", width=400)

        # Convert to OpenCV image
        file_bytes = uploaded_file.read()
        npimg = np.frombuffer(file_bytes, np.uint8)
        uploaded_img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

        try:
            # ✅ Save temporarily to pass file path to detect_Np
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
                cv2.imwrite(temp_file.name, uploaded_img)
                temp_path = temp_file.name

            # Call your path-based function
            detected_img, x1, y1, x2, y2 = detect_Np(temp_path)
            cropped_img = crop_image(detected_img, int(x1), int(x2), int(y1), int(y2))

            # Segment and recognize
            char_imgs = segment_characters(cropped_img)
            predicted_plate = show_results(model, char_imgs, idx_to_class, device='cpu')

            st.markdown(f"### 🔍 Detected Plate: `{predicted_plate}`")

            # Match to CSV
            match = df[df['Plate Number'] == predicted_plate]

            if not match.empty:
                for _, row in match.iterrows():
                    st.markdown(f"**Timestamp:** {row['Timestamp']}")

                    # ✅ Display the image with bounding box (from detect_Np)
                    # Convert from BGR to RGB
                    detected_img_rgb = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
                    st.image(detected_img_rgb, caption="Detected Plate (with box)", width=400)

            else:
                st.warning("No matching plate found in log.")

        except Exception as e:
            st.error(f"Failed to detect license plate: {e}")

# run streamlit run app.py