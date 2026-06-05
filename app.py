import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os
# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="MyDoc AI", layout="centered")

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>
body {
    background-color: #f4f8fb;
}
.main-title {
    text-align: center;
    font-size: 34px;
    font-weight: 700;
    color: #1f4e79;
    margin-bottom: 10px;
}
.subtitle {
    text-align: center;
    color: #4a4a4a;
    margin-bottom: 30px;
}
.card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.footer {
    text-align:center;
    margin-top:40px;
    color:gray;
    font-size:14px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.markdown('<div class="main-title">MyDoc: AI-Powered Chest X-ray Diagnostic System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Deep Learning Based Pneumonia Detection with Grad-CAM Visualization</div>', unsafe_allow_html=True)

# ------------------ SIDEBAR ------------------
page = st.sidebar.radio("Navigation", ["Home", "About Pneumonia", "Contact"])

# ------------------ LOAD MODEL ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, 1)
    model.load_state_dict(torch.load("mydoc_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()
target_layer = model.features[-1]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
# ------------------ HOME PAGE ------------------
if page == "Home":

    st.markdown('<div class="card">', unsafe_allow_html=True)
# ---------------- UPLOAD FIRST ----------------
    st.subheader("Upload Chest X-ray")

    uploaded_file = st.file_uploader(
        "Choose an X-ray image",
        type=["jpg", "jpeg", "png"]
    )

# ---------------- DEMO SECTION ----------------
    st.divider()

    st.subheader("Quick Demo")

    st.info(
        "Don't have a chest X-ray? Try one of our sample images or download them."
    )

    selected_image = None

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            "demo_normal.png",
            caption="Normal Sample",
            use_container_width=True
        )

        if st.button("Use Normal Demo"):
            selected_image = Image.open(
                "demo_normal.png"
            ).convert("RGB")

        with open("demo_normal.png", "rb") as file:
            st.download_button(
                "⬇ Download",
                file,
                "demo_normal.png",
                key="normal_download"
            )

    with col2:

        st.image(
            "demo_pneumonia.jpg",
            caption="Pneumonia Sample",
            use_container_width=True
        )

        if st.button("Use Pneumonia Demo"):
            selected_image = Image.open(
                "demo_pneumonia.jpg"
            ).convert("RGB")

        with open("demo_pneumonia.jpg", "rb") as file:
            st.download_button(
                "⬇ Download",
                file,
                "demo_pneumonia.jpg",
                key="pneumonia_download"
            )

# ---------------- IMAGE SELECTION ----------------
    image = None

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

    elif selected_image is not None:
        image = selected_image
# ---------------- PREDICTION ----------------
    if image is not None:

        st.divider()

        st.image(
            image,
            caption="Selected Image",
            width=400
        )

# ------------------ ABOUT PAGE ------------------
elif page == "About Pneumonia":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("What is Pneumonia?")
    st.write("""
Pneumonia is a lung infection that causes inflammation in the air sacs (alveoli).
The air sacs may fill with fluid or pus, causing cough, fever, chills, and difficulty breathing.
    """)

    st.subheader("Common Symptoms")
    st.write("""
• Chest pain while breathing  
• Persistent cough  
• Fever & chills  
• Fatigue  
• Shortness of breath  
    """)

    st.subheader("Why Early Detection Matters")
    st.write("""
Early diagnosis helps prevent complications and reduces mortality risk,
especially in elderly and immunocompromised patients.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ CONTACT PAGE ------------------
elif page == "Contact":

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Contact Developer")
    st.write("Developed by: **Srishti Bhatt**")
    st.write("Email: srishtibhatt100@gmail.com")
    st.write("Project: AI-powered medical image diagnostic prototype")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ FOOTER ------------------
st.markdown('<div class="footer">© 2026 MyDoc AI | Educational Prototype</div>', unsafe_allow_html=True)
