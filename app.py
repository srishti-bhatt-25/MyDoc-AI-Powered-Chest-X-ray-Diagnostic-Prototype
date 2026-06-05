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

# ---------------- UPLOAD FIRST ----------------
    st.subheader("Upload Chest X-ray")
    uploaded_file = st.file_uploader(
    "Choose an X-ray image",
    type=["jpg", "jpeg", "png"]
    )
    analyze_btn = st.button(
    "Analyze X-ray",
    type="primary"
    )
    if uploaded_file is not None and analyze_btn:
        st.success( "Analysis started. Please scroll down to view the report." )

# ---------------- DEMO SECTION ----------------
    st.subheader("Quick Demo")
    st.info(
        "Don't have a chest X-ray? Download our Sample Image"
    )

    selected_image = None

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            "demo_normal.jpg",
            caption="Sample X-ray",
            use_container_width=True
        )

        with open("demo_normal.jpg", "rb") as file:
            st.download_button(
                "Download",
                file,
                "demo_normal.jpg",
                key="normal_download"
            )

    
# ---------------- IMAGE SELECTION ----------------
    image = None

    if uploaded_file is not None and analyze_btn:
        image = Image.open(uploaded_file).convert("RGB")

    elif selected_image is not None:
        image = selected_image
# ---------------- PREDICTION ----------------
    if image is not None:
        with st.spinner("Analyzing X-ray..."):
            input_tensor = transform(image).unsqueeze(0).to(device)
            gradients = []
            activations = []

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])

        def forward_hook(module, input, output):
            activations.append(output)

        handle_fw = target_layer.register_forward_hook(forward_hook)
        handle_bw = target_layer.register_backward_hook(backward_hook)

        output = model(input_tensor)
        probability = torch.sigmoid(output).item()

        model.zero_grad()
        output.backward()

        grads = gradients[0]
        acts = activations[0]

        weights = torch.mean(grads, dim=[2, 3], keepdim=True)

        grad_cam = torch.sum(
            weights * acts,
            dim=1
        ).squeeze()

        grad_cam = F.relu(grad_cam)
        grad_cam = grad_cam.cpu().detach().numpy()

        grad_cam = (
            grad_cam - grad_cam.min()
        ) / (
            grad_cam.max() - grad_cam.min() + 1e-8
        )

        heatmap = cv2.resize(
            grad_cam,
            (224, 224)
        )

        heatmap = np.uint8(255 * heatmap)

        heatmap = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        original = cv2.resize(
            np.array(image),
            (224, 224)
        )

        superimposed = cv2.addWeighted(
            original,
            0.6,
            heatmap,
            0.4,
            0
        )

        st.warning(
            "Educational Prototype Only. "
            "This prediction is not a medical diagnosis."
        )
        st.subheader("Prediction Result")

        if probability > 0.5:

            st.error(
                f"Pneumonia Detected "
                f"(Confidence: {probability * 100:.2f}%)"
            )

        else:

            st.success(
                f"Normal "
                f"(Confidence: {(1 - probability) * 100:.2f}%)"
            )

        st.subheader("Model Attention (Grad-CAM)")

        st.image(
            superimposed,
            width=400
        )

        handle_fw.remove()
        handle_bw.remove()

# ------------------ ABOUT PAGE ------------------
elif page == "About Pneumonia":
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
    st.subheader("Contact Developer")
    st.write("Developed by: **Srishti Bhatt**")
    st.write("Email: srishtibhatt100@gmail.com")
    st.write("Project: AI-powered medical image diagnostic prototype")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ FOOTER ------------------
st.markdown('<div class="footer">© 2026 MyDoc AI | Educational Prototype</div>', unsafe_allow_html=True)
