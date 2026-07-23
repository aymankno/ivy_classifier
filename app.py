import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import ssl
import certifi
import urllib.request

ssl_context = ssl.create_default_context(cafile=certifi.where())
opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
urllib.request.install_opener(opener)

model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(512, 2)
model.load_state_dict(torch.load('ivy_classifier.pth', map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

st.title("Is it Ivy??")
st.write(f"Upload a photo and find out if it's her! 😻")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded image", use_column_width=True)
    
    tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        output = model(tensor)
        _, predicted = torch.max(output, 1)

        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence = probabilities[0][predicted.item()].item()

    if confidence < 0.98:
        st.subheader("That doesn't look like a cat.")
    elif predicted.item() == 0:
        st.subheader(f"Ivy - ({confidence*100:.1f}% confident)")
    else:
        st.subheader(f"Not Ivy - ({confidence*100:.1f}% confident)")