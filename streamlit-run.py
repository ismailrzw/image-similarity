import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os
import faiss
import numpy as np
from collections import OrderedDict
import io

# --- Configuration (Change these paths) ---
TRAINED_MODEL_PATH = 'siamese_last_test.pth'
EMBEDDING_DB_PATH = 'embedding_database.index'
METADATA_DB_PATH = 'embedding_metadata.pt'
TOP_K_RESULTS = 5
COSINE_THRESHOLD = 0.75  # Tune this threshold

# --- Define the correct model architecture ---
class EmbeddingNet(nn.Module):
    def __init__(self, original_model, embedding_dim=128):
        super(EmbeddingNet, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-1])
        self.embedding_layer = nn.Linear(original_model.fc.in_features, embedding_dim)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.embedding_layer(x)
        return F.normalize(x, p=2, dim=1)

# --- Streamlit App Functions ---

@st.cache_resource
def load_model_and_database():
    """
    Loads the trained model and FAISS database.
    Uses st.cache_resource to prevent reloading on every user interaction.
    """
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the model
        checkpoint = torch.load(TRAINED_MODEL_PATH, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        
        base_model = models.resnet18(weights=None)
        siamese_model = EmbeddingNet(base_model)
        
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '').replace('embedding_model.', '')
            new_state_dict[name] = v
        siamese_model.load_state_dict(new_state_dict)
        siamese_model.eval()
        siamese_model.to(device)
        
        # Load the database
        index = faiss.read_index(EMBEDDING_DB_PATH)
        metadata = torch.load(METADATA_DB_PATH, weights_only=False)
        all_labels = metadata['labels']
        class_names = metadata['class_names']
        all_image_paths = metadata['image_paths']

        # 💡 FIX: Return the all_image_paths variable as it's needed later
        return siamese_model, index, all_labels, class_names, all_image_paths, device
    except FileNotFoundError as e:
        st.error(f"Error: A required file was not found. Please check the paths. Details: {e}")
        # 💡 FIX: Return None for all variables, including the new one
        return None, None, None, None, None, None
    except Exception as e:
        st.error(f"An error occurred while loading the model or database: {e}")
        # 💡 FIX: Return None for all variables
        return None, None, None, None, None, None

def process_and_search_image(model, index, all_labels, class_names, all_image_paths, device, image):
    """
    Processes an image and searches the database for similar ones.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Process the image
    query_tensor = transform(image.convert('RGB')).unsqueeze(0).to(device)
    
    with torch.no_grad():
        query_embedding = model(query_tensor)
    
    query_embedding_np = query_embedding.cpu().numpy()
    
    # Search the database
    distances, indices = index.search(query_embedding_np, TOP_K_RESULTS)
    
    results = []
    for i in range(TOP_K_RESULTS):
        match_index = indices[0][i]
        match_label_id = all_labels[match_index]
        match_class_name = class_names[match_label_id]
        # 💡 FIX: Use the all_image_paths parameter to get the path
        match_image_path = all_image_paths[match_index]
        cosine_sim = distances[0][i]
        
        results.append({
            "rank": i + 1,
            "class_name": match_class_name,
            "cosine_similarity": cosine_sim,
            "accepted": cosine_sim >= COSINE_THRESHOLD,
            "image_path": match_image_path
        })
    return results

# --- Main Streamlit App Layout ---

st.set_page_config(page_title="Image Similarity Search", layout="wide")
st.title("Image Similarity Search Engine")
st.markdown("Upload an image to find the most similar images in our database.")
st.divider()

# 💡 FIX: Receive the new variable from the loading function
siamese_model, index, all_labels, class_names, all_image_paths, device = load_model_and_database()

if siamese_model is None:
    st.stop()  # Stop the app if model/db loading failed

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', width=250)
        st.write("")
        st.write("Searching for similar images...")

        # 💡 FIX: Pass the new all_image_paths variable to the function
        results = process_and_search_image(siamese_model, index, all_labels, class_names, all_image_paths, device, image)

        # Display the results
        st.subheader("Top Similar Images Found:")
        st.markdown("---")
        for res in results:
            col1, col2 = st.columns([3, 4])
            with col1:
                st.write(f"**Rank:** {res['rank']}")
                st.write(f"**Class:** {res['class_name']}")
                st.write(f"**Similarity:** {res['cosine_similarity']:.4f}")
                status = "Accepted" if res['accepted'] else "Rejected"
                emoji = "✅" if res['accepted'] else "❌"
                st.write(f"**Status:** {emoji} {status}")
            
            with col2:
                # You would display the similar image here.
                # Since we only have metadata (class_name), we can't show the exact image.
                # A proper solution would store image paths in your metadata.
                if os.path.exists(res['image_path']):
                    similar_image = Image.open(res['image_path'])
                    st.image(similar_image, caption=f"Match: {res['class_name']}", width=200)
                else:
                    st.warning(f"Image not found at path: {res['image_path']}")         
            st.markdown("---")

    except Exception as e:
        st.error(f"An error occurred during processing: {e}")
