import torch
import clip
import faiss
import numpy as np
from PIL import Image
import os
import shutil
from utils import PATHS, get_data_path, get_project_file

def clear_directory(directory):
    """Clear all contents of a directory."""
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)
    print(f"🗑️  Cleared output directory: {directory}")

# Use CPU for consistency
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()
print("✅ CLIP model loaded")

# Load FAISS index
index = faiss.read_index(PATHS['VECTORS_INDEX'])
print("✅ FAISS index loaded")

# Load metadata
with open(PATHS['METADATA_FILE'], "r") as f:
    metadata = f.read().splitlines()
print("✅ Metadata loaded")

def search_similar_images(query_img_path, top_k=5):
    """Search for similar images given a query image path."""
    # Load and preprocess image
    image = preprocess(Image.open(query_img_path)).unsqueeze(0).to(device)
    
    # Generate embedding
    with torch.no_grad():
        query_embedding = model.encode_image(image)
        query_embedding = query_embedding.cpu().numpy().astype(np.float32)
    
    # Normalize query embedding (since our index vectors are normalized)
    faiss.normalize_L2(query_embedding)
    
    # Search
    distances, indices = index.search(query_embedding, top_k)
    
    # Get results
    results = []
    for idx, distance in zip(indices[0], distances[0]):
        if idx < len(metadata):  # Safety check
            results.append((metadata[idx], distance))
    
    return results

def copy_images(query_path, results, output_dir):
    """Copy query and result images to output directory."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy query image
    query_filename = os.path.basename(query_path)
    query_output = os.path.join(output_dir, f"query_{query_filename}")
    shutil.copy2(query_path, query_output)
    print(f"\n📸 Query image saved as: {query_output}")
    
    # Copy result images
    print("\n📸 Similar images saved:")
    for i, (img_path, distance) in enumerate(results, 1):
        # Get the actual image path from metadata (first element before comma)
        source_path = img_path.split(',')[0]
        # Create a filename with the distance score
        filename = os.path.basename(source_path)
        result_filename = f"result_{i}_dist_{distance:.3f}_{filename}"
        output_path = os.path.join(output_dir, result_filename)
        
        shutil.copy2(source_path, output_path)
        print(f"  • {output_path}")

if __name__ == "__main__":
    # Setup input/output directories using project paths
    input_dir = get_project_file("test/input")
    output_dir = get_project_file("test/output")
    
    # Create input directory if it doesn't exist
    os.makedirs(input_dir, exist_ok=True)
    
    # Clear and recreate output directory
    clear_directory(output_dir)
    
    # Check if input directory is empty
    if not os.listdir(input_dir):
        print(f"\n⚠️  No images found in {input_dir}")
        print("Please place some images in the input directory and run again.")
        exit(0)
    
    # Process all images in input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Create query path and output directory
            query_image = os.path.join(input_dir, filename)
            query_output_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
            
            print(f"\n🔍 Searching for images similar to: {filename}")
            
            # Search similar images
            results = search_similar_images(query_image)
            
            # Print results
            print("\n🔍 Top matches:")
            for img_path, distance in results:
                metadata_parts = img_path.split(',')
                img_filename = os.path.basename(metadata_parts[0])
                category = metadata_parts[2]
                print(f"  • Distance: {distance:.3f} | Category: {category} | {img_filename}")
            
            # Copy images to output directory
            copy_images(query_image, results, query_output_dir)
