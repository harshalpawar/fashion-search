import os
import torch
import faiss
import clip
import numpy as np
from PIL import Image
from tqdm import tqdm
from utils import PATHS
import gc
import logging
from datetime import datetime

# Setup logging
log_file = f"extract_embeddings_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# Configuration
BATCH_SIZE = 1000  # Process 1000 images at a time
ENABLE_PARALLEL = False  # Set to True if you want to enable parallel processing
NUM_WORKERS = 4  # Number of workers for parallel processing if enabled

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_grad_enabled(False)  # Disable gradient tracking completely

# Load CLIP model
try:
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    logging.info(f"Using device: {device}")
    if device == "cuda":
        logging.info(f"GPU Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
except Exception as e:
    logging.error(f"Error loading CLIP model: {e}")
    raise

# First count the total number of files to process
logging.info("Counting files...")
total_files = sum(1 for root, _, files in os.walk(PATHS['IMAGE_DIR']) 
                 for file in files if file.endswith((".jpg", ".png")))
logging.info(f"Found {total_files} images to process")

# Initialize FAISS index (dimension is 512 for CLIP ViT-B/32)
dimension = 512
index = faiss.IndexFlatL2(dimension)

# If GPU is available, use a GPU-enabled FAISS index
if torch.cuda.is_available():
    try:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        logging.info("FAISS index moved to GPU")
    except Exception as e:
        logging.warning(f"Failed to move FAISS index to GPU, falling back to CPU: {e}")

def process_image(img_path):
    """Process a single image and return its embedding and metadata."""
    try:
        # Extract metadata
        parts = img_path.split(os.sep)
        gender, category, product_id = parts[-4], parts[-3], parts[-2]
        
        # Process image
        with Image.open(img_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            image = preprocess(img).unsqueeze(0).to(device)

        # Generate embedding
        embedding = model.encode_image(image)
        if device == "cuda":
            embedding = embedding.cpu()
        embedding = embedding.numpy().astype(np.float32)

        return embedding.squeeze(), f"{img_path},{gender},{category},{product_id}"
    
    except Exception as e:
        logging.error(f"Error processing {img_path}: {e}")
        return None, None

# Process images in batches
batch_embeddings = []
batch_metadata = []
total_processed = 0
failed_images = []

# Create a new metadata file
with open(PATHS['METADATA_FILE'], 'w') as f:
    f.write("")  # Initialize empty file

logging.info("\nProcessing images in batches...")
with tqdm(total=total_files, desc="Processing images") as pbar:
    for root, _, files in os.walk(PATHS['IMAGE_DIR']):
        for file in files:
            if not file.endswith((".jpg", ".png")):
                continue

            img_path = os.path.join(root, file)
            embedding, metadata = process_image(img_path)
            
            if embedding is not None:
                batch_embeddings.append(embedding)
                batch_metadata.append(metadata)
            else:
                failed_images.append(img_path)
            
            # Process batch if it reaches BATCH_SIZE
            if len(batch_embeddings) >= BATCH_SIZE:
                try:
                    # Convert batch to numpy array and normalize
                    embeddings_array = np.array(batch_embeddings, dtype=np.float32)
                    faiss.normalize_L2(embeddings_array)
                    
                    # Add to index
                    index.add(embeddings_array)
                    
                    # Save metadata
                    with open(PATHS['METADATA_FILE'], "a") as f:
                        f.write("\n".join(batch_metadata) + "\n")
                    
                    # Clear batch
                    total_processed += len(batch_embeddings)
                    batch_embeddings = []
                    batch_metadata = []
                    
                    # Force garbage collection
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    gc.collect()
                
                except Exception as e:
                    logging.error(f"Error processing batch: {e}")
                    failed_images.extend([m.split(',')[0] for m in batch_metadata])
                    batch_embeddings = []
                    batch_metadata = []

            pbar.update(1)

# Process remaining images in the last batch
if batch_embeddings:
    logging.info("\nProcessing final batch...")
    try:
        embeddings_array = np.array(batch_embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_array)
        index.add(embeddings_array)
        
        with open(PATHS['METADATA_FILE'], "a") as f:
            f.write("\n".join(batch_metadata) + "\n")
        
        total_processed += len(batch_embeddings)
    except Exception as e:
        logging.error(f"Error processing final batch: {e}")
        failed_images.extend([m.split(',')[0] for m in batch_metadata])

# Convert index back to CPU for saving if it's on GPU
if torch.cuda.is_available():
    logging.info("\nMoving index back to CPU for saving...")
    try:
        index = faiss.index_gpu_to_cpu(index)
    except Exception as e:
        logging.error(f"Error moving index back to CPU: {e}")
        raise

# Save the FAISS index
logging.info("\nSaving FAISS index...")
faiss.write_index(index, PATHS['VECTORS_INDEX'])

# Save failed images list if any
if failed_images:
    failed_images_file = "failed_images.txt"
    with open(failed_images_file, "w") as f:
        f.write("\n".join(failed_images))
    logging.warning(f"⚠️ {len(failed_images)} images failed to process. See {failed_images_file}")

# Final cleanup
if device == "cuda":
    torch.cuda.empty_cache()
gc.collect()

logging.info(f"\n✅ Successfully processed {total_processed} images")
logging.info(f"✅ Image embeddings saved to: {PATHS['VECTORS_INDEX']}")
logging.info(f"✅ Metadata saved to: {PATHS['METADATA_FILE']}")
logging.info(f"✅ Vector dimension: {dimension}")
logging.info(f"✅ Index contains {index.ntotal} vectors")
if device == "cuda":
    logging.info(f"✅ Final GPU Memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
