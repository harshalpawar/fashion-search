# Fashion Image Search

A simple image similarity search system using CLIP embeddings and FAISS index. The system finds visually similar fashion items from a dataset based on a query image.

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install torch clip-by-openai faiss-cpu pillow tqdm
```

## Usage

1. First, run the embedding extraction:
```bash
python src/extract_embeddings.py
```
This will process all images in the dataset and create embeddings.

2. To search for similar images:
- Place your query images in `test/input/`
- Run:
```bash
python src/search.py
```
- Find results in `test/output/[query_image_name]/`

## Output Format

For each query image, the results will be organized as:
- `query_image.jpg` - Your query image
- `result_1_dist_X.XXX_image.jpg` - Similar images with their distance scores (lower is better)

## Note

This project uses:
- OpenAI's CLIP model for image embeddings
- FAISS for efficient similarity search
- PyTorch for deep learning operations
