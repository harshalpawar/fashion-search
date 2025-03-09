import os

def get_project_root():
    """Get absolute path to project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_data_path(relative_path=""):
    """Get absolute path to a file/directory in the data directory."""
    return os.path.join(get_project_root(), "data", relative_path)

def get_project_file(relative_path):
    """Get absolute path to any file/directory in the project."""
    return os.path.join(get_project_root(), relative_path)

# Common paths used across the project
PATHS = {
    'IMAGE_DIR': get_data_path('images'),
    'VECTORS_INDEX': get_project_file('fashion_vectors.index'),
    'METADATA_FILE': get_project_file('metadata.csv'),
} 