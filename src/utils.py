import os
import joblib
from pathlib import Path

def save_object(file_path, obj):
    """
    Saves a Python object to the specified file path using joblib.

    Args:
        file_path (str or Path): The path where the object should be saved.
        obj (any): The Python object to be serialized and saved.

    Raises:
        Exception: If any error occurs during the saving process.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        joblib.dump(obj, file_path)
        print(f"Object saved to {file_path}")
    
    except Exception as e:
        raise Exception(f"Error saving object to {file_path}: {e}")
    

def load_object(file_path):
    """
    Loads a Python object from the specified file path using joblib.

    Args:
        file_path (str or Path): The path to the file from which the object should be loaded.

    Returns:
        any: The Python object that was deserialized from the file.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
        Exception: If any other error occurs during the loading process.
    """
    try:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found at {file_path}")
        
        obj = joblib.load(file_path)
        print(f"Object loaded from {file_path}")
        return obj
    
    except Exception as e:
        raise Exception(f"Error loading object from {file_path}: {e}")