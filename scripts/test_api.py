# test_endpoint.py
"""
Script for testing the /photography/background streaming endpoint.
Usage:
    python test_endpoint.py <image_path> [<api_url>]
"""
import sys
import os
import json
import base64
import requests

# Para poder importar src/ y config/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_image_as_base64
from config.config import DATA_DIR


def test_background_endpoint(image_path: str, api_url: str):
    """
    Sends the image as base64 to the streaming endpoint and prints each chunk as JSON.
    """
    data_uri = load_image_as_base64(image_path)
    payload = {
        "image_b64": data_uri,
        # "preferences": "your preferences here"  # Optional
    }

    print(f"POST {api_url} with image: {image_path}")
    response = requests.post(api_url, json=payload, stream=True)
    response.raise_for_status()

    print("Streaming response (NDJSON):")
    for line in response.iter_lines(decode_unicode=True):
        if line:
            print("Received:", line)


if __name__ == '__main__':
    sample_dir = DATA_DIR / "samples"
    images = [f for f in sample_dir.iterdir() if f.is_file()]

    test_image_path = images[0]

    api_url = 'http://localhost:5000/photography/background'
    test_background_endpoint(test_image_path, api_url)
