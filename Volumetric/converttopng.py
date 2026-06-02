import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

bmp_dir = "C:\\Users\\osman\\Documents\\FYP Datasets\\Batch 6 Sample"

def convert_bmp_to_png(bmp_path, png_path):
    """Convert a BMP image to PNG format."""
    try:
        with Image.open(bmp_path) as img:
            img.save(png_path, 'PNG')
        print(f"Converted: {bmp_path} -> {png_path}")
    except Exception as e:
        print(f"Error converting {bmp_path}: {e}")

def convert_all_bmps_to_pngs(bmp_directory):
    """Convert all BMP images in the specified directory to PNG format."""
    for filename in os.listdir(bmp_directory):
        if filename.lower().endswith('.bmp'):
            bmp_path = os.path.join(bmp_directory, filename)
            png_filename = os.path.splitext(filename)[0] + '.png'
            png_path = os.path.join(bmp_directory, png_filename)
            convert_bmp_to_png(bmp_path, png_path)
if __name__ == "__main__":
    convert_all_bmps_to_pngs(bmp_dir)
    