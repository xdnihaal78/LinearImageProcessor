import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

def convert_to_grayscale(img_array):
    """
    Convert an RGB image to grayscale using weighted averaging.
    
    Args:
        img_array: A numpy array representing an RGB image (H x W x C)
        
    Returns:
        A numpy array representing a grayscale image (H x W)
    """
    # Check if the image is already grayscale
    if len(img_array.shape) < 3 or img_array.shape[2] == 1:
        return img_array
    
    # Use standard weightings for RGB to grayscale conversion
    # These weights account for human perception of colors
    grayscale = np.dot(img_array[..., :3], [0.299, 0.587, 0.114])
    
    # Ensure the output is in the correct data type
    return grayscale.astype(np.uint8)

def convert_to_rgb(img_array):
    """
    Convert a grayscale image to RGB by duplicating the channel.
    
    Args:
        img_array: A numpy array representing a grayscale image (H x W)
        
    Returns:
        A numpy array representing an RGB image (H x W x 3)
    """
    # Check if image is already RGB
    if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
        return img_array
    
    # Ensure image is 2D
    if len(img_array.shape) == 3 and img_array.shape[2] == 1:
        img_array = img_array[:, :, 0]
    
    # Stack the same grayscale channel three times to create RGB
    rgb_img = np.stack((img_array,) * 3, axis=-1)
    
    return rgb_img.astype(np.uint8)

def display_image_matrix_info(img_array):
    """
    Display information about the image matrix.
    
    Args:
        img_array: A numpy array representing an image
    """
    # Get basic information
    shape = img_array.shape
    dtype = img_array.dtype
    
    # Create columns for display
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Matrix Information")
        st.write(f"Shape: {shape}")
        st.write(f"Data Type: {dtype}")
        st.write(f"Min Value: {img_array.min()}")
        st.write(f"Max Value: {img_array.max()}")
        
        # Determine if image is grayscale or color
        if len(shape) == 2 or (len(shape) == 3 and shape[2] == 1):
            st.write("Image Type: Grayscale")
        elif len(shape) == 3 and shape[2] >= 3:
            st.write("Image Type: Color (RGB/RGBA)")
            
    with col2:
        # Display a sample of the matrix (top-left corner)
        st.subheader("Matrix Sample (Top-Left Corner)")
        
        # For color images, show just the first channel
        if len(shape) == 3 and shape[2] >= 3:
            st.write("Red Channel:")
            st.write(img_array[:5, :5, 0])
        else:
            # For grayscale, show the intensity values
            st.write("Intensity Values:")
            display_array = img_array
            if len(shape) == 3:  # Handle grayscale with channel dimension
                display_array = display_array[:, :, 0]
            st.write(display_array[:5, :5])
