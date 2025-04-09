import numpy as np
import time

def compress_svd(image, k):
    """
    Compress an image using SVD by keeping only the k largest singular values.
    
    Args:
        image: Input grayscale image as numpy array
        k: Number of singular values to keep
        
    Returns:
        Compressed image as numpy array and compression statistics
    """
    # Ensure image is grayscale
    if len(image.shape) > 2:
        raise ValueError("SVD compression requires a grayscale image")
    
    # Perform SVD
    U, singular_values, Vt = np.linalg.svd(image, full_matrices=False)
    
    # Calculate cumulative energy (information) preserved
    total_energy = np.sum(singular_values**2)
    energy_preserved = np.sum(singular_values[:k]**2)
    information_retained = (energy_preserved / total_energy) * 100
    
    # Create the compressed image by using only k singular values
    compressed = np.zeros_like(image, dtype=float)
    for i in range(k):
        compressed += singular_values[i] * np.outer(U[:, i], Vt[i, :])
    
    # Ensure pixel values are within valid range
    compressed = np.clip(compressed, 0, 255).astype(np.uint8)
    
    # Calculate compression statistics
    original_size = image.shape[0] * image.shape[1]
    compressed_size = k * (image.shape[0] + image.shape[1] + 1)  # U + Vt + singular values
    compression_ratio = original_size / compressed_size
    
    # Create statistics dictionary
    stats = {
        'original_size': original_size,
        'compressed_size': compressed_size,
        'compression_ratio': compression_ratio,
        'information_retained': information_retained,
        'singular_values': singular_values
    }
    
    return compressed, stats
