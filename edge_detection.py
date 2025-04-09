import numpy as np
import cv2

def detect_edges(image, method='sobel', direction='both', threshold1=100, threshold2=200):
    """
    Detect edges in an image using various convolution-based methods.
    
    Args:
        image: Input grayscale image as numpy array
        method: Edge detection method ('sobel', 'prewitt', 'laplacian', 'canny')
        direction: Edge direction for directional operators ('both', 'horizontal', 'vertical')
        threshold1: Lower threshold for Canny edge detector
        threshold2: Upper threshold for Canny edge detector
        
    Returns:
        Edge image as numpy array
    """
    # Ensure image is grayscale
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Convert to float for processing
    image_float = image.astype(np.float32)
    
    if method == 'sobel':
        # Sobel operators for edge detection
        # Horizontal Sobel kernel:
        # [-1 -2 -1]
        # [ 0  0  0]
        # [ 1  2  1]
        #
        # Vertical Sobel kernel:
        # [-1  0  1]
        # [-2  0  2]
        # [-1  0  1]
        
        if direction == 'horizontal' or direction == 'both':
            sobel_h = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            
        if direction == 'vertical' or direction == 'both':
            sobel_v = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        
        if direction == 'both':
            # Combine horizontal and vertical edges
            edges = np.sqrt(sobel_h**2 + sobel_v**2)
        elif direction == 'horizontal':
            edges = np.abs(sobel_h)
        else:  # vertical
            edges = np.abs(sobel_v)
        
    elif method == 'prewitt':
        # Prewitt operators for edge detection
        # Horizontal Prewitt kernel:
        # [-1 -1 -1]
        # [ 0  0  0]
        # [ 1  1  1]
        #
        # Vertical Prewitt kernel:
        # [-1  0  1]
        # [-1  0  1]
        # [-1  0  1]
        
        kernel_h = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)
        kernel_v = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        
        if direction == 'horizontal' or direction == 'both':
            prewitt_h = cv2.filter2D(image, -1, kernel_h)
            
        if direction == 'vertical' or direction == 'both':
            prewitt_v = cv2.filter2D(image, -1, kernel_v)
        
        if direction == 'both':
            # Combine horizontal and vertical edges
            edges = np.sqrt(prewitt_h**2 + prewitt_v**2)
        elif direction == 'horizontal':
            edges = np.abs(prewitt_h)
        else:  # vertical
            edges = np.abs(prewitt_v)
            
    elif method == 'laplacian':
        # Laplacian operator for edge detection
        # Laplacian kernel:
        # [ 0  1  0]
        # [ 1 -4  1]
        # [ 0  1  0]
        
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        edges = np.abs(laplacian)
        
    elif method == 'canny':
        # Canny edge detection
        # This is a multi-stage algorithm:
        # 1. Noise reduction with Gaussian filter
        # 2. Gradient calculation
        # 3. Non-maximum suppression
        # 4. Double thresholding
        # 5. Edge tracking by hysteresis
        
        edges = cv2.Canny(image, threshold1, threshold2)
        return edges  # Canny already returns a binary image
    
    else:
        raise ValueError(f"Unknown edge detection method: {method}")
    
    # Normalize to 0-255 range
    edges = (edges / edges.max() * 255).astype(np.uint8)
    
    return edges
