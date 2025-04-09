import numpy as np
import cv2

def scale_image(image, scale_x, scale_y):
    """
    Scale an image using a transformation matrix.
    
    Args:
        image: Input image as numpy array
        scale_x: Scaling factor along x-axis
        scale_y: Scaling factor along y-axis
        
    Returns:
        Scaled image as numpy array
    """
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Create the scaling transformation matrix
    # [ scale_x      0      0 ]
    # [     0    scale_y    0 ]
    # [     0        0      1 ]
    scaling_matrix = np.array([
        [scale_x, 0, 0],
        [0, scale_y, 0]
    ], dtype=np.float32)
    
    # Calculate new dimensions
    new_width = int(width * scale_x)
    new_height = int(height * scale_y)
    
    # Apply the transformation
    scaled_image = cv2.warpAffine(
        image, 
        scaling_matrix, 
        (new_width, new_height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    return scaled_image

def rotate_image(image, angle_degrees):
    """
    Rotate an image using a transformation matrix.
    
    Args:
        image: Input image as numpy array
        angle_degrees: Rotation angle in degrees (positive = counterclockwise)
        
    Returns:
        Rotated image as numpy array
    """
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Calculate image center (pivot point for rotation)
    center = (width // 2, height // 2)
    
    # Convert angle to radians
    angle_radians = np.radians(angle_degrees)
    
    # Create the rotation matrix
    # [  cos(θ)  sin(θ)  0 ]
    # [ -sin(θ)  cos(θ)  0 ]
    # [    0       0     1 ]
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    
    # Calculate new dimensions to ensure the whole image is visible
    cos_theta = np.abs(rotation_matrix[0, 0])
    sin_theta = np.abs(rotation_matrix[0, 1])
    
    new_width = int((height * sin_theta) + (width * cos_theta))
    new_height = int((height * cos_theta) + (width * sin_theta))
    
    # Adjust the rotation matrix to account for the new dimensions
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]
    
    # Apply the transformation
    rotated_image = cv2.warpAffine(
        image, 
        rotation_matrix, 
        (new_width, new_height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    return rotated_image

def shear_image(image, shear_x, shear_y):
    """
    Apply shearing transformation to an image.
    
    Args:
        image: Input image as numpy array
        shear_x: Shearing factor along x-axis
        shear_y: Shearing factor along y-axis
        
    Returns:
        Sheared image as numpy array
    """
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Create the shearing transformation matrix
    # [  1   shear_x  0 ]
    # [ shear_y  1    0 ]
    # [  0      0    1 ]
    shear_matrix = np.array([
        [1, shear_x, 0],
        [shear_y, 1, 0]
    ], dtype=np.float32)
    
    # Calculate new dimensions to ensure the whole image is visible
    if shear_x != 0:
        new_width = width + int(abs(shear_x) * height)
    else:
        new_width = width
        
    if shear_y != 0:
        new_height = height + int(abs(shear_y) * width)
    else:
        new_height = height
    
    # Apply the transformation
    sheared_image = cv2.warpAffine(
        image, 
        shear_matrix, 
        (new_width, new_height),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    
    return sheared_image
