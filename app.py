import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import io
import time

# Import our custom modules
from image_utils import convert_to_grayscale, convert_to_rgb, display_image_matrix_info
from transformations import scale_image, rotate_image, shear_image
from compression import compress_svd
from edge_detection import detect_edges

# Set page title and configuration
st.set_page_config(
    page_title="Image Processing with Linear Algebra",
    layout="wide"
)

st.title("Image Processing with Linear Algebra")

st.markdown("""
This application demonstrates various image processing techniques using linear algebra concepts.
Upload an image to get started!
""")

# Sidebar with instructions
with st.sidebar:
    st.header("About")
    st.markdown("""
    This application uses linear algebra concepts to process images:
    
    1. **Image Representation**: Convert images to matrices
    2. **Transformations**: Apply scaling, rotation, and shearing
    3. **SVD Compression**: Reduce image size while preserving quality
    4. **Edge Detection**: Find edges using convolution matrices
    
    Every operation is backed by linear algebra theory explained in each section.
    """)

# File uploader for image
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# Main content
if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file)
    
    # Convert to numpy array for processing
    img_array = np.array(image)
    
    # Display original image
    st.header("Original Image")
    st.image(img_array, caption="Original Image", use_container_width=True)
    
    # Display image information
    display_image_matrix_info(img_array)
    
    # Create tabs for different operations
    tabs = st.tabs(["Color Conversion", "Transformations", "SVD Compression", "Edge Detection"])
    
    # Tab 1: Color Conversion
    with tabs[0]:
        st.header("Image Representation as Matrices")
        st.markdown("""
        ### Mathematical Background
        
        Digital images are represented as matrices where each element corresponds to a pixel value:
        - Grayscale images: One matrix with values from 0 (black) to 255 (white)
        - RGB images: Three matrices (channels) for Red, Green, and Blue values
        
        Converting between color representations involves matrix operations.
        """)
        
        col1, col2 = st.columns(2)
        
        # Convert to grayscale if image is color
        if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
            with col1:
                st.subheader("Grayscale Conversion")
                gray_img = convert_to_grayscale(img_array)
                st.image(gray_img, caption="Grayscale Image", use_container_width=True)
                st.write(f"Matrix shape: {gray_img.shape}")
                
                # Show a small portion of the matrix
                st.write("Sample of the matrix (top-left corner):")
                st.write(gray_img[:5, :5])
                
        # If image is grayscale, show RGB conversion
        if len(img_array.shape) == 2 or (len(img_array.shape) == 3 and img_array.shape[2] == 1):
            with col2:
                st.subheader("RGB Conversion")
                rgb_img = convert_to_rgb(img_array)
                st.image(rgb_img, caption="RGB Image", use_container_width=True)
                st.write(f"Matrix shape: {rgb_img.shape}")
                
                # Show separate channels
                r_channel = rgb_img[:,:,0]
                g_channel = rgb_img[:,:,1]
                b_channel = rgb_img[:,:,2]
                
                st.write("Red channel (sample):")
                st.write(r_channel[:5, :5])
    
    # Tab 2: Transformations
    with tabs[1]:
        st.header("Linear Transformations")
        st.markdown("""
        ### Mathematical Background
        
        Linear transformations in image processing involve applying transformation matrices to pixel coordinates.
        
        For a 2D image with coordinates (x, y), transformations are applied using a matrix multiplication:
        
        - **Scaling**: Multiplying coordinates by scaling factors
        - **Rotation**: Using a rotation matrix based on trigonometric functions
        - **Shearing**: Shifting one coordinate proportionally to the other
        
        These operations are represented by transformation matrices that are applied to every pixel.
        """)
        
        # Convert to grayscale for transformations if it's a color image
        if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
            working_img = convert_to_grayscale(img_array)
        else:
            working_img = img_array.copy()
        
        # Transformation parameters
        transform_type = st.selectbox(
            "Select Transformation Type",
            ["Scaling", "Rotation", "Shearing"]
        )
        
        if transform_type == "Scaling":
            col1, col2 = st.columns(2)
            with col1:
                scale_x = st.slider("Scale X", 0.1, 2.0, 1.0, 0.1)
                scale_y = st.slider("Scale Y", 0.1, 2.0, 1.0, 0.1)
                
                st.markdown(f"""
                Scaling Matrix:
                
                $\\begin{{bmatrix}} 
                {scale_x} & 0 \\\\
                0 & {scale_y}
                \\end{{bmatrix}}$
                """)
            
            with col2:
                transformed_img = scale_image(working_img, scale_x, scale_y)
                st.image(transformed_img, caption=f"Scaled Image (x: {scale_x}, y: {scale_y})", use_container_width=True)
        
        elif transform_type == "Rotation":
            col1, col2 = st.columns(2)
            with col1:
                angle = st.slider("Rotation Angle (degrees)", -180, 180, 0, 5)
                angle_rad = np.radians(angle)
                cos_val = round(np.cos(angle_rad), 3)
                sin_val = round(np.sin(angle_rad), 3)
                
                st.markdown(f"""
                Rotation Matrix:
                
                $\\begin{{bmatrix}} 
                {cos_val} & -{sin_val} \\\\
                {sin_val} & {cos_val}
                \\end{{bmatrix}}$
                """)
            
            with col2:
                transformed_img = rotate_image(working_img, angle)
                st.image(transformed_img, caption=f"Rotated Image ({angle}°)", use_container_width=True)
        
        elif transform_type == "Shearing":
            col1, col2 = st.columns(2)
            with col1:
                shear_x = st.slider("Shear X", -1.0, 1.0, 0.0, 0.1)
                shear_y = st.slider("Shear Y", -1.0, 1.0, 0.0, 0.1)
                
                st.markdown(f"""
                Shearing Matrix:
                
                $\\begin{{bmatrix}} 
                1 & {shear_x} \\\\
                {shear_y} & 1
                \\end{{bmatrix}}$
                """)
            
            with col2:
                transformed_img = shear_image(working_img, shear_x, shear_y)
                st.image(transformed_img, caption=f"Sheared Image (x: {shear_x}, y: {shear_y})", use_container_width=True)
    
    # Tab 3: SVD Compression
    with tabs[2]:
        st.header("Singular Value Decomposition (SVD) Compression")
        st.markdown("""
        ### Mathematical Background
        
        SVD decomposes a matrix A into three matrices: A = UΣV^T, where:
        - U and V are orthogonal matrices
        - Σ is a diagonal matrix with singular values
        
        For image compression, we can keep only the k largest singular values, 
        creating a lower-rank approximation of the original image.
        
        The compression works because most images have information concentrated in 
        the largest singular values, allowing us to discard the smaller ones.
        """)
        
        # Work in grayscale for SVD visualization
        if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
            working_img = convert_to_grayscale(img_array)
        else:
            working_img = img_array.copy()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Get maximum rank (min of height, width)
            max_rank = min(working_img.shape[0], working_img.shape[1])
            compression_ratio = st.slider(
                "Compression Ratio (%)", 
                1, 100, 20, 1, 
                help="Percentage of singular values to keep (lower = more compression)"
            )
            
            k = int(max(1, (compression_ratio / 100) * max_rank))
            st.write(f"Using {k} singular values out of {max_rank} maximum")
            
            # Measure compression time
            start_time = time.time()
            compressed_img, compression_stats = compress_svd(working_img, k)
            end_time = time.time()
            
            # Display compression statistics
            st.subheader("Compression Statistics")
            st.write(f"Compression time: {end_time - start_time:.4f} seconds")
            st.write(f"Original size: {compression_stats['original_size']} elements")
            st.write(f"Compressed size: {compression_stats['compressed_size']} elements")
            st.write(f"Compression ratio: {compression_stats['compression_ratio']:.2f}x")
            st.write(f"Information retained: {compression_stats['information_retained']:.2f}%")
            
            # Display the singular values
            st.subheader("Singular Values Distribution")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(compression_stats['singular_values'][:50], marker='o', linestyle='-', markersize=4)
            ax.set_xlabel('Index')
            ax.set_ylabel('Magnitude')
            ax.set_title('First 50 Singular Values')
            ax.grid(True)
            st.pyplot(fig)
            
        with col2:
            st.image(compressed_img, caption=f"Compressed Image (k={k})", use_container_width=True)
            
            # Calculate and display the error
            error_img = np.abs(working_img.astype(float) - compressed_img.astype(float))
            error_img = (error_img / error_img.max() * 255).astype(np.uint8)
            
            st.subheader("Error Visualization")
            st.image(error_img, caption="Error (darker = more accurate)", use_container_width=True)

    # Tab 4: Edge Detection
    with tabs[3]:
        st.header("Edge Detection with Convolution")
        st.markdown("""
        ### Mathematical Background
        
        Edge detection uses convolution operations with specialized kernels (small matrices) 
        to identify areas of rapid intensity change in an image.
        
        Convolution is a mathematical operation where each pixel is replaced with a weighted sum 
        of neighboring pixels. The weights are defined by the kernel matrix.
        
        Common edge detection kernels include:
        - Sobel operators (detect horizontal and vertical edges)
        - Prewitt operators (similar to Sobel but with uniform weights)
        - Laplacian (detects edges in all directions)
        """)
        
        # Work in grayscale for edge detection
        if len(img_array.shape) == 3 and img_array.shape[2] >= 3:
            working_img = convert_to_grayscale(img_array)
        else:
            working_img = img_array.copy()
        
        col1, col2 = st.columns(2)
        
        with col1:
            edge_method = st.selectbox(
                "Edge Detection Method",
                ["Sobel", "Prewitt", "Laplacian", "Canny"]
            )
            
            if edge_method in ["Sobel", "Prewitt"]:
                direction = st.radio("Edge Direction", ["Both", "Horizontal", "Vertical"])
            
            if edge_method == "Canny":
                threshold1 = st.slider("Lower Threshold", 0, 255, 100, 5)
                threshold2 = st.slider("Upper Threshold", 0, 255, 200, 5)
            
            # Display the convolution kernel
            st.subheader("Convolution Kernel")
            
            if edge_method == "Sobel":
                if direction == "Horizontal":
                    kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
                elif direction == "Vertical":
                    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                else:
                    st.write("Using both horizontal and vertical Sobel operators")
                    st.write("Horizontal kernel:")
                    st.write(np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))
                    st.write("Vertical kernel:")
                    st.write(np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
                    kernel = None
            
            elif edge_method == "Prewitt":
                if direction == "Horizontal":
                    kernel = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
                elif direction == "Vertical":
                    kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
                else:
                    st.write("Using both horizontal and vertical Prewitt operators")
                    st.write("Horizontal kernel:")
                    st.write(np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]]))
                    st.write("Vertical kernel:")
                    st.write(np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]))
                    kernel = None
            
            elif edge_method == "Laplacian":
                kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
                st.write(kernel)
            
            elif edge_method == "Canny":
                st.write("Canny uses multiple steps including Gaussian blur and gradient calculation")
                kernel = None
        
        with col2:
            # Process the image with the selected edge detection method
            if edge_method == "Sobel":
                edges = detect_edges(working_img, method="sobel", direction=direction.lower())
            elif edge_method == "Prewitt":
                edges = detect_edges(working_img, method="prewitt", direction=direction.lower())
            elif edge_method == "Laplacian":
                edges = detect_edges(working_img, method="laplacian")
            elif edge_method == "Canny":
                edges = detect_edges(working_img, method="canny", threshold1=threshold1, threshold2=threshold2)
            
            st.image(edges, caption=f"Edge Detection using {edge_method}", use_container_width=True)

else:
    # Display placeholder when no image is uploaded
    st.info("Please upload an image to begin processing.")
    
    # Show a brief explanation of what the app can do
    st.header("What You Can Do With This App")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Image Representation")
        st.write("""
        - Convert images between grayscale and RGB
        - View the matrix representation of images
        - Understand how digital images are stored as matrices
        """)
        
        st.subheader("Linear Transformations")
        st.write("""
        - Scale images along X and Y axes
        - Rotate images by any angle
        - Apply shearing transformations
        - See the underlying transformation matrices
        """)
    
    with col2:
        st.subheader("SVD Compression")
        st.write("""
        - Compress images using Singular Value Decomposition
        - Control the compression ratio
        - Visualize the information loss
        - Understand how SVD can reduce image size
        """)
        
        st.subheader("Edge Detection")
        st.write("""
        - Apply various edge detection algorithms
        - Understand convolution operations
        - Compare different edge detection methods
        - Visualize how convolution matrices detect edges
        """)

st.markdown("---")
st.markdown("### Linear Algebra in Image Processing")
st.markdown("""
This application demonstrates how linear algebra principles form the foundation of image processing:

- **Matrices** represent images as 2D or 3D arrays of pixel values
- **Linear transformations** modify images through matrix multiplication
- **SVD (Singular Value Decomposition)** provides a way to compress images by approximating matrices
- **Convolution** operations use small matrices (kernels) to detect features like edges
""")
