import numpy as np
import imageio.v3 as imageio
import matplotlib.pyplot as plt
from scipy.ndimage import convolve

def roberts_operator(image):
    """Menerapkan operator Roberts untuk deteksi tepi"""
    Gx = np.array([[1, 0], [0, -1]])
    Gy = np.array([[0, 1], [-1, 0]])
    
    edge_x = convolve(image, Gx)
    edge_y = convolve(image, Gy)
    
    return np.sqrt(edge_x**2 + edge_y**2)

def sobel_operator(image):
    """Menerapkan operator Sobel untuk deteksi tepi"""
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    edge_x = convolve(image, Gx)
    edge_y = convolve(image, Gy)
    
    return np.sqrt(edge_x**2 + edge_y**2)

# Load gambar grayscale
gray_image = imageio.imread("https://upload.wikimedia.org/wikipedia/commons/2/24/Lenna.png", pil_mode="L")

# Terapkan deteksi tepi
edges_roberts = roberts_operator(gray_image)
edges_sobel = sobel_operator(gray_image)

# Plot hasil
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(gray_image, cmap='gray')
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(edges_roberts, cmap='gray')
axes[1].set_title("Roberts Edge Detection")
axes[1].axis("off")

axes[2].imshow(edges_sobel, cmap='gray')
axes[2].set_title("Sobel Edge Detection")
axes[2].axis("off")

plt.show()

# Analisis Perbandingan
print("\nAnalisis Perbandingan:")
print("- Roberts Operator: Lebih tajam namun lebih sensitif terhadap noise. Cocok untuk gambar dengan kontras tinggi.")
print("- Sobel Operator: Lebih tahan terhadap noise dan memberikan hasil yang lebih halus dibandingkan Roberts.")
