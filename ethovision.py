import cv2
import numpy as np
from scipy.stats import mannwhitneyu
from matplotlib import pyplot as plt

# Load the two heatmap images
image1_path = "C:/Users/User/Documents/Uni/Uni 3rd year AI/Semester 1/Neuroscience Minor/Ethovision practical 23-10-2024/Neuroscience SSRI ethovision practical/Heatmap Image 0001 CTR.jpg"  # Replace with the actual image path
image2_path = "C:/Users/User/Documents/Uni/Uni 3rd year AI/Semester 1/Neuroscience Minor/Ethovision practical 23-10-2024/Neuroscience SSRI ethovision practical/Heatmap Image 0001FLX.jpg"  # Replace with the actual image path

heatmap1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
heatmap2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

# Ensure the images have the same dimensions
if heatmap1.shape != heatmap2.shape:
    raise ValueError("The heatmaps do not have the same dimensions.")

# Define the Region of Interest (ROI) - center of the image
height, width = heatmap1.shape
roi_center_fraction = 0.5  # Example: 50% of the image width and height for the ROI
roi_h_start = int(height * (1 - roi_center_fraction) / 2)
roi_h_end = int(height * (1 + roi_center_fraction) / 2)
roi_w_start = int(width * (1 - roi_center_fraction) / 2)
roi_w_end = int(width * (1 + roi_center_fraction) / 2)

# Extract ROI for both heatmaps
roi_heatmap1 = heatmap1[roi_h_start:roi_h_end, roi_w_start:roi_w_end]
roi_heatmap2 = heatmap2[roi_h_start:roi_h_end, roi_w_start:roi_w_end]

# Flatten the ROI arrays to 1D for statistical comparison
roi_heatmap1_flat = roi_heatmap1.flatten()
roi_heatmap2_flat = roi_heatmap2.flatten()

# Perform the Mann-Whitney U Test to compare the two distributions
stat, p_value = mannwhitneyu(roi_heatmap1_flat, roi_heatmap2_flat)

# Display the p-value and interpret significance
print(f"U Statistic: {stat}")
print(f"P-value: {p_value}")

if p_value < 0.05:
    print("The difference in time spent in the center is statistically significant.")
else:
    print("The difference in time spent in the center is NOT statistically significant.")

# Optional: Visualize the ROIs
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(roi_heatmap1, cmap='hot')
plt.title('ROI Heatmap 1')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(roi_heatmap2, cmap='hot')
plt.title('ROI Heatmap 2')
plt.colorbar()

plt.show()
