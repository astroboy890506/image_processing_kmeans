import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt

def kmeans_segmentation(img_path, num_clusters):
    # Load the image
    img = cv2.imread(img_path)

    # Reshape the image to a 2D array of pixels
    pixels = img.reshape((-1, 3))  # Modify to (-1, 3) if color image

    # Convert to float32
    pixels = np.float32(pixels)

    # Define the criteria and flags for k-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    k = num_clusters  # Number of clusters
    flags = cv2.KMEANS_RANDOM_CENTERS

    # Apply k-means clustering
    ret, label, center = cv2.kmeans(pixels, k, None, criteria, 20, flags)  # Increase the iteration count

    # Convert the center values back to uint8
    center = np.uint8(center)

    # Separate pixels based on their labels (clusters)
    segmented_imgs = [np.zeros_like(img) for _ in range(k)]

    for i in range(k):
        cluster_mask = (label == i).reshape(img.shape[:2])
        segmented_imgs[i][cluster_mask] = img[cluster_mask]

    return img, segmented_imgs

def main():
    # Rest of your code

    if uploaded_file is not None:
        num_clusters = st.slider("Number of Clusters (k)", min_value=2, max_value=10, value=5)

        img_path = "uploaded_image.jpg"
        with open(img_path, "wb") as f:
            f.write(uploaded_file.read())

        original_img, segmented_imgs = kmeans_segmentation(img_path, num_clusters)

        # Display the original image and segmented images
        plt.figure(figsize=(20, 15))  # Increase the figure size

        plt.subplot(2, num_clusters + 1, 1)  # Adjust the subplot layout
        plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        for i in range(num_clusters):
            plt.subplot(2, num_clusters + 1, i + 2)  # Adjust the subplot layout
            plt.imshow(cv2.cvtColor(segmented_imgs[i], cv2.COLOR_BGR2RGB))
            plt.title(f'Cluster {i + 1}')
            plt.axis('off')

        plt.tight_layout()  # Automatically adjust subplot spacing
        fig = plt.gcf()  # Get the current figure
        st.pyplot(fig)    # Pass the figure to st.pyplot()

if __name__ == "__main__":
    main()
