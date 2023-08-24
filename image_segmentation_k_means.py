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
    st.set_page_config(
        page_title="Image Segmentation App",
        page_icon=":camera:",
        layout="centered",
        initial_sidebar_state="expanded"
    )

    st.title("Image Segmentation using K-Means")

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        num_clusters = st.slider("Number of Clusters (k)", min_value=2, max_value=10, value=5)

        img_path = "uploaded_image.jpg"
        with open(img_path, "wb") as f:
            f.write(uploaded_file.read())

        original_img, segmented_imgs = kmeans_segmentation(img_path, num_clusters)

        image_width = st.slider("Adjust Image Width", min_value=100, max_value=800, value=500)

        st.subheader("Original Image")
        st.image(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB), width=image_width)

        col1, col2 = st.beta_columns(2)

        for i in range(num_clusters):
            with col1:
                st.subheader(f"Cluster {i + 1}")
                st.image(cv2.cvtColor(segmented_imgs[i], cv2.COLOR_BGR2RGB), width=image_width)

            if i + 1 < num_clusters:  # Add a check to avoid IndexError
                with col2:
                    st.subheader(f"Cluster {i + 2}")
                    st.image(cv2.cvtColor(segmented_imgs[i + 1], cv2.COLOR_BGR2RGB), width=image_width)
                i += 1

if __name__ == "__main__":
    main()
