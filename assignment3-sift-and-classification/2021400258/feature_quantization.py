import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from pathlib import Path

def load_image_paths(file_path):
    """Load image paths from txt file"""
    with open(file_path, 'r') as f:
        paths = [line.strip() for line in f.readlines()]
    return paths

def get_descriptor_filename(image_path):
    """Convert image path to descriptor filename
    Example: kitchen/int474.jpg -> int474.npy
    """
    filename = os.path.basename(image_path)
    filename_without_ext = os.path.splitext(filename)[0]
    return filename_without_ext + '.npy'

def load_training_descriptors(train_paths, descriptor_folder):
    """Load all descriptors from training images"""
    print(f"Loading descriptors from {descriptor_folder}...")

    all_descriptors = []

    for i, img_path in enumerate(train_paths):
        desc_file = get_descriptor_filename(img_path)
        desc_path = os.path.join(descriptor_folder, desc_file)

        # Load the descriptor
        if os.path.exists(desc_path):
            desc = np.load(desc_path)
            all_descriptors.append(desc)
        else:
            print(f"Warning: {desc_path} not found!")

        # Show progress
        if (i + 1) % 100 == 0:
            print(f"Loaded {i + 1}/{len(train_paths)} descriptors")

    # Stack all descriptors together
    all_descriptors = np.vstack(all_descriptors)
    print(f"Total descriptors: {all_descriptors.shape}")

    return all_descriptors

def perform_kmeans(descriptors, k):
    """Perform k-means clustering"""
    print(f"\nRunning k-means with k={k}...")
    print(f"Number of samples: {descriptors.shape[0]}")
    print(f"Descriptor dimension: {descriptors.shape[1]}")

    kmeans = MiniBatchKMeans(
        n_clusters=k,
        random_state=55,
        verbose=1,
        batch_size=10000,
        n_init=2,
        max_iter=20
    )
    # Decreased n_init to 3 for faster execution; increase if we have enough time/resources
    kmeans.fit(descriptors)

    print(f"K-means finished!")
    print(f"Cluster centers shape: {kmeans.cluster_centers_.shape}")

    return kmeans

def create_histogram_for_image(descriptor, kmeans):
    """Create histogram of visual words for one image"""

    # Handle 1D descriptors (e.g., from Swin with global pooling)
    if descriptor.ndim == 1:
        descriptor = descriptor.reshape(1, -1)  # Shape: (1, feature_dim)

    # Assign each descriptor to nearest cluster
    labels = kmeans.predict(descriptor)

    # Count occurrences of each cluster
    k = kmeans.n_clusters
    hist = np.bincount(labels, minlength=k)

    # Normalize the histogram
    hist = hist.astype(float)
    if hist.sum() > 0:
        hist = hist / hist.sum()

    return hist

def create_histograms_for_all(image_paths, descriptor_folder, kmeans, output_folder, k):
    """Create histograms for all images"""
    print(f"\nCreating histograms for all images...")

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    for i, img_path in enumerate(image_paths):
        desc_file = get_descriptor_filename(img_path)
        desc_path = os.path.join(descriptor_folder, desc_file)

        if os.path.exists(desc_path):
            # Load descriptor
            desc = np.load(desc_path)

            # Create histogram
            hist = create_histogram_for_image(desc, kmeans)

            # Save histogram with k value in filename
            hist_filename = f'k{k}_{desc_file}'
            hist_path = os.path.join(output_folder, hist_filename)
            np.save(hist_path, hist)
        else:
            print(f"Warning: {desc_path} not found!")

        # Show progress
        if (i + 1) % 100 == 0:
            print(f"Created {i + 1}/{len(image_paths)} histograms")

    print(f"All histograms saved to {output_folder}")

def main():
    print("="*60)
    print("Feature Quantization - Bag of Visual Words")
    print("="*60)

    # Load image paths
    train_paths = load_image_paths('../data/TrainImages.txt')
    test_paths = load_image_paths('../data/TestImages.txt')
    all_paths = train_paths + test_paths

    print(f"\nDataset info:")
    print(f"Training images: {len(train_paths)}")
    print(f"Test images: {len(test_paths)}")
    print(f"Total images: {len(all_paths)}")

    # K values to try
    k_values = [50, 100, 500]

    # Process SIFT descriptors
    print("\n" + "="*60)
    print("Processing SIFT descriptors")
    print("="*60)

    # Load all training descriptors for SIFT
    sift_train_desc = load_training_descriptors(train_paths, '../data/descriptors/sift')

    # Run k-means for each k value
    for k in k_values:
        print(f"\n--- K = {k} ---")

        # Perform k-means
        kmeans = perform_kmeans(sift_train_desc, k)

        # Create histograms for all images
        create_histograms_for_all(
            all_paths,
            '../data/descriptors/sift',
            kmeans,
            '../data/histograms/sift',
            k
        )

    # Process Swin-B descriptors
    print("\n" + "="*60)
    print("Processing Swin-B descriptors")
    print("="*60)

    # K values to try
    k_values = [50, 100, 500]

    # Load all training descriptors for Swin-B
    swin_train_desc = load_training_descriptors(train_paths, '../data/descriptors/swin')

    # Run k-means for each k value
    for k in k_values:
        print(f"\n--- K = {k} ---")

        # Perform k-means
        kmeans = perform_kmeans(swin_train_desc, k)

        # Create histograms for all images
        create_histograms_for_all(
            all_paths,
            '../data/descriptors/swin',
            kmeans,
            '../data/histograms/swin',
            k
        )

    print("\n" + "="*60)
    print("Feature quantization completed!")
    print("="*60)
    print("\nHistograms saved in:")
    print("- data/histograms/sift/")
    print("- data/histograms/swin/")

if __name__ == "__main__":
    main()