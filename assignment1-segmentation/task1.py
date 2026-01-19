from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

IMAGE_PATH = "task1_images/image01.jpeg"
K_VALUES = [2, 4, 8, 16, 32]

def convert_to_matrix(image_path):
    im = Image.open(image_path).convert("RGB")
    img_matrix = np.array(im)
    im.close()
    return img_matrix

def k_means_clustering(input_matrix, k, max_iters=60, initial_points=None):
    # Reshape to (num_pixels, 3) for clustering in color space
    original_shape = input_matrix.shape
    height, width, channels = input_matrix.shape
    num_pixels = height * width
    pixels = input_matrix.reshape(num_pixels, 3)

    if initial_points:
        # Use colors at clicked points as initial centroids
        centroids = np.array([input_matrix[y, x] for x, y in initial_points], dtype=float)
    else:
        # Random initialization: pick K random pixel colors
        random_indices = np.random.choice(pixels.shape[0], size=k, replace=False)
        centroids = pixels[random_indices].astype(float)

    for iteration in range(max_iters):
        # Calculate distances from each pixel to each centroid
        distances = np.linalg.norm(pixels[:, np.newaxis, :] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        # Find new centroids as the mean of assigned pixels
        new_centroids = np.array([
            pixels[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])

        # Check for convergence
        if np.allclose(centroids, new_centroids):
            break

        # Update centroids for next iteration
        centroids = new_centroids

    # Reshape labels back to image shape
    labels = labels.reshape(original_shape[0], original_shape[1])

    return centroids, labels

def rgb_to_lab(rgb_image):
    # Convert to [0, 255] and uint8 for PIL
    rgb_uint8 = (rgb_image * 255).astype(np.uint8)

    # Create PIL Image and convert to LAB
    pil_image = Image.fromarray(rgb_uint8, mode='RGB')
    lab_image = pil_image.convert('LAB')

    # Convert back to numpy array
    lab_array = np.array(lab_image).astype(float)

    return lab_array

def lab_to_rgb(lab_image):
    # Convert to uint8 for PIL
    lab_uint8 = lab_image.astype(np.uint8)

    # Create PIL Image in LAB mode and convert to RGB
    pil_lab = Image.fromarray(lab_uint8, mode='LAB')
    rgb_image = pil_lab.convert('RGB')

    # Convert to numpy and normalize to [0, 1]
    rgb_array = np.array(rgb_image).astype(float) / 255.0

    return rgb_array

def k_means_clustering_lab(input_matrix, k, max_iters=60, initial_points=None):
    original_shape = input_matrix.shape

    # Convert RGB to LAB
    img_lab = rgb_to_lab(input_matrix)

    # Reshape for clustering
    height, width, channels = img_lab.shape
    num_pixels = height * width
    pixels_lab = img_lab.reshape(num_pixels, 3)

    if initial_points:
        # Get LAB values at clicked points
        centroids = np.array([img_lab[y, x] for x, y in initial_points], dtype=float)
    else:
        random_indices = np.random.choice(pixels_lab.shape[0], size=k, replace=False)
        centroids = pixels_lab[random_indices].astype(float)

    # K-means in LAB space
    for iteration in range(max_iters):
        distances = np.linalg.norm(pixels_lab[:, np.newaxis, :] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)

        new_centroids = np.array([
            pixels_lab[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i]
            for i in range(k)
        ])

        if np.allclose(centroids, new_centroids):
            print(f"  Converged at iteration {iteration + 1}")
            break

        centroids = new_centroids

    # Create quantized image in LAB space
    labels = labels.reshape(original_shape[0], original_shape[1])
    segmented_lab = np.zeros_like(img_lab)
    for i in range(k):
        segmented_lab[labels == i] = centroids[i]

    # Convert back to RGB for display
    segmented_rgb = lab_to_rgb(segmented_lab)

    return segmented_rgb, labels


def quantize(img_matrix, k, initial_points=None, use_lab=False):
    if use_lab:
        segmented_img, _ = k_means_clustering_lab(img_matrix, k, initial_points=initial_points)
    else:
        centroids, labels = k_means_clustering(img_matrix, k, initial_points=initial_points)
        segmented_img = np.zeros_like(img_matrix)
        for i in range(k):
            segmented_img[labels == i] = centroids[i]

    return segmented_img

def collect_manual_points(image_path, k_values):
    max_k = max(k_values)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(Image.open(image_path))
    ax.set_title(f"Click {max_k} points to initialize centroids\n(We'll use subsets for smaller K values)")

    input_points = plt.ginput(max_k, show_clicks=True)
    plt.close(fig)

    # Convert to integer coordinates
    input_points = [(int(x), int(y)) for x, y in input_points]

    return input_points

def run_quantization_experiment(image_path, k_values, save_individual=False, lab_space=False):
    """
    Run quantization with both manual and random initialization for all K values.
    """
    img_matrix = convert_to_matrix(image_path)

    # Normalize to [0, 1]
    img_matrix_normalized = img_matrix.astype(float) / 255.0

    # Collect manual initialization points
    manual_points = collect_manual_points(image_path, k_values)

    # Store results
    manual_results = {}
    random_results = {}

    # Run quantization for each K value
    for k in k_values:
        print(f"\n{'='*50}")
        print(f"Processing K={k}")
        print(f"{'='*50}")

        # Manual initialization
        print(f"Running manual initialization (K={k})...")
        manual_init_points = manual_points[:k]  # Use first k points
        manual_results[k] = quantize(img_matrix_normalized, k, initial_points=manual_init_points, use_lab=lab_space)

        # Random initialization
        print(f"Running random initialization (K={k})...")
        random_results[k] = quantize(img_matrix_normalized, k, initial_points=None)

    # Display all results
    display_results(image_path, k_values, manual_results, random_results)

    if save_individual:
        # Save individual results
        save_results(image_path, k_values, manual_results, random_results)
        print("\nIndividual quantized images saved.")

def display_results(image_path, k_values, manual_results, random_results):
    """
    Display all quantization results in a grid.
    """
    num_k = len(k_values)

    fig, axes = plt.subplots(3, num_k, figsize=(4*num_k, 12))

    original_img = Image.open(image_path)

    # First row: original images
    for idx, k in enumerate(k_values):
        axes[0, idx].imshow(original_img)
        axes[0, idx].set_title(f"Original\n(K={k})", fontsize=10)
        axes[0, idx].axis('off')

    # Second row: manual initialization
    for idx, k in enumerate(k_values):
        axes[1, idx].imshow(manual_results[k])
        axes[1, idx].set_title(f"Manual Init\nK={k}", fontsize=10)
        axes[1, idx].axis('off')

    # Third row: random initialization
    for idx, k in enumerate(k_values):
        axes[2, idx].imshow(random_results[k])
        axes[2, idx].set_title(f"Random Init\nK={k}", fontsize=10)
        axes[2, idx].axis('off')

    plt.tight_layout()

    try:
        plt.show()
    except Exception as e:
        print(f"Error displaying plot: {e}")
    finally:
        plt.close()

def save_results(image_path, k_values, manual_results, random_results):
    """
    Save individual quantized images.
    """
    import os

    # Get base filename
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    for k in k_values:
        # Save manual initialization result
        manual_img = (manual_results[k] * 255).astype(np.uint8)
        Image.fromarray(manual_img).save(f"{base_name}_k{k}_manual.png")

        # Save random initialization result
        random_img = (random_results[k] * 255).astype(np.uint8)
        Image.fromarray(random_img).save(f"{base_name}_k{k}_random.png")

    print(f"\nIndividual images saved with pattern: {base_name}_k[2,4,8,16,32]_[manual/random].png")

if __name__ == "__main__":
    run_quantization_experiment(IMAGE_PATH, K_VALUES, False, False)
    print("\nExperiment complete!")
