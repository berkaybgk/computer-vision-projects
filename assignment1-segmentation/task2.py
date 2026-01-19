from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def convert_to_matrix(image_path):
    im = Image.open(image_path).convert("RGB")
    img_matrix = np.array(im)
    im.close()
    return img_matrix


def rgb_to_grayscale(img):
    """Convert RGB image to grayscale using luminosity method."""
    return 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]


def apply_threshold(img, threshold):
    """
    Apply binary threshold to image.
    Returns binary image with values [0, 1].
    """
    binary = np.zeros_like(img, dtype=np.uint8)
    binary[img > threshold] = 1
    return binary


def erode(img, kernel_size=3):
    """
    Apply erosion morphological operation.
    Removes small white noise from binary image.
    """
    height, width = img.shape
    result = np.zeros_like(img)
    pad = kernel_size // 2

    padded = np.pad(img, pad, mode='constant', constant_values=0)

    for i in range(height):
        for j in range(width):
            region = padded[i:i+kernel_size, j:j+kernel_size]
            if np.all(region == 1):
                result[i, j] = 1

    return result


def dilate(img, kernel_size=3):
    """
    Apply dilation morphological operation.
    Fills small black holes in binary image.
    """
    height, width = img.shape
    result = np.zeros_like(img)
    pad = kernel_size // 2

    padded = np.pad(img, pad, mode='constant', constant_values=0)

    for i in range(height):
        for j in range(width):
            region = padded[i:i+kernel_size, j:j+kernel_size]
            if np.any(region == 1):
                result[i, j] = 1

    return result


def opening(img, kernel_size=3):
    """
    Morphological opening: erosion followed by dilation.
    Removes small objects while preserving larger ones.
    """
    eroded = erode(img, kernel_size)
    opened = dilate(eroded, kernel_size)
    return opened


def closing(img, kernel_size=3):
    """
    Morphological closing: dilation followed by erosion.
    Fills small holes while preserving object boundaries.
    """
    dilated = dilate(img, kernel_size)
    closed = erode(dilated, kernel_size)
    return closed


def connected_components_8(binary_img):
    """
    Perform connected component analysis with 8-connectivity.
    Uses iterative flood-fill approach.

    Returns:
        labeled_img: Image where each component has a unique label
        num_components: Number of connected components found
    """
    height, width = binary_img.shape
    labeled = np.zeros((height, width), dtype=np.int32)
    current_label = 0

    # 8-connected neighborhood offsets (row, col)
    neighbors_8 = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]

    def flood_fill(start_i, start_j, label):
        """Flood fill algorithm to label all connected pixels."""
        stack = [(start_i, start_j)]

        while stack:
            i, j = stack.pop()

            if i < 0 or i >= height or j < 0 or j >= width:
                continue
            if labeled[i, j] != 0 or binary_img[i, j] == 0:
                continue

            labeled[i, j] = label

            for di, dj in neighbors_8:
                ni, nj = i + di, j + dj
                if (0 <= ni < height and 0 <= nj < width and
                        binary_img[ni, nj] == 1 and labeled[ni, nj] == 0):
                    stack.append((ni, nj))

    # Traverse entire image
    for i in range(height):
        for j in range(width):
            if binary_img[i, j] == 1 and labeled[i, j] == 0:
                current_label += 1
                flood_fill(i, j, current_label)

    return labeled, current_label


def filter_components_by_size(labeled_img, num_components, min_size=50, max_size=None):
    """
    Filter connected components by size to remove noise.
    """
    filtered = np.zeros_like(labeled_img, dtype=np.uint8)
    valid_count = 0

    for label in range(1, num_components + 1):
        component_size = np.sum(labeled_img == label)

        if component_size >= min_size:
            if max_size is None or component_size <= max_size:
                filtered[labeled_img == label] = 1
                valid_count += 1

    return filtered, valid_count


def count_connected_components(img, threshold=128, morph_operation=None,
                               kernel_size=3, min_size=50, max_size=None):
    """
    Main function to count connected components in an image.

    Args:
        img: RGB image (numpy array)
        threshold: Threshold value for binary conversion (0-255)
        morph_operation: None, 'opening', 'closing'
        kernel_size: Size of morphological kernel
        min_size: Minimum component size to count
        max_size: Maximum component size to count (None for no limit)

    Returns:
        binary_img: Binary image used for counting
        count: Number of components found
    """
    # Convert to grayscale
    if len(img.shape) == 3:
        gray = rgb_to_grayscale(img)
    else:
        gray = img

    # Normalize to 0-255 range
    if gray.max() <= 1:
        gray = (gray * 255).astype(np.uint8)
    else:
        gray = gray.astype(np.uint8)

    # Apply threshold - inverted for dark objects on light background
    binary = apply_threshold(gray, threshold)
    binary = 1 - binary  # Invert so dark objects become white (1)

    # Apply morphological operations if specified
    if morph_operation == 'opening':
        binary = opening(binary, kernel_size)
    elif morph_operation == 'closing':
        binary = closing(binary, kernel_size)

    # Perform connected component analysis
    labeled, num_components = connected_components_8(binary)

    # Filter by size
    if min_size > 0 or max_size is not None:
        binary_filtered, count = filter_components_by_size(
            labeled, num_components, min_size, max_size
        )
        return binary_filtered, count

    return binary, num_components


def visualize_results(original_img, binary_img, count, title=""):
    """Visualize original image, binary image, and component count."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if len(original_img.shape) == 3:
        axes[0].imshow(original_img)
    else:
        axes[0].imshow(original_img, cmap='gray')
    axes[0].set_title(f"Original Image\n{title}")
    axes[0].axis('off')

    axes[1].imshow(binary_img, cmap='gray')
    axes[1].set_title(f"Binary Image\nComponents Found: {count}")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


def process_image_pipeline(image_path, visualize=True, **kwargs):
    """Complete pipeline: load image, count components, and visualize."""

    img = convert_to_matrix(image_path)
    binary_img, count = count_connected_components(img, **kwargs)

    if visualize:
        visualize_results(img, binary_img, count, title=image_path)

    return binary_img, count


# Example usage
if __name__ == "__main__":
    # For bird images
    birds = ["./task2_images/birds1.jpg", "./task2_images/birds2.jpg", "./task2_images/birds3.jpg"]
    for bird_image in birds:
        binary, count = process_image_pipeline(
            bird_image,
            threshold=150,
            morph_operation='opening',
            kernel_size=1,
            min_size=20,
            visualize=False
        )
        print(f"{bird_image}: {count} birds")

    # For dice images - dots are small and need different parameters
    dice = ["./task2_images/dice5.PNG", "./task2_images/dice6.PNG"]
    for dice_image in dice:
        binary, count = process_image_pipeline(
            dice_image,
            threshold=100,
            morph_operation='opening',
            kernel_size=2,
            min_size=100,
            max_size=1000,
            visualize=False
        )
        print(f"{dice_image}: {count} dots")

    binary, count = process_image_pipeline(
        "task2_images/demo4.png",
        threshold=140,
        morph_operation='opening',
        kernel_size=4,
        min_size=2000,
        max_size=5000,
        visualize=False
    )
    print(f"./task2_images/demo4.png: {count} okey tiles")
