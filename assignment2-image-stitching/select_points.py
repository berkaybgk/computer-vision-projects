import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

# === CONFIGURATION ===
set_name = "paris"
num_images = 3
num_points = 5

# List images in LEFT-TO-RIGHT order
image_names = [
    "paris_a.jpg",
    "paris_b.jpg",
    "paris_c.jpg",
    # "left_2.jpg",
    # "left_1.jpg",
    # "middle.jpg",
    # "right_1.jpg",
    # "right_2.jpg",
]

output_dir = f"./points/{set_name}"
# =========================

# Ensure save directory exists
os.makedirs(output_dir, exist_ok=True)

# Load all images
print(f"Loading {num_images} images...")
images = []
for img_name in image_names:
    img_path = f"./images/{set_name}/{img_name}"
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    images.append(img)
    print(f"  Loaded: {img_name}")

# Dictionary to store all points for each image
# Key: image index, Value: dict with 'left' and 'right' point arrays
all_points = {i: {'left': None, 'right': None} for i in range(num_images)}

# Process each adjacent pair
for pair_idx in range(num_images - 1):
    left_img_idx = pair_idx
    right_img_idx = pair_idx + 1

    left_img = images[left_img_idx]
    right_img = images[right_img_idx]
    left_name = image_names[left_img_idx]
    right_name = image_names[right_img_idx]

    print(f"\n{'='*60}")
    print(f"PAIR {pair_idx + 1}/{num_images - 1}: {left_name} <--> {right_name}")
    print(f"{'='*60}")

    # Display the pair side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(left_img)
    axes[0].set_title(f"{left_name}\n(LEFT IMAGE)", fontsize=11, fontweight='bold')
    axes[0].axis("off")

    axes[1].imshow(right_img)
    axes[1].set_title(f"{right_name}\n(RIGHT IMAGE)", fontsize=11, fontweight='bold')
    axes[1].axis("off")

    plt.suptitle(f"PAIR {pair_idx + 1}: Select {num_points} corresponding points\n"
                 f"Click on LEFT image first, then on RIGHT image\n"
                 f"(Close window when done)",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()

    # Gather clicks for this pair
    points_left = []
    points_right = []

    for i in range(num_points):
        print(f"  Point pair {i+1}/{num_points}: Click on LEFT image first...")
        plt.sca(axes[0])
        pts_left = plt.ginput(1, timeout=0)

        print(f"  Point pair {i+1}/{num_points}: Now click on RIGHT image...")
        plt.sca(axes[1])
        pts_right = plt.ginput(1, timeout=0)

        points_left.append(pts_left[0])
        points_right.append(pts_right[0])

        # Draw the selected points for visual feedback
        axes[0].plot(pts_left[0][0], pts_left[0][1], 'r+', markersize=12, markeredgewidth=2)
        axes[1].plot(pts_right[0][0], pts_right[0][1], 'r+', markersize=12, markeredgewidth=2)
        axes[0].text(pts_left[0][0], pts_left[0][1], f' {i+1}', color='red', fontsize=10, fontweight='bold')
        axes[1].text(pts_right[0][0], pts_right[0][1], f' {i+1}', color='red', fontsize=10, fontweight='bold')
        plt.draw()

    plt.close(fig)

    points_left = np.array(points_left)
    points_right = np.array(points_right)

    # Store points in the dictionary
    # Left image's RIGHT correspondence points
    all_points[left_img_idx]['right'] = points_left
    # Right image's LEFT correspondence points
    all_points[right_img_idx]['left'] = points_right

    print(f"    Collected {num_points} point pairs for this connection")

# Save all points
print(f"\n{'='*60}")
print("SAVING POINTS...")
print(f"{'='*60}")

for img_idx in range(num_images):
    base_name = os.path.splitext(image_names[img_idx])[0]

    # For leftmost image: only has 'right' points
    if img_idx == 0:
        save_path = os.path.join(output_dir, f"points_{base_name}_right.npy")
        np.save(save_path, all_points[img_idx]['right'])
        print(f"Image {img_idx + 1} ({image_names[img_idx]}): RIGHT points → {save_path}")

    # For rightmost image: only has 'left' points
    elif img_idx == num_images - 1:
        save_path = os.path.join(output_dir, f"points_{base_name}_left.npy")
        np.save(save_path, all_points[img_idx]['left'])
        print(f"Image {img_idx + 1} ({image_names[img_idx]}): LEFT points → {save_path}")

    # For middle images: has both 'left' and 'right' points
    else:
        save_path_left = os.path.join(output_dir, f"points_{base_name}_left.npy")
        save_path_right = os.path.join(output_dir, f"points_{base_name}_right.npy")
        np.save(save_path_left, all_points[img_idx]['left'])
        np.save(save_path_right, all_points[img_idx]['right'])
        print(f"Image {img_idx + 1} ({image_names[img_idx]}):")
        print(f"  LEFT points  -> {save_path_left}")
        print(f"  RIGHT points -> {save_path_right}")

print(f"\n{'='*60}")
print("ALL POINTS SAVED SUCCESSFULLY!")
print(f"{'='*60}")
print(f"\nPoint files saved to: {output_dir}")
print("\nFile naming convention:")
print("  - Leftmost image: *_right.npy (connects to its right neighbor)")
print("  - Middle images: *_left.npy and *_right.npy (connects to both neighbors)")
print("  - Rightmost image: *_left.npy (connects to its left neighbor)")
print("\nYou can now load these in your stitching code using np.load()")
