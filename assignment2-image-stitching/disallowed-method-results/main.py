import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

def computeH(points_im1, points_im2):
    """
    Compute homography matrix from corresponding points.

    Args:
        points_im1: Nx2 array of points from image 1
        points_im2: Nx2 array of corresponding points from image 2

    Returns:
        3x3 homography matrix
    """
    n = points_im1.shape[0]

    # Build matrix A for homogeneous linear system
    A = []
    for i in range(n):
        x, y = points_im1[i]
        x_prime, y_prime = points_im2[i]

        A.append([-x, -y, -1, 0, 0, 0, x*x_prime, y*x_prime, x_prime])
        A.append([0, 0, 0, -x, -y, -1, x*y_prime, y*y_prime, y_prime])

    A = np.array(A)

    # Solve using SVD
    U, S, Vt = np.linalg.svd(A)

    # Last row of V (last column of Vt) gives solution
    H = Vt[-1].reshape(3, 3)

    # Normalize so that H[2,2] = 1
    H = H / H[2, 2]

    return H


def normalize_points(points):
    """
    Normalize points: move average to (0,0) and scale so average distance is sqrt(2).

    Args:
        points: Nx2 array of points

    Returns:
        normalized_points: Nx2 array of normalized points
        T: 3x3 transformation matrix used for normalization
    """
    # Compute centroid
    centroid = np.mean(points, axis=0)

    # Center the points
    centered = points - centroid

    # Compute average distance from origin
    avg_dist = np.mean(np.sqrt(np.sum(centered**2, axis=1)))

    # Scale factor to make average distance sqrt(2)
    scale = np.sqrt(2) / avg_dist

    # Create transformation matrix
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])

    # Apply transformation
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    normalized_homogeneous = (T @ points_homogeneous.T).T
    normalized_points = normalized_homogeneous[:, :2]

    return normalized_points, T


def computeH_normalized(points_im1, points_im2):
    """
    Compute homography with point normalization for numerical stability.
    """
    # Normalize both sets of points
    norm_points1, T1 = normalize_points(points_im1)
    norm_points2, T2 = normalize_points(points_im2)

    # Compute homography on normalized points
    H_norm = computeH(norm_points1, norm_points2)

    # Denormalize: H = T2^(-1) * H_norm * T1
    H = np.linalg.inv(T2) @ H_norm @ T1

    return H


def get_canvas_bounds(images, homographies):
    """
    Calculate the bounds needed to fit all warped images.
    Returns min/max coordinates instead of creating full canvas.
    """
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    for img, H in zip(images, homographies):
        h, w = img.shape[:2]
        corners = np.array([
            [0, 0],
            [w-1, 0],
            [w-1, h-1],
            [0, h-1]
        ], dtype=np.float32)

        # Transform corners
        corners_hom = np.hstack([corners, np.ones((4, 1))])
        transformed = (H @ corners_hom.T).T
        transformed = transformed / transformed[:, 2:3]

        min_x = min(min_x, transformed[:, 0].min())
        max_x = max(max_x, transformed[:, 0].max())
        min_y = min(min_y, transformed[:, 1].min())
        max_y = max(max_y, transformed[:, 1].max())

    return min_x, max_x, min_y, max_y


def warp_image_to_canvas(image, homography, canvas_shape, offset):
    """
    Warp a single image to canvas coordinates efficiently.
    """
    out_h, out_w = canvas_shape

    # Create offset transformation
    T_offset = np.array([
        [1, 0, offset[1]],
        [0, 1, offset[0]],
        [0, 0, 1]
    ], dtype=np.float32)

    # Combine transformations
    H_full = T_offset @ homography

    # Use cv2.warpPerspective for efficiency
    warped = cv2.warpPerspective(image, H_full, (out_w, out_h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=0)

    return warped


def blend_images_efficient(images, homographies, scale_factor=0.5):
    """
    Efficiently blend multiple images using maximum intensity.
    Works at reduced resolution to save memory.

    Args:
        images: List of images to blend
        homographies: List of homography matrices for each image
        scale_factor: Scale down factor to reduce memory usage
    """
    # Scale down images for processing
    scaled_images = []
    scaled_homographies = []

    scale_matrix = np.array([
        [scale_factor, 0, 0],
        [0, scale_factor, 0],
        [0, 0, 1]
    ])

    for img, H in zip(images, homographies):
        # Resize image
        new_size = (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))
        scaled_img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        scaled_images.append(scaled_img)

        # Scale homography
        scaled_H = scale_matrix @ H @ np.linalg.inv(scale_matrix)
        scaled_homographies.append(scaled_H)

    # Get canvas bounds
    min_x, max_x, min_y, max_y = get_canvas_bounds(scaled_images, scaled_homographies)

    # Calculate canvas size with some padding
    canvas_w = int(np.ceil(max_x - min_x)) + 10
    canvas_h = int(np.ceil(max_y - min_y)) + 10
    offset = (-min_y + 5, -min_x + 5)

    print(f"  Canvas size: {canvas_w}x{canvas_h} (scaled)")

    # Initialize canvas
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # Warp and blend each image
    for i, (img, H) in enumerate(zip(scaled_images, scaled_homographies)):
        print(f"  Processing image {i+1}/{len(scaled_images)}...")

        warped = warp_image_to_canvas(img, H, (canvas_h, canvas_w), offset)

        # Create mask for valid pixels
        mask = np.any(warped > 0, axis=2)

        # Blend using maximum intensity
        canvas[mask] = np.maximum(canvas[mask], warped[mask])

    # Scale back up to original resolution
    final_size = (int(canvas_w / scale_factor), int(canvas_h / scale_factor))
    result = cv2.resize(canvas, final_size, interpolation=cv2.INTER_CUBIC)

    return result


def stitch_paris(base_path, points_path, normalize=False):
    """
    Stitch 3 Paris images with paris_b as base.
    """
    # Load images
    img_a = cv2.imread(str(base_path / 'paris_a.jpg'))
    img_b = cv2.imread(str(base_path / 'paris_b.jpg'))
    img_c = cv2.imread(str(base_path / 'paris_c.jpg'))

    # Load points
    pts_a_right = np.load(points_path / 'points_paris_a_right.npy')
    pts_b_left = np.load(points_path / 'points_paris_b_left.npy')
    pts_b_right = np.load(points_path / 'points_paris_b_right.npy')
    pts_c_left = np.load(points_path / 'points_paris_c_left.npy')

    # Compute homographies (base is paris_b, so identity for it)
    if normalize:
        H_a = computeH_normalized(pts_a_right, pts_b_left)
        H_c = computeH_normalized(pts_c_left, pts_b_right)
    else:
        H_a = computeH(pts_a_right, pts_b_left)
        H_c = computeH(pts_c_left, pts_b_right)

    H_b = np.eye(3)

    # Blend images
    images = [img_a, img_b, img_c]
    homographies = [H_a, H_b, H_c]

    result = blend_images_efficient(images, homographies, scale_factor=1.0)

    return result


def stitch_five_images_direct(base_path, points_path, method='left_to_right', normalize=False):
    """
    Stitch 5 images directly by computing all homographies relative to final canvas.
    This avoids creating huge intermediate mosaics.
    """
    # Load images
    images = {}
    for name in ['left_2', 'left_1', 'middle', 'right_1', 'right_2']:
        img = cv2.imread(str(base_path / f'{name}.jpg'))
        # Reduce image size if very large
        if img.shape[0] > 2000 or img.shape[1] > 2000:
            scale = min(2000 / img.shape[0], 2000 / img.shape[1])
            new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        images[name] = img

    # Load all points
    pts = {}
    for name in ['left_1', 'left_2', 'middle', 'right_1', 'right_2']:
        for side in ['left', 'right']:
            key = f'points_{name}_{side}.npy'
            try:
                pts[f'{name}_{side}'] = np.load(points_path / key)
            except:
                pass

    compute_func = computeH_normalized if normalize else computeH

    if method == 'left_to_right':
        # Left-to-right: base is left_2, each image maps to the one on its left
        # The assignment says: "Stitch left_2 and left_1 by taking left_2 as base"
        # So we want left_1 points to map to left_2 coordinate system

        H_l2 = np.eye(3)  # Base image

        # left_1 -> left_2: map left_1's left edge to left_2's right edge
        H_l1_to_l2 = compute_func(pts['left_1_left'], pts['left_2_right'])
        H_l1 = H_l1_to_l2

        # middle -> mosaic_1 (which has left_1's coordinate system)
        # Map middle's left edge to left_1's right edge, then transform to left_2
        H_m_to_l1 = compute_func(pts['middle_left'], pts['left_1_right'])
        H_m = H_l1 @ H_m_to_l1

        # right_1 -> mosaic_2 (which has middle's coordinate system)
        H_r1_to_m = compute_func(pts['right_1_left'], pts['middle_right'])
        H_r1 = H_m @ H_r1_to_m

        # right_2 -> mosaic_3 (which has right_1's coordinate system)
        H_r2_to_r1 = compute_func(pts['right_2_left'], pts['right_1_right'])
        H_r2 = H_r1 @ H_r2_to_r1

        # Debug: Check if any homography is degenerate
        det_r2 = np.linalg.det(H_r2[:2, :2])
        print(f"  H_r2 determinant (2x2): {det_r2:.4f}")
        if abs(det_r2) < 0.01 or abs(det_r2) > 100:
            print(f"  WARNING: H_r2 might be degenerate!")
            print(f"  H_r2:\n{H_r2}")

        homographies = [H_l2, H_l1, H_m, H_r1, H_r2]
        image_list = [images['left_2'], images['left_1'], images['middle'],
                      images['right_1'], images['right_2']]

    elif method == 'middle_out':
        # All homographies relative to middle
        H_m = np.eye(3)
        H_l1 = compute_func(pts['left_1_right'], pts['middle_left'])
        H_r1 = compute_func(pts['right_1_left'], pts['middle_right'])
        H_l2 = H_l1 @ compute_func(pts['left_2_right'], pts['left_1_left'])
        H_r2 = H_r1 @ compute_func(pts['right_2_left'], pts['right_1_right'])

        homographies = [H_l2, H_l1, H_m, H_r1, H_r2]
        image_list = [images['left_2'], images['left_1'], images['middle'],
                      images['right_1'], images['right_2']]

    else:  # first_out_then_middle
        # All homographies relative to middle
        H_m = np.eye(3)
        H_l1 = compute_func(pts['left_1_right'], pts['middle_left'])
        H_r1 = compute_func(pts['right_1_left'], pts['middle_right'])
        H_l2 = H_l1 @ compute_func(pts['left_2_right'], pts['left_1_left'])
        H_r2 = H_r1 @ compute_func(pts['right_2_left'], pts['right_1_right'])

        homographies = [H_l2, H_l1, H_m, H_r1, H_r2]
        image_list = [images['left_2'], images['left_1'], images['middle'],
                      images['right_1'], images['right_2']]

    # Blend all images at once
    result = blend_images_efficient(image_list, homographies, scale_factor=0.5)

    return result


def check_point_correspondences(base_path, points_path, dataset_name):
    """
    Check if point correspondences make sense by computing homographies
    and checking for degeneracies.
    """
    print(f"\n=== Checking {dataset_name} point correspondences ===")

    pts = {}
    for name in ['left_1', 'left_2', 'middle', 'right_1', 'right_2']:
        for side in ['left', 'right']:
            key = f'points_{name}_{side}.npy'
            try:
                pts[f'{name}_{side}'] = np.load(points_path / key)
                print(f"  {key}: {len(pts[f'{name}_{side}'])} points")
            except:
                pass

    # Check each pair
    pairs = [
        ('left_1_left', 'left_2_right', 'left_1 -> left_2'),
        ('middle_left', 'left_1_right', 'middle -> left_1'),
        ('right_1_left', 'middle_right', 'right_1 -> middle'),
        ('right_2_left', 'right_1_right', 'right_2 -> right_1'),
    ]

    for key1, key2, desc in pairs:
        if key1 in pts and key2 in pts:
            H = computeH_normalized(pts[key1], pts[key2])
            det = np.linalg.det(H[:2, :2])
            print(f"  {desc}: det={det:.4f}, H33={H[2,2]:.4f}")

            if abs(det) < 0.01:
                print(f"    WARNING: Very small determinant!")
            if abs(det) > 100:
                print(f"    WARNING: Very large determinant!")
            if abs(H[2,2]) < 0.01:
                print(f"    WARNING: H[2,2] close to zero!")


def main():
    base_dir = Path('.')

    # Check point correspondences first
    check_point_correspondences(
        base_dir / 'images' / 'north_campus',
        base_dir / 'points' / 'north_campus',
        'North Campus'
    )

    # Task 1: Paris images
    print("\n" + "="*60)
    print("Stitching Paris images...")
    print("="*60)
    paris_result = stitch_paris(
        base_dir / 'images' / 'paris',
        base_dir / 'points' / 'paris',
        normalize=True
    )
    cv2.imwrite('paris_panorama.jpg', paris_result)
    print("Paris panorama saved!\n")

    # Task 2: North Campus
    print("="*60)
    print("Stitching North Campus images...")
    print("="*60)

    print("\n  Method: Left to right...")
    nc_lr = stitch_five_images_direct(
        base_dir / 'images' / 'north_campus',
        base_dir / 'points' / 'north_campus',
        method='left_to_right',
        normalize=True
    )
    cv2.imwrite('north_campus_left_to_right.jpg', nc_lr)
    print("  Saved!\n")

    print("  Method: Middle out...")
    nc_mo = stitch_five_images_direct(
        base_dir / 'images' / 'north_campus',
        base_dir / 'points' / 'north_campus',
        method='middle_out',
        normalize=True
    )
    cv2.imwrite('north_campus_middle_out.jpg', nc_mo)
    print("  Saved!\n")

    print("  Method: First out then middle...")
    nc_fo = stitch_five_images_direct(
        base_dir / 'images' / 'north_campus',
        base_dir / 'points' / 'north_campus',
        method='first_out_then_middle',
        normalize=True
    )
    cv2.imwrite('north_campus_first_out_then_middle.jpg', nc_fo)
    print("  Saved!\n")

    # Task 3: CMPE Building
    print("="*60)
    print("Stitching CMPE Building images...")
    print("="*60)

    print("\n  Method: Left to right...")
    cmpe_lr = stitch_five_images_direct(
        base_dir / 'images' / 'cmpe_building',
        base_dir / 'points' / 'cmpe_building',
        method='left_to_right',
        normalize=True
    )
    cv2.imwrite('cmpe_building_left_to_right.jpg', cmpe_lr)
    print("  Saved!\n")

    print("  Method: Middle out...")
    cmpe_mo = stitch_five_images_direct(
        base_dir / 'images' / 'cmpe_building',
        base_dir / 'points' / 'cmpe_building',
        method='middle_out',
        normalize=True
    )
    cv2.imwrite('cmpe_building_middle_out.jpg', cmpe_mo)
    print("  Saved!\n")

    print("  Method: First out then middle...")
    cmpe_fo = stitch_five_images_direct(
        base_dir / 'images' / 'cmpe_building',
        base_dir / 'points' / 'cmpe_building',
        method='first_out_then_middle',
        normalize=True
    )
    cv2.imwrite('cmpe_building_first_out_then_middle.jpg', cmpe_fo)
    print("  Saved!\n")

    print("="*60)
    print("All panoramas created successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
