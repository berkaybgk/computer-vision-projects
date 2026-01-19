import numpy as np
import cv2
from pathlib import Path
from scipy.ndimage import map_coordinates

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
    norm_points1, T1 =  normalize_points(points_im1)
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


def warp(image, homography):
    """
    Warp image using homography with backward warping.

    Args:
        image: Input image to warp
        homography: 3x3 homography matrix

    Returns:
        Warped image
    """
    h, w = image.shape[:2]

    # Find corners of image after warping
    corners = np.array([
        [0, 0],
        [w-1, 0],
        [w-1, h-1],
        [0, h-1]
    ], dtype=np.float32)

    corners_hom = np.hstack([corners, np.ones((4, 1))])
    transformed = (homography @ corners_hom.T).T
    transformed = transformed / transformed[:, 2:3]

    # Calculate output image bounds
    min_x, max_x = transformed[:, 0].min(), transformed[:, 0].max()
    min_y, max_y = transformed[:, 1].min(), transformed[:, 1].max()

    # Create output image size
    out_w = int(np.ceil(max_x - min_x))
    out_h = int(np.ceil(max_y - min_y))

    # Translation to move to positive coordinates
    T_offset = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1]
    ], dtype=np.float32)

    H_full = T_offset @ homography
    H_inv = np.linalg.inv(H_full)

    # Create coordinate grid
    y_coords, x_coords = np.mgrid[0:out_h, 0:out_w]
    ones = np.ones_like(x_coords)
    coords_homogeneous = np.stack([x_coords.ravel(), y_coords.ravel(), ones.ravel()])

    # Backward warping
    src_coords = H_inv @ coords_homogeneous
    src_x = (src_coords[0] / src_coords[2]).reshape(out_h, out_w)
    src_y = (src_coords[1] / src_coords[2]).reshape(out_h, out_w)

    # Interpolate
    if len(image.shape) == 3:
        warped = np.zeros((out_h, out_w, image.shape[2]), dtype=image.dtype)
        for c in range(image.shape[2]):
            warped[:, :, c] = map_coordinates(image[:, :, c], [src_y, src_x],
                                              order=1, mode='constant', cval=0)
    else:
        warped = map_coordinates(image, [src_y, src_x],
                                 order=1, mode='constant', cval=0)

    return warped.astype(image.dtype)


def blend_images_efficient(images, homographies, scale_factor=0.5):
    """
    Efficiently blend multiple images using maximum intensity.
    """
    # Scale down images
    scaled_images = []
    scaled_homographies = []

    scale_matrix = np.array([
        [scale_factor, 0, 0],
        [0, scale_factor, 0],
        [0, 0, 1]
    ])

    for img, H in zip(images, homographies):
        new_size = (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))
        scaled_img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        scaled_images.append(scaled_img)
        scaled_H = scale_matrix @ H @ np.linalg.inv(scale_matrix)
        scaled_homographies.append(scaled_H)

    # Warp all images
    warped_images = []
    for i, (img, H) in enumerate(zip(scaled_images, scaled_homographies)):
        print(f"  Warping image {i+1}/{len(scaled_images)}...")
        warped = warp(img, H)
        warped_images.append(warped)

    # Find global bounds for all warped images
    min_x, min_y = 0, 0
    max_x, max_y = 0, 0

    offsets = []
    for i, (img, H) in enumerate(zip(scaled_images, scaled_homographies)):
        h, w = img.shape[:2]
        corners = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
        corners_hom = np.hstack([corners, np.ones((4, 1))])
        transformed = (H @ corners_hom.T).T
        transformed = transformed / transformed[:, 2:3]

        offset_x = transformed[:, 0].min()
        offset_y = transformed[:, 1].min()
        offsets.append((offset_x, offset_y))

        min_x = min(min_x, offset_x)
        max_x = max(max_x, transformed[:, 0].max())
        min_y = min(min_y, offset_y)
        max_y = max(max_y, transformed[:, 1].max())

    # Create canvas
    canvas_w = int(np.ceil(max_x - min_x)) + 10
    canvas_h = int(np.ceil(max_y - min_y)) + 10
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    print(f"  Canvas size: {canvas_w}x{canvas_h}")

    # Place warped images on canvas
    for i, (warped, (ox, oy)) in enumerate(zip(warped_images, offsets)):
        y_start = int(oy - min_y) + 5
        x_start = int(ox - min_x) + 5
        y_end = y_start + warped.shape[0]
        x_end = x_start + warped.shape[1]

        mask = np.any(warped > 0, axis=2)
        canvas[y_start:y_end, x_start:x_end][mask] = np.maximum(
            canvas[y_start:y_end, x_start:x_end][mask],
            warped[mask]
        )

    # Scale back up
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
        # Left-to-right: base is left_2, sequential chaining
        H_l2 = np.eye(3)  # Base image

        # left_1 -> left_2
        H_l1_to_l2 = compute_func(pts['left_1_left'], pts['left_2_right'])
        H_l1 = H_l1_to_l2

        # middle -> mosaic_1 (left_2 + left_1)
        H_m_to_l1 = compute_func(pts['middle_left'], pts['left_1_right'])
        H_m = H_l1 @ H_m_to_l1

        # right_1 -> mosaic_2 (left_2 + left_1 + middle)
        H_r1_to_m = compute_func(pts['right_1_left'], pts['middle_right'])
        H_r1 = H_m @ H_r1_to_m

        # right_2 -> mosaic_3 (left_2 + left_1 + middle + right_1)
        H_r2_to_r1 = compute_func(pts['right_2_left'], pts['right_1_right'])
        H_r2 = H_r1 @ H_r2_to_r1

        homographies = [H_l2, H_l1, H_m, H_r1, H_r2]
        image_list = [images['left_2'], images['left_1'], images['middle'],
                      images['right_1'], images['right_2']]

    elif method == 'middle_out':
        # Middle-out: Stitch inner 3 first (left_1, middle, right_1), then add outer 2
        # mosaic_1 = left_1 + middle + right_1 (base: middle)
        # mosaic_final = mosaic_1 + left_2 + right_2

        H_m = np.eye(3)  # middle is base

        # First layer: left_1 and right_1 relative to middle
        H_l1 = compute_func(pts['left_1_right'], pts['middle_left'])
        H_r1 = compute_func(pts['right_1_left'], pts['middle_right'])

        # Second layer: left_2 and right_2 relative to mosaic_1
        # mosaic_1 has middle's coordinate system, so chain through l1/r1
        H_l2 = H_l1 @ compute_func(pts['left_2_right'], pts['left_1_left'])
        H_r2 = H_r1 @ compute_func(pts['right_2_left'], pts['right_1_right'])

        homographies = [H_l2, H_l1, H_m, H_r1, H_r2]
        image_list = [images['left_2'], images['left_1'], images['middle'],
                      images['right_1'], images['right_2']]

    else:  # first_out_then_middle
        # First-out-then-middle: Create left pair and right pair, then bridge with middle
        # mosaic_left = left_1 + left_2 (base: left_1)
        # mosaic_right = right_1 + right_2 (base: right_1)
        # mosaic_final = mosaic_left + middle + mosaic_right (base: middle)

        H_m = np.eye(3)  # middle is final base

        # Left pair relative to left_1, then left_1 to middle
        H_l1_to_m = compute_func(pts['left_1_right'], pts['middle_left'])
        H_l1 = H_l1_to_m
        H_l2_to_l1 = compute_func(pts['left_2_right'], pts['left_1_left'])
        H_l2 = H_l1_to_m @ H_l2_to_l1

        # Right pair relative to right_1, then right_1 to middle
        H_r1_to_m = compute_func(pts['right_1_left'], pts['middle_right'])
        H_r1 = H_r1_to_m
        H_r2_to_r1 = compute_func(pts['right_2_left'], pts['right_1_right'])
        H_r2 = H_r1_to_m @ H_r2_to_r1

        homographies = [H_l2, H_l1, H_m, H_r1, H_r2]
        image_list = [images['left_2'], images['left_1'], images['middle'],
                      images['right_1'], images['right_2']]

    # Blend all images at once
    result = blend_images_efficient(image_list, homographies, scale_factor=0.5)

    return result

def main():
    base_dir = Path('.')

    # Task 1: Paris images
    print("\n" + "="*60)
    print("Stitching Paris images...")
    print("="*60)
    paris_result = stitch_paris(
        base_dir / 'images' / 'paris',
        base_dir / 'points' / 'paris',
        normalize=True
    )
    cv2.imwrite('normalized/paris_panorama.jpg', paris_result)
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
    cv2.imwrite('normalized/north_campus_left_to_right.jpg', nc_lr)
    print("  Saved!\n")

    print("  Method: Middle out...")
    nc_mo = stitch_five_images_direct(
        base_dir / 'images' / 'north_campus',
        base_dir / 'points' / 'north_campus',
        method='middle_out',
        normalize=True
    )
    cv2.imwrite('normalized/north_campus_middle_out.jpg', nc_mo)
    print("  Saved!\n")

    print("  Method: First out then middle...")
    nc_fo = stitch_five_images_direct(
        base_dir / 'images' / 'north_campus',
        base_dir / 'points' / 'north_campus',
        method='first_out_then_middle',
        normalize=True
    )
    cv2.imwrite('normalized/north_campus_first_out_then_middle.jpg', nc_fo)
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
    cv2.imwrite('normalized/cmpe_building_left_to_right.jpg', cmpe_lr)
    print("  Saved!\n")

    print("  Method: Middle out...")
    cmpe_mo = stitch_five_images_direct(
        base_dir / 'images' / 'cmpe_building',
        base_dir / 'points' / 'cmpe_building',
        method='middle_out',
        normalize=True
    )
    cv2.imwrite('normalized/cmpe_building_middle_out.jpg', cmpe_mo)
    print("  Saved!\n")

    print("  Method: First out then middle...")
    cmpe_fo = stitch_five_images_direct(
        base_dir / 'images' / 'cmpe_building',
        base_dir / 'points' / 'cmpe_building',
        method='first_out_then_middle',
        normalize=True
    )
    cv2.imwrite('normalized/cmpe_building_first_out_then_middle.jpg', cmpe_fo)
    print("  Saved!\n")

    print("="*60)
    print("All panoramas created successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
