import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

class PointCorrespondenceChecker:
    def __init__(self, img1, img2, pts1, pts2, title1, title2):
        self.img1 = img1
        self.img2 = img2
        self.pts1 = pts1
        self.pts2 = pts2
        self.title1 = title1
        self.title2 = title2
        self.selected_order = []

    def visualize_current(self):
        """Show current point correspondences"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        ax1.imshow(cv2.cvtColor(self.img1, cv2.COLOR_BGR2RGB))
        ax1.scatter(self.pts1[:, 0], self.pts1[:, 1], c='red', s=150, edgecolors='white', linewidths=2)
        for i, pt in enumerate(self.pts1):
            ax1.text(pt[0]+20, pt[1]-20, str(i), color='white', fontsize=16,
                     fontweight='bold', bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
        ax1.set_title(f'{self.title1} ({len(self.pts1)} points)', fontsize=14, fontweight='bold')
        ax1.axis('off')

        ax2.imshow(cv2.cvtColor(self.img2, cv2.COLOR_BGR2RGB))
        ax2.scatter(self.pts2[:, 0], self.pts2[:, 1], c='blue', s=150, edgecolors='white', linewidths=2)
        for i, pt in enumerate(self.pts2):
            ax2.text(pt[0]+20, pt[1]-20, str(i), color='white', fontsize=16,
                     fontweight='bold', bbox=dict(boxstyle='round', facecolor='blue', alpha=0.7))
        ax2.set_title(f'{self.title2} ({len(self.pts2)} points)', fontsize=14, fontweight='bold')
        ax2.axis('off')

        plt.suptitle('Point Correspondences - Same number should be same feature!',
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        return fig

    def show_side_by_side_correspondence(self):
        """Show what each point should match"""
        fig, axes = plt.subplots(len(self.pts1), 2, figsize=(12, 3*len(self.pts1)))

        if len(self.pts1) == 1:
            axes = axes.reshape(1, -1)

        for i in range(len(self.pts1)):
            # Left image - show point i in red
            ax1 = axes[i, 0]
            ax1.imshow(cv2.cvtColor(self.img1, cv2.COLOR_BGR2RGB))
            ax1.scatter(self.pts1[i, 0], self.pts1[i, 1], c='red', s=300, marker='*', edgecolors='yellow', linewidths=3)
            # Show other points dimmed
            other_pts = np.delete(self.pts1, i, axis=0)
            ax1.scatter(other_pts[:, 0], other_pts[:, 1], c='red', s=50, alpha=0.3)
            ax1.set_title(f'{self.title1} - Point {i}', fontsize=12, fontweight='bold')
            ax1.axis('off')

            # Right image - show point i in blue
            ax2 = axes[i, 1]
            ax2.imshow(cv2.cvtColor(self.img2, cv2.COLOR_BGR2RGB))
            ax2.scatter(self.pts2[i, 0], self.pts2[i, 1], c='blue', s=300, marker='*', edgecolors='yellow', linewidths=3)
            # Show other points dimmed
            other_pts = np.delete(self.pts2, i, axis=0)
            ax2.scatter(other_pts[:, 0], other_pts[:, 1], c='blue', s=50, alpha=0.3)
            ax2.set_title(f'{self.title2} - Point {i} (should match left)', fontsize=12, fontweight='bold')
            ax2.axis('off')

        plt.suptitle('Point-by-Point Correspondence Check\nEach row should show the SAME feature on both sides!',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig


def check_and_suggest_reorder(pts1, pts2):
    """
    Suggest a reordering based on spatial proximity.
    This assumes points in pts1 are correct and tries to reorder pts2.
    """
    n = len(pts1)
    reordered_indices = []
    used = set()

    for i in range(n):
        pt1 = pts1[i]
        # Find closest unused point in pts2
        min_dist = float('inf')
        best_idx = -1

        for j in range(n):
            if j in used:
                continue
            dist = np.linalg.norm(pt1 - pts2[j])
            if dist < min_dist:
                min_dist = dist
                best_idx = j

        reordered_indices.append(best_idx)
        used.add(best_idx)

    return reordered_indices


def main():
    # Example for North Campus right_1 and right_2
    print("="*60)
    print("Point Correspondence Checker")
    print("="*60)

    # Load images
    img_r2 = cv2.imread('images/north_campus/right_2.jpg')
    img_r1 = cv2.imread('images/north_campus/right_1.jpg')

    # Load points
    pts_r2_left = np.load('points/north_campus/points_right_2_left.npy')
    pts_r1_right = np.load('points/north_campus/points_right_1_right.npy')

    print(f"\nright_2 LEFT edge: {len(pts_r2_left)} points")
    print(f"right_1 RIGHT edge: {len(pts_r1_right)} points")

    # Create checker
    checker = PointCorrespondenceChecker(
        img_r2, img_r1,
        pts_r2_left, pts_r1_right,
        'right_2 - LEFT edge', 'right_1 - RIGHT edge'
    )

    # Show current correspondence
    print("\n1. Showing current point correspondences...")
    fig1 = checker.visualize_current()
    plt.savefig('current_correspondences.png', dpi=150, bbox_inches='tight')
    print("   Saved: current_correspondences.png")

    # Show side-by-side what should match
    print("\n2. Showing point-by-point correspondence check...")
    fig2 = checker.show_side_by_side_correspondence()
    plt.savefig('point_by_point_check.png', dpi=150, bbox_inches='tight')
    print("   Saved: point_by_point_check.png")
    print("   CHECK: Does each row show the SAME feature on both sides?")

    # Suggest reordering
    print("\n3. Analyzing spatial correspondence...")
    suggested_order = check_and_suggest_reorder(pts_r2_left, pts_r1_right)
    print(f"\nCurrent order in right_1_right: [0, 1, 2, 3, 4]")
    print(f"Suggested reorder to match right_2_left: {suggested_order}")

    # Apply suggested reordering
    pts_r1_right_reordered = pts_r1_right[suggested_order]

    # Visualize suggested fix
    checker_fixed = PointCorrespondenceChecker(
        img_r2, img_r1,
        pts_r2_left, pts_r1_right_reordered,
        'right_2 - LEFT edge', 'right_1 - RIGHT edge (REORDERED)'
    )

    print("\n4. Showing suggested fix...")
    fig3 = checker_fixed.visualize_current()
    plt.savefig('suggested_fix.png', dpi=150, bbox_inches='tight')
    print("   Saved: suggested_fix.png")

    # Ask user if they want to apply the fix
    print("\n" + "="*60)
    print("Do you want to apply this reordering?")
    print("="*60)
    response = input("Type 'yes' to save the reordered points, or 'no' to cancel: ").lower().strip()

    if response == 'yes':
        np.save('points/north_campus/points_right_1_right.npy', pts_r1_right_reordered)
        print("✓ Saved reordered points to points/north_campus/points_right_1_right.npy")
        print("\nNow try running your main.py again!")
    else:
        print("No changes made.")
        print("\nIf the automatic reordering doesn't look right, you may need to:")
        print("1. Manually reorder the points, or")
        print("2. Re-select the points in the correct order")

    plt.show()


if __name__ == '__main__':
    main()