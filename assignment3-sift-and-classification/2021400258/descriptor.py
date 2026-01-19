import cv2
import numpy as np
import torch
import torchvision.models as models
from pathlib import Path
from PIL import Image


class DescriptorExtractor:
    """
    Extract SIFT and Swin-B descriptors from images.
    """

    def __init__(self, data_dir='data', resize_dim=None):
        """
        Initialize the descriptor extractor.

        Args:
            data_dir: Path to the data directory
            resize_dim: Tuple (width, height) to resize images, or None to keep original size
        """
        self.data_dir = Path(data_dir)
        self.images_dir = self.data_dir / 'Images'
        self.sift_dir = self.data_dir / 'descriptors' / 'sift'
        self.swin_dir = self.data_dir / 'descriptors' / 'swin'
        self.resize_dim = resize_dim

        # Create directories if they don't exist
        self.sift_dir.mkdir(parents=True, exist_ok=True)
        self.swin_dir.mkdir(parents=True, exist_ok=True)

        # Initialize SIFT detector
        self.sift = cv2.SIFT_create()

        # Initialize Swin-B model
        self.device = torch.device('cpu')

        # Load pretrained Swin-B model
        weights = models.Swin_B_Weights.IMAGENET1K_V1
        swin_full_model = models.swin_b(weights=weights)

        # Extract only the feature extraction layers (exclude avgpool and head)
        # The Swin-B architecture: features -> norm -> permute -> avgpool -> flatten -> head
        # We want just the 'features' part
        self.swin_features = swin_full_model.features
        self.swin_norm = swin_full_model.norm  # Layer norm after features

        # Set to evaluation mode
        self.swin_features.eval()
        self.swin_norm.eval()
        self.swin_features.to(self.device)
        self.swin_norm.to(self.device)

        # Get the preprocessing transform from the weights
        self.swin_transform = weights.transforms()

        print(f"SIFT descriptor initialized")
        print(f"Swin-B model loaded with IMAGENET1K_V1 weights")

    def extract_sift_descriptors(self, image_path):
        """
        Extract SIFT descriptors from an image.

        SIFT works on grayscale images because it detects keypoints based on
        intensity gradients (brightness changes), not color information.

        Args:
            image_path: Path to the image file

        Returns:
            descriptors: Array of shape (n_keypoints, 128)
        """
        # Read image in grayscale (SIFT only uses intensity/brightness information)
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        # Resize if specified
        if self.resize_dim is not None:
            img = cv2.resize(img, self.resize_dim)

        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.sift.detectAndCompute(img, None)

        if descriptors is None:
            # Return empty array if no keypoints detected
            descriptors = np.array([]).reshape(0, 128)

        return descriptors

    def extract_swin_descriptors(self, image_path, pooling_method='avg'):
        """
        Extract Swin-B descriptors from an image.

        Args:
            image_path: Path to the image file
            pooling_method: 'avg', 'max', or 'flatten'
                - 'avg': Global average pooling (compact, ~1024-dim)
                - 'max': Global max pooling (compact, ~1024-dim)
                - 'flatten': Flatten spatial features (large, ~50k-dim)

        Returns:
            descriptors: Feature vector from Swin-B model
        """
        # Load image (Swin-B expects RGB color images)
        img = Image.open(image_path).convert('RGB')

        # Resize if specified
        if self.resize_dim is not None:
            img = img.resize(self.resize_dim, Image.BILINEAR)

        # Apply preprocessing transform (normalization, etc.)
        img_tensor = self.swin_transform(img)
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        # Extract features
        with torch.no_grad():
            # Pass through feature extraction layers
            features = self.swin_features(img_tensor)
            # features shape: (B, H, W, C) in torchvision Swin implementation
            # For 224x224 input: (1, 7, 7, 1024)

            # Apply normalization
            features = self.swin_norm(features)
            # Still (1, 7, 7, 1024)

            # Permute to (B, C, H, W) for pooling operations
            features = features.permute(0, 3, 1, 2)
            # Now: (1, 1024, 7, 7)

            # Convert spatial feature map to single vector
            if pooling_method == 'avg':
                # Global average pooling: average across spatial dimensions (H, W)
                features = torch.nn.functional.adaptive_avg_pool2d(features, (1, 1))
                # Shape: (1, 1024, 1, 1)
                features = features.squeeze()  # Shape: (1024,)
            elif pooling_method == 'max':
                # Global max pooling: max across spatial dimensions
                features = torch.nn.functional.adaptive_max_pool2d(features, (1, 1))
                # Shape: (1, 1024, 1, 1)
                features = features.squeeze()  # Shape: (1024,)
            elif pooling_method == 'flatten':
                # Flatten: concatenate all spatial features
                features = features.flatten()  # Shape: (1024*7*7=50176,)
            else:
                raise ValueError(f"Unknown pooling method: {pooling_method}")

        return features.cpu().numpy()

    def process_image_list(self, image_list_file, descriptor_type='both'):
        """
        Process a list of images and extract descriptors.

        Args:
            image_list_file: Path to text file containing image filenames
            descriptor_type: 'sift', 'swin', or 'both'
        """
        # Read image list
        with open(image_list_file, 'r') as f:
            image_names = [line.strip() for line in f if line.strip()]

        print(f"Processing {len(image_names)} images from {image_list_file}")

        for img_name in image_names:
            img_path = self.images_dir / img_name

            if not img_path.exists():
                print(f"Warning: Image not found: {img_path}")
                continue

            # Create output filename (remove extension, add .npy)
            base_name = Path(img_name).stem

            try:
                # Extract SIFT descriptors
                if descriptor_type in ['sift', 'both']:
                    sift_desc = self.extract_sift_descriptors(img_path)
                    sift_output = self.sift_dir / f"{base_name}.npy"
                    np.save(sift_output, sift_desc)

                # Extract Swin-B descriptors
                if descriptor_type in ['swin', 'both']:
                    swin_desc = self.extract_swin_descriptors(img_path)
                    swin_output = self.swin_dir / f"{base_name}.npy"
                    np.save(swin_output, swin_desc)

            except Exception as e:
                print(f"Error processing {img_name}: {str(e)}")

    def extract_all_descriptors(self):
        """
        Extract descriptors for both train and test images.
        """
        train_file = self.data_dir / 'TrainImages.txt'
        test_file = self.data_dir / 'TestImages.txt'

        if train_file.exists():
            print("\n" + "="*60)
            print("Extracting descriptors for training images")
            print("="*60)
            self.process_image_list(train_file, descriptor_type='both')
        else:
            print(f"Warning: {train_file} not found")

        if test_file.exists():
            print("\n" + "="*60)
            print("Extracting descriptors for test images")
            print("="*60)
            self.process_image_list(test_file, descriptor_type='both')
        else:
            print(f"Warning: {test_file} not found")

        print("\n" + "="*60)
        print("Descriptor extraction completed!")
        print(f"SIFT descriptors saved in: {self.sift_dir}")
        print(f"Swin-B descriptors saved in: {self.swin_dir}")
        print("="*60)


def main():
    """
    Main function to run descriptor extraction.
    """
    # Option 1: Extract descriptors at original image size
    extractor = DescriptorExtractor(data_dir='../data', resize_dim=None)

    # Option 2: Extract by resizing images
    # Uncomment the line below and comment the line above
    # extractor = DescriptorExtractor(data_dir='../data', resize_dim=(224, 224))

    # Extract all descriptors
    extractor.extract_all_descriptors()

    # Count files
    sift_files = list(extractor.sift_dir.glob('*.npy'))
    swin_files = list(extractor.swin_dir.glob('*.npy'))

    print(f"Total SIFT descriptor files: {len(sift_files)}")
    print(f"Total Swin-B descriptor files: {len(swin_files)}")

    # Sample descriptor shapes
    if sift_files:
        sample_sift = np.load(sift_files[0])
        print(f"\nSample SIFT descriptor shape: {sample_sift.shape}")
        print(f"(n_keypoints={sample_sift.shape[0]}, descriptor_dim={sample_sift.shape[1]})")

    if swin_files:
        sample_swin = np.load(swin_files[0])
        print(f"\nSample Swin-B descriptor shape: {sample_swin.shape}")
        print(f"Feature dimension: {sample_swin.shape[0]}")


if __name__ == "__main__":
    main()
