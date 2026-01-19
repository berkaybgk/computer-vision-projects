import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Test NumPy
a = np.array([[1, 2, 3], [4, 5, 6]])
print("NumPy array:\n", a)
print("Array mean:", np.mean(a))

# Test Pillow
img = Image.new("RGB", (100, 100), color="red")
img.save("test_image_pillow.jpg")
print("Saved an image with Pillow!")

# Test OpenCV
img_cv = cv2.imread("test_image_pillow.jpg")
if img_cv is not None:
    print("OpenCV successfully read the image! Shape:", img_cv.shape)
else:
    print("OpenCV failed to read the image.")

# Test Matplotlib
plt.imshow(img_cv[:, :, ::-1])  # Convert BGR to RGB
plt.title("Image loaded with OpenCV")
plt.show()

