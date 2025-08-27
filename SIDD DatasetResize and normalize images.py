import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

def resize_and_normalize_tensor(image, size=(48, 48)):
    """
    Resize the image and normalize pixel values to [0, 1], then convert to torch tensor [C, H, W].

    Args:
        image (numpy array): Original BGR image from OpenCV.
        size (tuple): Target size (width, height) for resizing.

    Returns:
        torch.Tensor: Normalized float tensor of shape [3, H, W].
    """
    resized = cv2.resize(image, size)
    rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb_image.astype(np.float32) / 255.0
    tensor = torch.tensor(normalized).permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
    return tensor

def process_images_to_tensor(main_folder, output_folder):
    """
    Process images by resizing to 48×48, converting to normalized tensors, saving as PNGs.

    Args:
        main_folder (str): Path to input image folder.
        output_folder (str): Path where processed images will be saved.
    """
    for root, dirs, files in os.walk(main_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                image_path = os.path.join(root, file)
                original = cv2.imread(image_path)

                if original is None:
                    continue

                tensor = resize_and_normalize_tensor(original)

                # Convert tensor back to image for saving (denormalize)
                processed_np = (tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                processed_bgr = cv2.cvtColor(processed_np, cv2.COLOR_RGB2BGR)

                # Recreate output folder structure
                rel_path = os.path.relpath(root, main_folder)
                save_dir = os.path.join(output_folder, rel_path)
                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(save_dir, file)
                cv2.imwrite(save_path, processed_bgr)

                # Display original and processed image
                plt.figure(figsize=(8, 4))

                plt.subplot(1, 2, 1)
                plt.title("Original")
                plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.title("48×48 Normalized Tensor")
                plt.imshow(processed_np)
                plt.axis('off')

                plt.tight_layout()
                plt.show()

# ===========================
# ✅ Run the Processing
# ===========================
input_folder = "/content/drive/MyDrive/Colab Notebooks/iot final/sdd1/n048"
output_folder = "/content/drive/MyDrive/Colab Notebooks/iot final/sdd1/Resized_Tensor_48x48"

process_images_to_tensor(input_folder, output_folder)