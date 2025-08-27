import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def augment_image(image):
    # 1. Rotation
    angle = 15
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h))

    # 2. Horizontal Flip
    flipped_h = cv2.flip(image, 1)

    # 3. Vertical Flip
    flipped_v = cv2.flip(image, 0)

    # 4. Contrast Normalization
    contrast_img = cv2.convertScaleAbs(image, alpha=1.5, beta=0)

    return [rotated, flipped_h, flipped_v, contrast_img], ['rotated', 'flipped_h', 'flipped_v', 'contrast']

def display_images(original, augmented_list, titles):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, len(augmented_list) + 1, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis("off")

    for i, (aug, title) in enumerate(zip(augmented_list, titles), start=2):
        plt.subplot(1, len(augmented_list) + 1, i)
        plt.imshow(cv2.cvtColor(aug, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

def augment_images_display_and_save(input_folder, output_folder):
    for root, _, files in os.walk(input_folder):
        for file in sorted(files):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                image = cv2.imread(img_path)
                if image is None:
                    continue

                # Display
                aug_images, aug_names = augment_image(image)
                display_images(image, aug_images, [name.capitalize() for name in aug_names])

                # Create corresponding output path
                relative_path = os.path.relpath(root, input_folder)
                output_subfolder = os.path.join(output_folder, relative_path)
                os.makedirs(output_subfolder, exist_ok=True)

                base_filename = os.path.splitext(file)[0]
                cv2.imwrite(os.path.join(output_subfolder, f'original_{file}'), image)
                for aug_img, aug_name in zip(aug_images, aug_names):
                    aug_filename = f'{aug_name}_{base_filename}.png'
                    cv2.imwrite(os.path.join(output_subfolder, aug_filename), aug_img)

# ======= Set input and output folders ========
input_main_folder = '/content/drive/MyDrive/Colab Notebooks/iot final/sdd1/Resized_Tensor_48x48'
output_main_folder = '/content/drive/MyDrive/Colab Notebooks/iot final/sdd1/Augmented_Output'

# Run the processing
augment_images_display_and_save(input_main_folder, output_main_folder)