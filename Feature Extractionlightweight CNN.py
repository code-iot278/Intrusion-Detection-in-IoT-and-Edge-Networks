import os
import cv2
import torch
import numpy as np
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# --------- Spatial Attention Module ---------
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        return x * attention

# --------- Image Encoder with MobileNetV2 + ResNet18 ---------
class ImageEncoder(nn.Module):
    def __init__(self, embed_dim=256):
        super(ImageEncoder, self).__init__()

        mobilenet = models.mobilenet_v2(pretrained=True).features
        resnet = models.resnet18(pretrained=True)

        self.mobilenet = mobilenet
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])  # remove avgpool & fc

        self.reduce_mobilenet = nn.Conv2d(1280, 512, kernel_size=1)
        self.reduce_resnet = nn.Conv2d(512, 512, kernel_size=1)

        self.attention = SpatialAttention()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, embed_dim)  # 512 + 512 = 1024

    def forward(self, x):
        feat_mobile = self.mobilenet(x)
        feat_mobile = self.reduce_mobilenet(feat_mobile)

        feat_resnet = self.resnet(x)
        feat_resnet = self.reduce_resnet(feat_resnet)

        feat_fused = torch.cat([feat_mobile, feat_resnet], dim=1)
        feat_attended = self.attention(feat_fused)

        pooled = self.gap(feat_attended).view(x.size(0), -1)
        embedding = self.fc(pooled)
        return embedding

# --------- Custom Dataset Loader ---------
class ImageFolderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.filenames = []
        self.transform = transform

        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    path = os.path.join(root, file)
                    self.image_paths.append(path)
                    self.labels.append(os.path.basename(root))  # folder name as label
                    self.filenames.append(file)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)

        return image, self.labels[idx], self.filenames[idx], img_path

# --------- Feature Extraction and Save to CSV ---------
def extract_features_to_csv(input_folder, output_csv, embed_dim=256, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = ImageFolderDataset(input_folder, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = ImageEncoder(embed_dim=embed_dim).to(device)
    model.eval()

    all_features = []

    with torch.no_grad():
        for images, labels, filenames, paths in dataloader:
            images = images.to(device)
            embeddings = model(images).cpu().numpy()

            for i in range(embeddings.shape[0]):
                row = {
                    "filename": filenames[i],
                    "label": labels[i],
                    "path": paths[i],
                }
                # Add features
                for j in range(embeddings.shape[1]):
                    row[f"f{j+1}"] = embeddings[i][j]
                all_features.append(row)

    df = pd.DataFrame(all_features)
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Feature extraction completed. CSV saved at: {output_csv}")

# --------- Set Paths and Run ---------
if __name__ == "__main__":
    input_main_folder = "/content/drive/MyDrive/Colab Notebooks/iot final/sdd1/Augmented_Output"
    output_csv_file = "/content/drive/MyDrive/Colab Notebooks/iot final/sdd1/image_features.csv"
    extract_features_to_csv(input_main_folder, output_csv_file)
df=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/iot final/sdd1/image_features.csv')
df