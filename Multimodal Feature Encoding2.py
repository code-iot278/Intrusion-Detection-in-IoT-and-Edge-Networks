import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- Self-Attention Layer ----------------
class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** 0.5

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output

# ---------------- Residual MLP Block ----------------
class ResidualMLPBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualMLPBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        out = self.norm(out + residual)
        return out

# ---------------- Full Tabular Encoder ----------------
class ModifiedMLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_blocks=2):
        super(ModifiedMLPEncoder, self).__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attn = SelfAttention(hidden_dim)
        self.blocks = nn.Sequential(*[ResidualMLPBlock(hidden_dim) for _ in range(num_blocks)])
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        x = self.input_proj(x)
        x = x.unsqueeze(1)
        x = self.attn(x)
        x = x.squeeze(1)
        x = self.blocks(x)
        x = self.output_proj(x)
        return x

# ---------------- Feature Extraction and Label Append ----------------
def extract_features_with_labels(csv_path, output_path):
    # Load data
    df = pd.read_csv(csv_path)

    # Keep only numeric columns for input (exclude Attack_label)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if 'Attack_label' in numeric_cols:
        numeric_cols.remove('Attack_label')

    input_tensor = torch.tensor(df[numeric_cols].values, dtype=torch.float32)

    # Initialize model
    input_dim = input_tensor.shape[1]
    encoder = ModifiedMLPEncoder(input_dim=input_dim)
    encoder.eval()

    # Extract features
    with torch.no_grad():
        features = encoder(input_tensor)

    # Convert features to DataFrame
    features_df = pd.DataFrame(features.numpy())

    # Add Attack_label column
    features_df['Attack_label'] = df['Attack_label']

    # Save final CSV
    features_df.to_csv(output_path, index=False)
    print(f"✅ Feature extraction with label completed and saved to: {output_path}")

# ---------------- Run It ----------------
if __name__ == "__main__":
    input_csv = '/content/drive/MyDrive/Colab Notebooks/iot final/normalized_output.csv'
    output_csv = '/content/drive/MyDrive/Colab Notebooks/iot final/features_with_labels.csv'
    extract_features_with_labels(input_csv, output_csv)

import pandas as pd
df=pd.read_csv('/content/drive/MyDrive/Colab Notebooks/iot final/features_with_labels.csv')
df