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
        x = x.unsqueeze(1)  # Add sequence dimension for attention
        x = self.attn(x)
        x = x.squeeze(1)    # Remove sequence dimension
        x = self.blocks(x)
        x = self.output_proj(x)
        return x

# ---------------- Feature Extraction Function ----------------
def extract_features_with_labels(csv_path, output_path):
    # Load CSV file
    df = pd.read_csv(csv_path)

    # Select numeric input features (excluding label)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if 'label' in numeric_cols:
        numeric_cols.remove('label')

    input_tensor = torch.tensor(df[numeric_cols].values, dtype=torch.float32)

    # Initialize encoder model
    input_dim = input_tensor.shape[1]
    encoder = ModifiedMLPEncoder(input_dim=input_dim)
    encoder.eval()

    # Feature extraction
    with torch.no_grad():
        features = encoder(input_tensor)

    # Create output DataFrame with features
    features_df = pd.DataFrame(features.numpy())

    # Add label column back
    features_df['label'] = df['label'].values

    # Save to CSV
    features_df.to_csv(output_path, index=False)
    print(f"✅ Feature extraction with labels saved to: {output_path}")

# ---------------- Main Execution ----------------
if __name__ == "__main__":
    input_csv = '/content/drive/MyDrive/Colab Notebooks/iot final/normalized_output.csv'
    output_csv = '/content/drive/MyDrive/Colab Notebooks/iot final/features_output.csv'
    extract_features_with_labels(input_csv, output_csv)

    # Optional: Load the output CSV
    df = pd.read_csv(output_csv)
    print(df.head())
