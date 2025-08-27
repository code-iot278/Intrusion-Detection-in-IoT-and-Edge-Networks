# ============================================================
# Full Pipeline: Ghost Feature Expansion + Efficient Head
# + H2A Optimizer + AECR (IQR) + γ-Transfer Reassignment
# ============================================================

import os
import time
import copy
import math
from collections import Counter

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, matthews_corrcoef
)

# -----------------------------
# Utils: Device + Seeds (optional)
# -----------------------------
DEVICE = torch.device("cpu")  # change to "cuda" if you know you have it
torch.set_num_threads(max(1, os.cpu_count() // 2))

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)

# ============================================================
# Data Loader
# ============================================================
def load_csv_to_dataset(csv_path, target_column='label', n_components=128):
    if not os.path.exists(csv_path):
        print(f"[WARN] File not found: {csv_path}")
        return None, None

    df = pd.read_csv(csv_path)

    if target_column not in df.columns:
        print(f"[WARN] Target column '{target_column}' not in {csv_path}")
        print("       Available columns:", df.columns.tolist())
        return None, None

    print(f"[INFO] Unique values in '{target_column}' from {os.path.basename(csv_path)}:")
    print(df[target_column].dropna().unique())

    # Map common text labels to {0,1}
    if df[target_column].dtype == 'O':
        label_map = {
            'benign': 0, 'normal': 0, 'safe': 0, 'no': 0, '0': 0,
            'malicious': 1, 'attack': 1, 'abnormal': 1, 'yes': 1, '1': 1
        }
        df[target_column] = df[target_column].apply(
            lambda x: label_map.get(str(x).lower().strip(), np.nan)
        )
    else:
        # Keep only binary 0/1 rows
        df = df[df[target_column].isin([0, 1])]

    df = df.dropna(subset=[target_column])
    if df.empty:
        print(f"[WARN] No valid rows after mapping in {csv_path}")
        return None, None

    y = df[target_column].values
    X = df.drop(columns=[target_column]).select_dtypes(include=[np.number])

    if X.empty:
        print(f"[WARN] No numeric features in {csv_path}")
        return None, None

    # Standardize + PCA (auto-shrink components if needed)
    X = StandardScaler().fit_transform(X)
    n_samples, n_features = X.shape
    actual_components = min(n_components, n_samples, n_features)
    if actual_components < n_components:
        print(f"[INFO] Reducing PCA components: {n_components} -> {actual_components}")
    X = PCA(n_components=actual_components).fit_transform(X)
    if actual_components < n_components:
        X = np.pad(X, ((0, 0), (0, n_components - actual_components)), mode='constant')

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)
    ds = TensorDataset(X_tensor, y_tensor)
    return ds, X_tensor.shape[1]

# ============================================================
# Ghost Feature Expansion Block
# ============================================================
class GhostFeatureExpansion(nn.Module):
    """
    Lightweight feature expansion:
      - primary 1x1 conv to get intrinsic channels
      - cheap depthwise conv to synthesize ghost features
    """
    def __init__(self, in_channels, out_channels, ratio=2):
        super().__init__()
        init_channels = max(1, int(out_channels / ratio))
        new_channels = out_channels - init_channels
        self.primary = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True),
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, kernel_size=3, stride=1, padding=1,
                      groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        p = self.primary(x)
        c = self.cheap(p)
        return torch.cat([p, c], dim=1)  # shape: (B, out_channels, H, W)

# ============================================================
# Efficient Classifier Head
# ============================================================
class EfficientClassifier(nn.Module):
    """
    Dense -> reshape to small 2D map -> ghost expansion -> GAP -> linear
    Embedding dimension after GAP is ghost_out_channels (default 64).
    """
    def __init__(self, input_dim, base_channels=128, ghost_out_channels=64):
        super().__init__()
        self.base_channels = base_channels
        self.ghost_out_channels = ghost_out_channels

        self.feature_map = nn.Sequential(
            nn.Linear(input_dim, base_channels * 4 * 4),
            nn.ReLU(inplace=True),
        )
        self.ghost = nn.Sequential(
            GhostFeatureExpansion(base_channels, ghost_out_channels),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(ghost_out_channels, 1)
        )

    def forward(self, x):
        x = self.feature_map(x)             # (B, base*4*4)
        x = x.view(-1, self.base_channels, 4, 4)
        x = self.ghost(x)                   # (B, ghost_out, 1, 1)
        x = self.head(x)                    # (B, 1)
        return x

    @torch.no_grad()
    def embed(self, x):
        """
        Return the 64-dim embedding (after GAP, before final linear).
        """
        x = self.feature_map(x)             # (B, base*4*4)
        x = x.view(-1, self.base_channels, 4, 4)
        x = self.ghost(x)                   # (B, ghost_out, 1, 1)
        return x.view(x.size(0), -1)        # (B, ghost_out)

# ============================================================
# H2A Optimizer: Hybrid Harris–Aquila Adaptive Optimizer
#   - Population-based over flattened parameters
#   - Fitness: 1 - accuracy on dataset
# ============================================================
class H2AOptimizer:
    def __init__(self, model, dataset, max_iter=10, population_size=5,
                 batch_size=32, clamp=(-1.0, 1.0)):
        self.model = model
        self.dataset = dataset
        self.max_iter = max_iter
        self.population_size = population_size
        self.batch_size = batch_size
        self.dim = sum(p.numel() for p in model.parameters())
        self.template_params = self._flatten_params(model).to(torch.float32)
        self.clamp = clamp

    def _flatten_params(self, model):
        return torch.cat([p.data.view(-1) for p in model.parameters()]).clone().detach()

    def _load_flat_params(self, model, flat):
        idx = 0
        for p in model.parameters():
            n = p.numel()
            p.data.copy_(flat[idx: idx + n].view_as(p))
            idx += n

    @torch.no_grad()
    def _fitness(self, flat_params):
        # Evaluate accuracy (higher is better -> lower fitness value)
        local = copy.deepcopy(self.model).to(DEVICE)
        self._load_flat_params(local, flat_params)
        local.eval()

        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        preds, labels = [], []
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            logits = local(xb).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.extend((probs > 0.5).astype(int))
            labels.extend(yb.numpy().astype(int).ravel())
        acc = accuracy_score(labels, preds)
        return 1.0 - acc

    def _levy_step(self, size, beta=1.5):
        # Simple heavy-tailed step (approx)
        u = torch.randn(size)
        v = torch.randn(size)
        sigma_u = (math.gamma(1+beta) * torch.sin(torch.tensor(math.pi*beta/2)) /
                   (math.gamma((1+beta)/2) * beta * 2**((beta-1)/2))) ** (1/beta)
        step = u * sigma_u / (torch.abs(v) ** (1/beta) + 1e-8)
        return step

    def optimize(self):
        # Initialize population near template
        population = [
            (self.template_params + 0.01 * torch.randn_like(self.template_params))
            for _ in range(self.population_size)
        ]
        population = [torch.clamp(ind, *self.clamp) for ind in population]
        fitness = [self._fitness(ind) for ind in population]

        leader_idx = int(np.argmin(fitness))
        leader = population[leader_idx].clone()
        leader_score = fitness[leader_idx]

        for t in range(1, self.max_iter + 1):
            E1 = 2 * (1 - t / self.max_iter)  # Harris energy decay
            for i in range(self.population_size):
                X = population[i]
                # Hybrid switch: Harris (exploit) vs Aquila (explore)
                if np.random.rand() < 0.5:
                    # -------- Harris Hawks (exploit/soft besiege) --------
                    E0 = 2 * np.random.rand() - 1
                    E = E1 * E0
                    if abs(E) >= 1:
                        # exploration around random other solution
                        rnd = population[np.random.randint(self.population_size)]
                        r1, r2 = torch.rand_like(X), torch.rand_like(X)
                        X_new = X + r1 * (X - r2 * rnd)
                    else:
                        # soft besiege towards leader with exponential decay
                        decay = torch.exp(torch.tensor(-t / self.max_iter))
                        X_new = leader - torch.abs(X - leader) * decay
                else:
                    # -------- Aquila (exploration with Levy flights) --------
                    alpha = 0.01
                    step_levy = self._levy_step(X.shape[0])
                    X_new = leader + alpha * step_levy * (leader - X) + 0.005 * torch.randn_like(X)

                X_new = torch.clamp(X_new, *self.clamp)
                f_new = self._fitness(X_new)
                if f_new < fitness[i]:
                    population[i] = X_new
                    fitness[i] = f_new
                if f_new < leader_score:
                    leader = X_new.clone()
                    leader_score = f_new

        # Return best found params as state_dict
        best_model = copy.deepcopy(self.model)
        self._load_flat_params(best_model, leader)
        return best_model.state_dict()

# ============================================================
# Metrics / Evaluation
# ============================================================
@torch.no_grad()
def extended_evaluation(model, dataset, batch_size=64):
    model.eval().to(DEVICE)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    y_true, y_pred, y_prob = [], [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        logits = model(xb)
        prob = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        pred = (prob > 0.5).astype(int)
        y_prob.extend(prob)
        y_pred.extend(pred)
        y_true.extend(yb.numpy().astype(int).ravel())

    # Basic metrics
    acc  = round(accuracy_score(y_true, y_pred), 6)
    prec = round(precision_score(y_true, y_pred, zero_division=0), 6)
    rec  = round(recall_score(y_true, y_pred, zero_division=0), 6)
    f1   = round(f1_score(y_true, y_pred, zero_division=0), 6)
    auc  = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float('nan')

    # Confusion metrics
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = round(tn / (tn + fp), 6) if (tn + fp) > 0 else 0.0
        fpr = round(fp / (fp + tn), 6) if (fp + tn) > 0 else 0.0
        fnr = round(fn / (fn + tp), 6) if (fn + tp) > 0 else 0.0
        npv = round(tn / (tn + fn), 6) if (tn + fn) > 0 else 0.0
    else:
        specificity = fpr = fnr = npv = 0.0

    mcc = round(matthews_corrcoef(y_true, y_pred), 6) if len(np.unique(y_pred)) > 1 else 0.0
    return {
        "acc": acc, "prec": prec, "rec": rec, "f1": f1,
        "auc": round(auc, 6) if not np.isnan(auc) else 'NaN',
        "mcc": mcc, "specificity": specificity,
        "fpr": fpr, "fnr": fnr, "npv": npv,
    }

# ============================================================
# AECR: IQR-based Fitting / Non-Fitting Detection
# ============================================================
def aecr_iqr_labels(scores):
    """Return list ['Fitting'/'NonFitting'] for each score based on IQR."""
    q1 = np.percentile(scores, 25)
    q3 = np.percentile(scores, 75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    labels = ["Fitting" if (lower <= s <= upper) else "NonFitting" for s in scores]
    return labels, dict(q1=q1, q3=q3, iqr=iqr, lower=lower, upper=upper)

# ============================================================
# γ-Vectors (knowledge) + Reassignment of Non-Fitting Participants
# ============================================================
@torch.no_grad()
def compute_gamma_vector(model, dataset, max_batches=5, batch_size=64):
    """
    γ-vector: mean embedding over a few batches.
    """
    model.eval().to(DEVICE)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    embs = []
    for b_idx, (xb, _) in enumerate(loader):
        if b_idx >= max_batches:
            break
        xb = xb.to(DEVICE)
        em = model.embed(xb).cpu()  # (B, 64)
        embs.append(em)
    if not embs:
        return torch.zeros(64)
    embs = torch.cat(embs, dim=0)
    return embs.mean(dim=0)

@torch.no_grad()
def best_segment_by_gamma(data_gamma, segment_gammas):
    """
    Choose the segment with highest cosine similarity (lowest "distance").
    """
    best_k, best_sim = None, -1.0
    for k, g in segment_gammas.items():
        if g is None:
            continue
        sim = F.cosine_similarity(data_gamma, g, dim=0).item()
        if sim > best_sim:
            best_sim = sim
            best_k = k
    return best_k, best_sim

# ============================================================
# Federated Averaging
# ============================================================
def federated_averaging(local_weights):
    avg = copy.deepcopy(local_weights[0])
    for k in avg:
        avg[k] = avg[k].float()
        for i in range(1, len(local_weights)):
            avg[k] += local_weights[i][k].float()
        avg[k] /= float(len(local_weights))
    return avg

# ============================================================
# Main
# ============================================================
def main():
    # -----------------------------
    # 1) Datasets (edit paths here)
    # -----------------------------
    csv_paths = [
        ('/content/drive/MyDrive/Colab Notebooks/iot final/features_output.csv', 'label'),
        ('/content/drive/MyDrive/Colab Notebooks/iot final/features_with_labels.csv', 'Attack_label'),
        ('/content/drive/MyDrive/Colab Notebooks/iot final/sdd1/image_features.csv', 'label')
    ]

    datasets = []
    input_dim = None
    for path, target in csv_paths:
        ds, in_dim = load_csv_to_dataset(path, target_column=target, n_components=128)
        if ds is not None:
            datasets.append(ds)
            if input_dim is None:
                input_dim = in_dim

    if not datasets or input_dim is None:
        print("[FATAL] No datasets loaded. Check paths/columns.")
        return

    num_segments = len(datasets)
    print(f"\n[INFO] Loaded {num_segments} participant datasets. input_dim={input_dim}")

    # -----------------------------
    # 2) Round 1: Local H2A optimize
    # -----------------------------
    print("\n=== Global Model Round 1 Training (H2A Optimizer) ===")
    local_weights = []
    per_participant_stats = []
    comm_bytes_round1 = 0
    start_time = time.time()

    # Train each participant locally with H2A
    for i, ds in enumerate(datasets):
        print(f"\n[Participant P{i+1}] Class Distribution: {Counter(ds.tensors[1].numpy().flatten())}")
        model_i = EfficientClassifier(input_dim=input_dim).to(DEVICE)
        opt = H2AOptimizer(
            model=model_i, dataset=ds,
            max_iter=6, population_size=4, batch_size=64
        )
        best_state = opt.optimize()
        local_weights.append(best_state)

        # Communication cost ~ size of weights
        tmp_model = EfficientClassifier(input_dim=input_dim)
        tmp_model.load_state_dict(best_state, strict=True)
        weight_size = sum(p.nelement() * p.element_size() for p in tmp_model.parameters())
        comm_bytes_round1 += weight_size

    time_round1 = time.time() - start_time

    # -----------------------------
    # 3) Aggregate (FedAvg)
    # -----------------------------
    global_model = EfficientClassifier(input_dim=input_dim).to(DEVICE)
    global_weights = federated_averaging(local_weights)
    global_model.load_state_dict(global_weights, strict=True)

    # -----------------------------
    # 4) Evaluate per participant
    # -----------------------------
    print("\n=== Evaluation After Round 1 ===")
    accs = []
    for i, ds in enumerate(datasets):
        model_eval = EfficientClassifier(input_dim=input_dim).to(DEVICE)
        model_eval.load_state_dict(global_weights, strict=True)
        m = extended_evaluation(model_eval, ds)
        accs.append(m["acc"])
        per_participant_stats.append({"participant": f"P{i+1}", **m})
        print(f"P{i+1}: "
              f"Accuracy={m['acc']}, Precision={m['prec']}, F1={m['f1']}, "
              f"Specificity={m['specificity']}, Sensitivity={m['rec']}, "
              f"MCC={m['mcc']}, NPV={m['npv']}, FPR={m['fpr']}, FNR={m['fnr']}")

    print("\n=== Efficiency Metrics (Round 1) ===")
    print(f"Convergence Time: {time_round1:.2f} s")
    print(f"Communication Overhead: {comm_bytes_round1} bytes")

    # -----------------------------
    # 5) AECR: IQR-based fitting/non-fitting
    # -----------------------------
    labels, iqr_info = aecr_iqr_labels(accs)
    print("\n=== AECR (IQR) Fitting / NonFitting ===")
    print(f"IQR Stats: Q1={iqr_info['q1']:.6f}, Q3={iqr_info['q3']:.6f}, "
          f"IQR={iqr_info['iqr']:.6f}, Range=[{iqr_info['lower']:.6f}, {iqr_info['upper']:.6f}]")
    for i, (acc, lab) in enumerate(zip(accs, labels)):
        print(f"P{i+1}: acc={acc:.6f} -> {lab}")

    # -----------------------------
    # 6) γ-Transfer reassignment for NonFitting
    # (one-shot reassignment; guard oscillation by allowing at most 1 reassign now)
    # -----------------------------
    print("\n=== γ-Transfer Reassignment (one-shot) ===")
    # Compute γ-vectors for current segments using the global model
    segment_gammas = {}
    with torch.no_grad():
        for i, ds in enumerate(datasets):
            gamma = compute_gamma_vector(global_model, ds)  # (64,)
            segment_gammas[i] = gamma

    # Identify non-fitting and attempt reassignment to best segment by cosine similarity
    reassigned = {}
    for i, lab in enumerate(labels):
        if lab == "NonFitting":
            print(f"[Reassign] P{i+1} marked NonFitting; searching best segment by γ-similarity...")
            # Compute γ of this participant's data
            data_gamma = compute_gamma_vector(global_model, datasets[i])
            # Find best segment (other than itself)
            cand_gammas = {k: v for k, v in segment_gammas.items() if k != i}
            best_k, best_sim = best_segment_by_gamma(data_gamma, cand_gammas)
            if best_k is None:
                print(f"  -> No suitable segment found. Skipping.")
                continue
            print(f"  -> Best segment: P{best_k+1} (cosine sim = {best_sim:.4f})")
            # Reassign: here we conceptually re-label this participant to use the target segment's model
            # For demo, we simply evaluate the global model again (no weights change) —
            # in a real system you'd re-train locally with the target segment's global weights.
            reassigned[i] = best_k

    # -----------------------------
    # 7) Post-reassignment evaluation (optional demonstration)
    # -----------------------------
    if reassigned:
        print("\n=== Post-Reassignment (Demonstration) ===")
        for i, tgt in reassigned.items():
            # Evaluate the same global model; in real flow, you might pull GM_target and continue training.
            m = extended_evaluation(global_model, datasets[i])
            print(f"P{i+1} -> reassigned to P{tgt+1}'s segment | "
                  f"Acc={m['acc']}, Prec={m['prec']}, F1={m['f1']}")

    # -----------------------------
    # 8) Save results (CSV + NPY) - optional
    # -----------------------------
    results_dir = "/content/drive/MyDrive/Colab Notebooks/iot final/results"
    try:
        os.makedirs(results_dir, exist_ok=True)
        df = pd.DataFrame(per_participant_stats)
        csv_path = os.path.join(results_dir, "round1_metrics.csv")
        npy_path = os.path.join(results_dir, "round1_metrics.npy")
        df.to_csv(csv_path, index=False)
        np.save(npy_path, df.to_records(index=False))
        print(f"\n[INFO] Saved metrics to:\n  - {csv_path}\n  - {npy_path}")
    except Exception as e:
        print(f"[WARN] Could not save results: {e}")

if __name__ == "__main__":
    main()

