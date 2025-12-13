import mmbra
import mmbracategories
import os
import scipy.io as sio
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.metrics.pairwise import cosine_similarity
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# =============================
# 1. Set paths and subject info
# =============================
data_dir_root = './data/ThingsEEG-Text'
sbj = 'sub-10'
image_model = 'pytorch/cornet_s'
text_model = 'CLIPText'
roi = '17channels'

brain_dir = os.path.join(data_dir_root, 'brain_feature', roi, sbj)
image_dir_seen = os.path.join(data_dir_root, 'visual_feature/ThingsTrain', image_model, sbj)
image_dir_unseen = os.path.join(data_dir_root, 'visual_feature/ThingsTest', image_model, sbj)
text_dir_seen = os.path.join(data_dir_root, 'textual_feature/ThingsTrain/text', text_model, sbj)
text_dir_unseen = os.path.join(data_dir_root, 'textual_feature/ThingsTest/text', text_model, sbj)

# =============================
# 2. Load brain, image, text data with error checking
# =============================
def load_brain_data(file_path, time_slice=(27, 60)):
    """Load brain data with validation."""
    try:
        mat = sio.loadmat(file_path)
        if 'data' not in mat:
            raise KeyError(f"'data' key not found in {file_path}")
        if 'class_idx' not in mat:
            raise KeyError(f"'class_idx' key not found in {file_path}")
        
        data = mat['data'].astype('double') * 2.0
        data = data[:, :, time_slice[0]:time_slice[1]]
        data = data.reshape(data.shape[0], -1)
        labels = mat['class_idx'].T.astype('int')
        return data, labels
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        raise

def load_feature_data(file_path, scale=1.0):
    """Load feature data with validation."""
    try:
        mat = sio.loadmat(file_path)
        if 'data' not in mat:
            raise KeyError(f"'data' key not found in {file_path}")
        data = mat['data'].astype('double') * scale
        return data
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        raise

print("=== Loading Data ===")
brain_seen, label_seen = load_brain_data(os.path.join(brain_dir, 'eeg_train_data_within.mat'))
brain_unseen, label_unseen = load_brain_data(os.path.join(brain_dir, 'eeg_test_data.mat'))

image_seen = load_feature_data(os.path.join(image_dir_seen, 'feat_pca_train.mat'), scale=50.0)
image_unseen = load_feature_data(os.path.join(image_dir_unseen, 'feat_pca_test.mat'), scale=50.0)

text_seen = load_feature_data(os.path.join(text_dir_seen, 'text_feat_train.mat'), scale=2.0)
text_unseen = load_feature_data(os.path.join(text_dir_unseen, 'text_feat_test.mat'), scale=2.0)

print("Data loaded successfully!")

# =============================
# 3. Multi-Modal Pairing & Standardization
# =============================
X_seen = np.concatenate([image_seen, text_seen], axis=1)
X_unseen = np.concatenate([image_unseen, text_unseen], axis=1)
y_seen = label_seen.ravel()
y_unseen = label_unseen.ravel()

# Check for NaNs/Infs
def check_array(arr, name="Array"):
    """Check for NaNs and Infs."""
    nan_count = np.isnan(arr).sum()
    inf_count = np.isinf(arr).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"WARNING {name}: {nan_count} NaNs, {inf_count} Infs")
        arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=-1.0)
    return arr

X_seen = check_array(X_seen, "X_seen")
X_unseen = check_array(X_unseen, "X_unseen")

# Standardize features
scaler = StandardScaler()
X_seen = scaler.fit_transform(X_seen)
X_unseen = scaler.transform(X_unseen)

print("\n=== Data Summary ===")
print("Seen (train) Data - Samples:", X_seen.shape[0], "Features:", X_seen.shape[1])
print("Unseen (test) Data - Samples:", X_unseen.shape[0], "Features:", X_unseen.shape[1])

# =============================
# 4. Class Counts & Zero-Shot Split
# =============================
def print_class_counts(y, label="Dataset"):
    """Print class distribution."""
    unique, counts = np.unique(y, return_counts=True)
    df = pd.DataFrame({"Class": unique, "Count": counts})
    print(f"\n=== {label} Class Counts ===")
    print(df.to_string(index=False))
    print("Total samples:", y.shape[0])

print_class_counts(y_seen, label="Seen (Train) Data")
print_class_counts(y_unseen, label="Unseen (Test) Data")

CSeen = np.unique(y_seen)
CUnseen = np.unique(y_unseen)
df_split = pd.DataFrame({
    "CSeen": pd.Series(CSeen),
    "CUnseen": pd.Series(CUnseen)
})
print("\n=== Zero-Shot Split (Disjoint Classes) ===")
print(df_split.to_string(index=False))

# Check disjointness
inter = set(CSeen).intersection(set(CUnseen))
if inter:
    print(f"WARNING: Class sets overlap: {inter}")
else:
    print("✓ Confirmed: Seen and unseen class sets are disjoint.")

# =============================
# 5. Optional: mmbra visualization (wrapped in try/except)
# =============================
try:
    print("\n=== Running mmbra Analysis ===")
    mmbra.data_analysis_example(torch.from_numpy(brain_seen), 
                                 torch.from_numpy(image_seen), 
                                 torch.from_numpy(text_seen))
    mmbra.data_visualization_example(torch.from_numpy(label_seen))
    mmbra.data_visualization_example(torch.from_numpy(label_unseen))
except Exception as e:
    print(f"mmbra functions skipped: {e}")

# =============================
# 6. Baseline SVM Classifier
# =============================
print("\n=== Training Baseline SVM ===")
clf = SVC(kernel='linear', class_weight='balanced', probability=False)
clf.fit(X_seen, y_seen)
y_pred_baseline = clf.predict(X_unseen)

acc_baseline = accuracy_score(y_unseen, y_pred_baseline)
f1_baseline = f1_score(y_unseen, y_pred_baseline, average='macro', zero_division=0)

print("Baseline SVM Accuracy:", f"{acc_baseline:.4f}")
print("Baseline SVM Macro F1:", f"{f1_baseline:.4f}")

# =============================
# Step 1: Compute Semantic Prototypes
# =============================
def compute_prototypes(text_feats, labels):
    """Compute semantic prototypes (mean embeddings) for each class."""
    classes = np.unique(labels)
    prototypes = {}
    for c in classes:
        mask = labels == c
        prototypes[c] = text_feats[mask].mean(axis=0)
    return prototypes

prototypes_seen = compute_prototypes(text_seen, y_seen)
prototypes_unseen = compute_prototypes(text_unseen, y_unseen)

print("\n=== Semantic Prototypes ===")
print("Seen class prototypes:", len(prototypes_seen))
print("Unseen class prototypes:", len(prototypes_unseen))
proto_dim = list(prototypes_seen.values())[0].shape[0]
print("Prototype dimension:", proto_dim)

first_class = list(prototypes_seen.keys())[0]
print(f"Example prototype (class {first_class}, first 5 dims):", prototypes_seen[first_class][:5])

# =============================
# Step 2: Train Local Mapping Network
# =============================
def create_semantic_targets(labels, prototypes):
    """Create semantic target vectors for each sample."""
    sem_dim = list(prototypes.values())[0].shape[0]
    S_targets = np.zeros((len(labels), sem_dim))
    for i, label in enumerate(labels):
        S_targets[i] = prototypes[label]
    return S_targets

S_seen = create_semantic_targets(y_seen, prototypes_seen)
S_seen = check_array(S_seen, "S_seen")

print("\n=== Semantic Targets ===")
print("S_seen shape:", S_seen.shape)
print("First sample target (first 5 dims):", S_seen[0, :5])

# Ridge regression: map X -> semantic space
print("\n=== Training Mapping Network (Ridge Regression) ===")
ridge = Ridge(alpha=1.0)
ridge.fit(X_seen, S_seen)

W_mapping = ridge.coef_.T
b_mapping = ridge.intercept_

print("Mapping matrix W shape:", W_mapping.shape)
print("Input dim:", X_seen.shape[1], "→ Semantic dim:", S_seen.shape[1])

S_pred_seen = X_seen @ W_mapping + b_mapping
mse_train = np.mean((S_pred_seen - S_seen) ** 2)
print(f"Training MSE: {mse_train:.6f}")

# =============================
# Step 3: Federated Training (FedAvg)
# =============================
def client_train_ridge(Xk, yk, prototypes_dict, alpha=1.0):
    """Train local ridge regression model."""
    S = np.vstack([prototypes_dict[int(lbl)] for lbl in yk])
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(Xk, S)
    return model.coef_.T, len(yk)

def fedavg_train(client_splits_X, client_splits_y, prototypes_dict, rounds=3, alpha=1.0):
    """Federated Averaging training."""
    n_total = sum([len(y) for y in client_splits_y])
    num_clients = len(client_splits_X)
    
    input_dim = client_splits_X[0].shape[1]
    sem_dim = list(prototypes_dict.values())[0].shape[0]
    W_global = np.zeros((input_dim, sem_dim))
    
    print("\n=== Federated Training (FedAvg) ===")
    print(f"Clients: {num_clients}, Rounds: {rounds}, Total samples: {n_total}")
    
    for r in range(rounds):
        Ws, ns = [], []
        for k, (Xk, yk) in enumerate(zip(client_splits_X, client_splits_y)):
            Wk, nk = client_train_ridge(Xk, yk, prototypes_dict, alpha=alpha)
            Ws.append(Wk)
            ns.append(nk)
        
        W_global = np.zeros_like(Ws[0])
        for Wk, nk in zip(Ws, ns):
            W_global += Wk * (nk / n_total)
        
        print(f"Round {r+1}/{rounds}: W_global aggregated")
    
    return W_global

# Create client splits (simple stratified split)
num_clients = 3
indices = np.arange(len(y_seen))
np.random.shuffle(indices)
client_splits = np.array_split(indices, num_clients)
client_X = [X_seen[split] for split in client_splits]
client_y = [y_seen[split] for split in client_splits]

print(f"\nClient data distribution: {[len(y) for y in client_y]}")

# Train FedAvg
W_global = fedavg_train(client_X, client_y, prototypes_seen, rounds=3, alpha=1.0)

print("\n=== Global Mapping Network ===")
print("W_global shape:", W_global.shape)

S_pred_global = X_seen @ W_global
mse_global = np.mean((S_pred_global - S_seen) ** 2)
print(f"Global model MSE on seen data: {mse_global:.6f}")

# =============================
# Step 4: Zero-Shot Prediction
# =============================
print("\n=== Zero-Shot Prediction ===")

# Check shapes before multiplication
if X_unseen.shape[1] != W_global.shape[0]:
    raise ValueError(f"Shape mismatch: X_unseen {X_unseen.shape[1]} vs W_global {W_global.shape[0]}")

Z_unseen = X_unseen @ W_global
Z_unseen = check_array(Z_unseen, "Z_unseen")

print("Mapped embeddings Z_unseen shape:", Z_unseen.shape)

labels_unseen_sorted = sorted(prototypes_unseen.keys())
P_unseen = np.vstack([prototypes_unseen[c] for c in labels_unseen_sorted])

print("Unseen prototypes shape:", P_unseen.shape)
print("Number of unseen classes:", len(labels_unseen_sorted))

sim_matrix = cosine_similarity(Z_unseen, P_unseen)
indices = sim_matrix.argmax(axis=1)
y_pred_zsl = np.array([labels_unseen_sorted[i] for i in indices])

print("ZSL predictions (first 10):", y_pred_zsl[:10])
print("True labels (first 10):    ", y_unseen[:10])

# =============================
# 5. Evaluation: ZSL vs Baseline
# =============================
acc_zsl = accuracy_score(y_unseen, y_pred_zsl)
f1_zsl = f1_score(y_unseen, y_pred_zsl, average='macro', zero_division=0)

print("\n" + "="*60)
print("=== ZERO-SHOT LEARNING EVALUATION ===")
print("="*60)
print(f"Accuracy:  {acc_zsl:.4f}")
print(f"Macro F1:  {f1_zsl:.4f}")
print("="*60)

cm_zsl = confusion_matrix(y_unseen, y_pred_zsl)
print(f"\nConfusion Matrix shape: {cm_zsl.shape}")

print("\nClassification Report:\n")
print(classification_report(y_unseen, y_pred_zsl, zero_division=0))

print("\n" + "="*60)
print("=== COMPARISON: BASELINE SVM vs ZERO-SHOT (FedAvg) ===")
print("="*60)
print(f"Baseline SVM Accuracy:       {acc_baseline:.4f}")
print(f"Zero-Shot (FedAvg) Accuracy: {acc_zsl:.4f}")
print(f"Improvement:                 {acc_zsl - acc_baseline:+.4f}")
print("="*60)

# =============================
# 6. Visualization: PCA & t-SNE
# =============================
def plot_embeddings(X, y, title="Embedding Visualization", method="PCA", perplexity=20, max_samples=1000):
    """Plot 2D reduction of embeddings."""
    X = np.asarray(X)
    y = np.asarray(y)
    n = X.shape[0]
    
    if n > max_samples:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_samples, replace=False)
        X_sub, y_sub = X[idx], y[idx]
    else:
        X_sub, y_sub = X, y
    
    if method == "PCA":
        reducer = PCA(n_components=2, random_state=0)
    elif method == "t-SNE":
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(perplexity, len(np.unique(y_sub))-1))
    else:
        raise ValueError("Method must be 'PCA' or 't-SNE'")
    
    try:
        X_reduced = reducer.fit_transform(X_sub)
        plt.figure(figsize=(8, 6))
        for cls in np.unique(y_sub):
            mask = (y_sub == cls)
            plt.scatter(X_reduced[mask, 0], X_reduced[mask, 1], label=f"Class {cls}", alpha=0.6, s=10)
        plt.title(f"{title} ({method})")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Visualization failed ({method}): {e}")

print("\n=== Embedding Visualizations ===")
try:
    plot_embeddings(X_seen, y_seen, title="Seen Data", method="PCA")
    plot_embeddings(X_seen, y_seen, title="Seen Data", method="t-SNE", perplexity=15)
except Exception as e:
    print(f"Seen data visualization failed: {e}")

try:
    plot_embeddings(X_unseen, y_unseen, title="Unseen Data", method="PCA")
    plot_embeddings(X_unseen, y_unseen, title="Unseen Data", method="t-SNE", perplexity=15)
except Exception as e:
    print(f"Unseen data visualization failed: {e}")

print("\n=== Analysis Complete ===")