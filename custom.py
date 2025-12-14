import mmbra
import mmbracategories
import os
import scipy.io as sio
import numpy as np
from sklearn.preprocessing import StandardScaler
# added ML imports
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
import torch
import torch.nn as nn
import torch.nn.functional as F

# Hyperparameters
tau = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def zsl_contrastive_loss(emb_v, prototypes, labels, tau=0.1):
    """
    Compute contrastive ZSL loss for a batch.

    emb_v: (batch_size, emb_dim) visual embeddings
    prototypes: dict {class_id: prototype_vector}
    labels: (batch_size,) true class labels
    tau: temperature
    """
    batch_size = emb_v.shape[0]
    emb_v = F.normalize(emb_v, dim=1)

    class_ids = sorted(prototypes.keys())
    proto_matrix = torch.stack([
        torch.tensor(prototypes[c], device=device, dtype=torch.float32)
        for c in class_ids
    ])
    proto_matrix = F.normalize(proto_matrix, dim=1)

    sim = torch.matmul(emb_v, proto_matrix.T) / tau

    label_to_idx = {c: i for i, c in enumerate(class_ids)}
    target_idx = torch.tensor([label_to_idx[int(l)] for l in labels], device=device)

    loss = F.cross_entropy(sim, target_idx)
    return loss

# -----------------------------
# PyTorch ZSL client training loop (FedAvg client)
# -----------------------------
def client_train_zsl(Xk, yk, prototypes_dict, lr=0.01, epochs=50, batch_size=32):
    """
    Train local ZSL mapping using contrastive loss (FedAvg client).
    Xk: np.ndarray (n_samples, feature_dim)
    yk: np.ndarray (n_samples,) class labels
    prototypes_dict: dict {class_id: prototype_vector}
    Returns: (W_np, n_samples) where W_np has shape (input_dim, sem_dim)
    """
    Xk = torch.tensor(Xk, dtype=torch.float32, device=device)
    yk = torch.tensor(yk, dtype=torch.long, device=device)

    input_dim = Xk.shape[1]
    sem_dim = list(prototypes_dict.values())[0].shape[0]
    W = nn.Linear(input_dim, sem_dim, bias=False).to(device)

    optimizer = torch.optim.Adam(W.parameters(), lr=lr)

    dataset = torch.utils.data.TensorDataset(Xk, yk)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            emb_v = W(xb)
            loss = zsl_contrastive_loss(emb_v, prototypes_dict, yb, tau=tau)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.shape[0]
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(Xk):.4f}")

    return W.weight.data.cpu().numpy().T, int(yk.shape[0])

# -----------------------------
# FedAvg orchestration for ZSL clients
# -----------------------------
def fedavg_train_zsl(client_splits_X, client_splits_y, prototypes_dict, rounds=3, lr=0.01, epochs=50):
    """
    Federated Averaging for ZSL mapping.
    Aggregates client-trained linear mappings using sample-count weights.
    Returns W_global with shape (input_dim, sem_dim).
    """
    n_total = sum([len(y) for y in client_splits_y])
    num_clients = len(client_splits_X)

    input_dim = client_splits_X[0].shape[1]
    sem_dim = list(prototypes_dict.values())[0].shape[0]
    W_global = np.zeros((input_dim, sem_dim))

    for r in range(rounds):
        Ws = []
        ns = []
        print(f"\n--- FedAvg Round {r+1}/{rounds} ---")
        for k, (Xk, yk) in enumerate(zip(client_splits_X, client_splits_y)):
            Wk, nk = client_train_zsl(Xk, yk, prototypes_dict, lr=lr, epochs=epochs)
            Ws.append(Wk)
            ns.append(nk)
        W_global = sum(Wk * (nk / n_total) for Wk, nk in zip(Ws, ns))

    return W_global

# -----------------------------
# 1. Set paths and subject info
# -----------------------------
data_dir_root = './data/ThingsEEG-Text'
sbj = 'sub-10'
image_model = 'pytorch/cornet_s'
text_model = 'CLIPText'
roi = '17channels'

# Brain data
brain_dir = os.path.join(data_dir_root, 'brain_feature', roi, sbj)

# Image data
image_dir_seen = os.path.join(data_dir_root, 'visual_feature/ThingsTrain', image_model, sbj)
image_dir_unseen = os.path.join(data_dir_root, 'visual_feature/ThingsTest', image_model, sbj)

# Text data
text_dir_seen = os.path.join(data_dir_root, 'textual_feature/ThingsTrain/text', text_model, sbj)
text_dir_unseen = os.path.join(data_dir_root, 'textual_feature/ThingsTest/text', text_model, sbj)

# -----------------------------
# 2. Load brain, image, text data
# -----------------------------
def load_brain_data(file_path, time_slice=(27,60)):
    mat = sio.loadmat(file_path)
    data = mat['data'].astype('double') * 2.0
    data = data[:, :, time_slice[0]:time_slice[1]]  # 70ms-400ms
    data = data.reshape(data.shape[0], -1)
    labels = mat['class_idx'].T.astype('int')
    return data, labels

brain_seen, label_seen = load_brain_data(os.path.join(brain_dir, 'eeg_train_data_within.mat'))
brain_unseen, label_unseen = load_brain_data(os.path.join(brain_dir, 'eeg_test_data.mat'))

# Image features
image_seen = sio.loadmat(os.path.join(image_dir_seen, 'feat_pca_train.mat'))['data'].astype('double') * 50.0
image_unseen = sio.loadmat(os.path.join(image_dir_unseen, 'feat_pca_test.mat'))['data'].astype('double') * 50.0

# Text features
text_seen = sio.loadmat(os.path.join(text_dir_seen, 'text_feat_train.mat'))['data'].astype('double') * 2.0
text_unseen = sio.loadmat(os.path.join(text_dir_unseen, 'text_feat_test.mat'))['data'].astype('double') * 2.0

# -----------------------------
# 3. Zero-Shot Split
# Keep only a subset of categories (e.g., first 20 classes)
# -----------------------------
num_classes = 20
index_seen = np.where(label_seen.ravel() < num_classes + 1)[0]
index_unseen = np.where(label_unseen.ravel() < num_classes + 1)[0]

brain_seen = brain_seen[index_seen, :]
image_seen = image_seen[index_seen, :]
text_seen = text_seen[index_seen, :]
label_seen = label_seen[index_seen]

brain_unseen = brain_unseen[index_unseen, :]
image_unseen = image_unseen[index_unseen, :]
text_unseen = text_unseen[index_unseen, :]
label_unseen = label_unseen[index_unseen]

# -----------------------------
# 4. Optional: Minimal Federated Simulation
# -----------------------------
# Split seen data into 3 pseudo-clients (optional)
num_clients = 3
client_data = np.array_split(np.concatenate([image_seen, text_seen], axis=1), num_clients)
client_labels = np.array_split(label_seen.ravel(), num_clients)
# For baseline, you can ignore this and train on the full seen set

# -----------------------------
# 5. Multi-Modal Pairing for scikit-learn
# Concatenate image + text embeddings
# -----------------------------
X_seen = np.concatenate([image_seen, text_seen], axis=1)
X_unseen = np.concatenate([image_unseen, text_unseen], axis=1)
y_seen = label_seen.ravel()
y_unseen = label_unseen.ravel()

# Standardize features
scaler = StandardScaler()
X_seen = scaler.fit_transform(X_seen)
X_unseen = scaler.transform(X_unseen)

# -----------------------------
# 6. Train/Test summary
# -----------------------------
print("=== Seen (train) Data ===")
print("Samples:", X_seen.shape[0], "Features:", X_seen.shape[1])
print("Labels shape:", y_seen.shape)
print("\n=== Unseen (test) Data ===")
print("Samples:", X_unseen.shape[0], "Features:", X_unseen.shape[1])
print("Labels shape:", y_unseen.shape)

# Added: class counts / imbalance stats
import pandas as pd

def print_class_counts(y, label="Dataset"):
    unique, counts = np.unique(y, return_counts=True)
    df = pd.DataFrame({"Class": unique, "Count": counts})
    print(f"\n=== {label} Class Counts ===")
    print(df)
    print("Total samples:", y.shape[0])

print_class_counts(y_seen, label="Seen (Train) Data")
print_class_counts(y_unseen, label="Unseen (Test) Data")

# -----------------------------
# Document Zero-Shot Split
# -----------------------------
CSeen = np.unique(y_seen)
CUnseen = np.unique(y_unseen)

df_split = pd.DataFrame({
    "CSeen": pd.Series(CSeen),
    "CUnseen": pd.Series(CUnseen)
})
print("\n=== Zero-Shot Split Table ===")
print(df_split)

mmbra.data_analysis_example(torch.from_numpy(brain_seen), torch.from_numpy(image_seen), torch.from_numpy(text_seen))
mmbra.data_visualization_example(torch.from_numpy(label_seen))
mmbra.data_visualization_example(torch.from_numpy(label_unseen))

# -----------------------------
# Baseline classifiers (integrated; does not remove previous functionality)
# -----------------------------
# X_seen, X_unseen, y_seen, y_unseen are prepared above

# Optional: visualize shapes
print("Training data shape:", X_seen.shape)
print("Test data shape:", X_unseen.shape)

# ZSL mechanics: cosine similarity (unseen vs seen)
cos_sim_matrix = cosine_similarity(X_unseen, X_seen)
print("Cosine similarity matrix shape (unseen vs seen):", cos_sim_matrix.shape)

# Choose classifier (default: linear SVM with balanced weights)
clf = SVC(kernel='linear', class_weight='balanced', probability=False)
# Alternative examples (commented)
# clf = LogisticRegression(max_iter=1000, class_weight='balanced')
# clf = KNeighborsClassifier(n_neighbors=5)

# Train on full seen data (baseline)
clf.fit(X_seen, y_seen)

# Predict on unseen/test set
y_pred = clf.predict(X_unseen)

# Evaluate
acc = accuracy_score(y_unseen, y_pred)
f1_macro = f1_score(y_unseen, y_pred, average='macro')

print("=== Baseline Model Evaluation ===")
print("Accuracy:", acc)
print("Macro F1:", f1_macro)

# Quick inspection
print("First 10 predicted labels:", y_pred[:10])
print("First 10 true labels:   ", y_unseen[:10])

# =============================
# 1. Zero-Shot Testing
# =============================
# Reminder: X_unseen contains features of classes NOT seen during training
# y_unseen contains the true labels of these unseen classes
# clf is the trained scikit-learn model from baseline

# -----------------------------
# 1a. Make Predictions on Unseen Classes
# -----------------------------
y_pred = clf.predict(X_unseen)  # baseline uses predict function

# -----------------------------
# 1b. Evaluate Predictions
# -----------------------------
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Accuracy and Macro F1
acc = accuracy_score(y_unseen, y_pred)
f1_macro = f1_score(y_unseen, y_pred, average='macro')

print("=== Zero-Shot Testing on Unseen Classes ===")
print("Accuracy:", acc)
print("Macro F1:", f1_macro)

# -----------------------------
# 1c. Optional: Confusion Matrix / Detailed Analysis
# -----------------------------
cm = confusion_matrix(y_unseen, y_pred)
print("\nConfusion Matrix:\n", cm)

print("\nClassification Report:\n")
print(classification_report(y_unseen, y_pred))

# -----------------------------
# 2. Explain ZSL Principle (logging note)
# -----------------------------
# This demonstrates zero-shot learning:
# - The model has never seen any examples from unseen classes (X_unseen)
# - It predicts based on patterns learned from seen classes
# - Any correct predictions are due to generalization across embeddings (image + text)

# =============================
# Step 1 — Compute / Obtain Semantic Prototypes for Classes
# =============================
# Use text embeddings to compute class prototypes (semantic embeddings)
# For each class c, compute prototype s_c = mean(text_embeddings for class c)
# Prototypes needed for BOTH seen and unseen classes (allowed in ZSL)

def compute_prototypes(text_feats, labels):
    """
    Compute semantic prototypes (mean embeddings) for each class.

    Args:
        text_feats: (n_samples, feature_dim) text feature matrix
        labels: (n_samples,) class labels

    Returns:
        prototypes: dict mapping class_id -> prototype_vector
    """
    classes = np.unique(labels)
    prototypes = {}
    for c in classes:
        mask = labels == c
        prototypes[c] = text_feats[mask].mean(axis=0)
    return prototypes

# Compute prototypes for seen and unseen classes
prototypes_seen = compute_prototypes(text_seen, y_seen)
prototypes_unseen = compute_prototypes(text_unseen, y_unseen)

print("\n=== Semantic Prototypes (Class Embeddings) ===")
print("Number of seen class prototypes:", len(prototypes_seen))
print("Number of unseen class prototypes:", len(prototypes_unseen))
print("Prototype vector dimension:", list(prototypes_seen.values())[0].shape[0])

# Display first prototype as example
first_class = list(prototypes_seen.keys())[0]
print(f"\nExample prototype for seen class {first_class}:")
print("Shape:", prototypes_seen[first_class].shape)
print("First 5 values:", prototypes_seen[first_class][:5])

# =============================
# Step 2 — Define Mapping Network (Encoder) for Local Training
# =============================
# Learn a linear mapping from concatenated features (image+text or brain+image+text)
# to semantic space: W*x ≈ s_y
# Use ridge regression to map feature space → semantic prototype space

from sklearn.linear_model import Ridge

def create_semantic_targets(labels, prototypes):
    """
    Create semantic target vectors for each sample based on its class label.

    Args:
        labels: (n_samples,) class labels
        prototypes: dict mapping class_id -> prototype_vector

    Returns:
        S_targets: (n_samples, sem_dim) semantic embedding targets
    """
    sem_dim = list(prototypes.values())[0].shape[0]
    S_targets = np.zeros((len(labels), sem_dim))
    for i, label in enumerate(labels):
        S_targets[i] = prototypes[label]
    return S_targets

# Create semantic targets for seen classes (training targets)
S_seen = create_semantic_targets(y_seen, prototypes_seen)

print("\n=== Semantic Target Vectors ===")
print("Seen semantic targets shape:", S_seen.shape)
print("First sample semantic target (first 5 dims):", S_seen[0, :5])

# =============================
# Train Local Mapping Network (Ridge Regression)
# =============================
# Ridge regression: W = argmin ||X*W - S||^2 + lambda||W||^2
# Maps concatenated features → semantic prototype space

print("\n=== Training Local Mapping Network (Ridge Regression) ===")

ridge = Ridge(alpha=1.0)  # L2 regularization parameter
ridge.fit(X_seen, S_seen)

W_mapping = ridge.coef_.T  # shape: (input_dim, semantic_dim)
b_mapping = ridge.intercept_  # shape: (semantic_dim,)

print("Mapping matrix W shape:", W_mapping.shape)
print("Mapping bias b shape:", b_mapping.shape)
print("Input dimension:", X_seen.shape[1])
print("Semantic dimension:", S_seen.shape[1])

# Verify mapping quality on training data
S_pred_seen = X_seen @ W_mapping + b_mapping
mse_train = np.mean((S_pred_seen - S_seen) ** 2)
print(f"\nTraining MSE (feature → semantic):", mse_train)

# =============================
# Optional: Visualize Mapping Quality
# =============================
# Compare predicted vs true semantic vectors (first 5 samples)
print("\n=== Mapping Quality Check (first 3 samples) ===")
for i in range(min(3, len(y_seen))):
    print(f"Sample {i} (class {y_seen[i]}):")
    print("  True semantic (first 5):", S_seen[i, :5])
    print("  Pred semantic (first 5):", S_pred_seen[i, :5])

# =============================
# Step 3 — Simulate Federated Training (FedAvg) with Clients
# =============================
# For each round:
#   - Each client k trains local mapping W_k on its local data
#   - Server aggregates: W_global = weighted average of W_k
#   - (Optional) broadcast new W and repeat for multiple rounds

from numpy import ndarray
# Use the PyTorch-based FedAvg ZSL trainer defined earlier
def fedavg_train(client_splits_X, client_splits_y, prototypes_dict, rounds=3, lr=0.01, epochs=50):
    """Wrapper to call fedavg_train_zsl with explicit lr/epochs."""
    return fedavg_train_zsl(client_splits_X, client_splits_y, prototypes_dict, rounds=rounds, lr=lr, epochs=epochs)

# =============================
# Execute FedAvg Training
# =============================
# Prepare client data splits (from earlier federated simulation)
client_X = [client_data[i] for i in range(num_clients)]
client_y = [client_labels[i] for i in range(num_clients)]

print("\n=== Client Data Distribution ===")
for k, (Xk, yk) in enumerate(zip(client_X, client_y)):
    print(f"Client {k}: {len(yk)} samples, {Xk.shape[1]} features")

# Run FedAvg with 3 communication rounds
W_global = fedavg_train(client_X, client_y, prototypes_seen, rounds=3, lr=0.01, epochs=50)

print("\n=== Global Mapping Network ===")
print("W_global shape:", W_global.shape)
print("First 5 values of W_global[:5, 0]:", W_global[:5, 0])

# =============================
# Optional: Verify Global Mapping on Full Seen Data
# =============================
S_pred_global = X_seen @ W_global
mse_global = np.mean((S_pred_global - S_seen) ** 2)
print(f"\nGlobal model MSE on full seen data:", mse_global)

# =============================
# Step 4 — Zero-Shot Prediction Using Global Mapping
# =============================
# For each test sample x (brain+image concat):
#   1. Compute embedding z = x @ W_global (map to semantic space)
#   2. Predict unseen class as prototype with highest cosine similarity to z

print("\n=== Zero-Shot Prediction with Learned Mapping ===")

# Map X_unseen to semantic space using global mapping
Z_unseen = X_unseen @ W_global  # shape: (n_unseen, sem_dim)

print("Mapped unseen embeddings Z_unseen shape:", Z_unseen.shape)
print("First sample embedding (first 5 dims):", Z_unseen[0, :5])

# Prepare prototype matrix for unseen classes
labels_unseen_sorted = sorted(prototypes_unseen.keys())
P_unseen = np.vstack([prototypes_unseen[c] for c in labels_unseen_sorted])

print("\nUnseen class prototypes P_unseen shape:", P_unseen.shape)
print("Number of unseen classes:", len(labels_unseen_sorted))
print("Unseen class labels:", labels_unseen_sorted)

# Compute cosine similarity: unseen samples vs unseen prototypes
# sim[i, j] = cosine_sim(Z_unseen[i], P_unseen[j])
sim_matrix = cosine_similarity(Z_unseen, P_unseen)  # shape: (n_unseen, num_unseen_classes)

print("\nCosine similarity matrix shape:", sim_matrix.shape)
print("Similarity matrix (first 5x5):\n", sim_matrix[:5, :5])

# Predict by argmax cosine similarity
indices = sim_matrix.argmax(axis=1)
y_pred_zsl = np.array([labels_unseen_sorted[i] for i in indices])

print("\n=== ZSL Predictions (First 10 Samples) ===")
print("Predicted labels:", y_pred_zsl[:10])
print("True labels:    ", y_unseen[:10])

# =============================
# Evaluate Zero-Shot Learning Performance
# =============================
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

acc_zsl = accuracy_score(y_unseen, y_pred_zsl)
f1_zsl = f1_score(y_unseen, y_pred_zsl, average='macro', zero_division=0)

print("\n" + "="*60)
print("=== ZERO-SHOT LEARNING EVALUATION ===")
print("="*60)
print(f"Accuracy:  {acc_zsl:.4f}")
print(f"Macro F1:  {f1_zsl:.4f}")
print("="*60)

# Confusion Matrix
cm_zsl = confusion_matrix(y_unseen, y_pred_zsl)
print("\nConfusion Matrix shape:", cm_zsl.shape)
print("Confusion Matrix (first 5x5):\n", cm_zsl[:5, :5])

# Detailed Classification Report
print("\nClassification Report:\n")
print(classification_report(y_unseen, y_pred_zsl, zero_division=0))

# =============================
# Optional: Per-Class Analysis
# =============================
print("\n=== Per-Class Performance ===")
unique_classes = np.unique(y_unseen)
for cls in sorted(unique_classes)[:5]:  # show first 5 classes
    mask = y_unseen == cls
    if mask.sum() > 0:
        acc_cls = accuracy_score(y_unseen[mask], y_pred_zsl[mask])
        print(f"Class {cls}: accuracy = {acc_cls:.4f} ({mask.sum()} samples)")

# =============================
# Comparison: Baseline SVM vs Zero-Shot Learning
# =============================
print("\n" + "="*60)
print("=== COMPARISON: BASELINE SVM vs ZERO-SHOT (FedAvg) ===")
print("="*60)
print(f"Baseline SVM Accuracy:       {accuracy_score(y_unseen, y_pred):.4f}")
print(f"Zero-Shot (FedAvg) Accuracy: {acc_zsl:.4f}")
print(f"Improvement:                 {acc_zsl - accuracy_score(y_unseen, y_pred):+.4f}")
print("="*60)

# -----------------------------
# PCA and t-SNE visualization of combined embeddings
# -----------------------------
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_embeddings(X, y, title="Embedding Visualization", method="PCA", perplexity=30, max_samples=2000):
    """
    Reduce X to 2D using PCA or t-SNE and plot colored by y.
    For large X, a random subset up to max_samples is used to keep plotting responsive.
    """
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
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    else:
        raise ValueError("Method must be 'PCA' or 't-SNE'")

    X_reduced = reducer.fit_transform(X_sub)
    plt.figure(figsize=(8,6))
    for cls in np.unique(y_sub):
        mask = (y_sub == cls)
        plt.scatter(
            X_reduced[mask, 0],
            X_reduced[mask, 1],
            label=f"Class {cls}",
            alpha=0.6,
            s=10
        )
    plt.title(f"{title} ({method})")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()

# Visualize seen data (PCA + t-SNE)
try:
    plot_embeddings(X_seen, y_seen, title="Seen Data Embeddings", method="PCA")
    plot_embeddings(X_seen, y_seen, title="Seen Data Embeddings", method="t-SNE", perplexity=30)
except Exception as e:
    print("Embedding visualization (seen) failed:", e)

# Visualize unseen data (PCA + t-SNE)
try:
    plot_embeddings(X_unseen, y_unseen, title="Unseen Data Embeddings", method="PCA")
    plot_embeddings(X_unseen, y_unseen, title="Unseen Data Embeddings", method="t-SNE", perplexity=30)
except Exception as e:
    print("Embedding visualization (unseen) failed:", e)