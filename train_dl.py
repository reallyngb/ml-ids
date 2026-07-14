"""
train_dl.py
-----------
Phase 6 of the IDS pipeline: Deep Learning via a PyTorch Autoencoder.

WHY AN AUTOENCODER FOR IDS:
Every model in train_ml.py is SUPERVISED — it needs labeled attack examples
to learn from. That's a problem for zero-day attacks: if an attack type
never appeared in training data, a supervised model has never seen its
pattern and may misclassify it as benign.

An autoencoder sidesteps this by training ONLY on BENIGN traffic. It learns
to compress ("encode") a benign flow into a small latent vector and then
reconstruct ("decode") the original from that compressed form. Because it
has only ever seen normal traffic, it becomes very good at reconstructing
normal flows and comparatively BAD at reconstructing anything unusual
(including attack types it has never seen). We flag high reconstruction
error as anomalous. This is Phase 12's zero-day detection strategy.

ARCHITECTURE (configured in config.yaml -> autoencoder):
  Input (n_features) -> 64 -> 32 -> 16 (bottleneck/latent) -> 32 -> 64 -> Input
  - Encoder progressively compresses; decoder mirrors it back out.
  - ReLU activations between layers (standard, avoids vanishing gradients).
  - Linear (no activation) on the final output layer, since our scaled
    features can be negative (StandardScaler), so Sigmoid/ReLU would clip.
  - Loss: MSE between input and reconstruction (standard for autoencoders).
  - Optimizer: Adam (adaptive learning rate, robust default for this task).
  - Early stopping: stop training if validation loss doesn't improve for
    `early_stopping_patience` epochs, to avoid overfitting the reconstruction.

THRESHOLD SELECTION:
  After training, we compute reconstruction error on a benign validation
  set and set the anomaly threshold at the `threshold_percentile` (default
  95th percentile) of that error distribution. Any flow -- at inference
  time -- with reconstruction error above this threshold is flagged
  anomalous. Raising the percentile lowers false positives but risks
  missing subtler attacks; this is a tunable operating point (Phase 12).

Run directly: `python src/models/train_dl.py`
Requires `torch` (pip install torch) - not required for the rest of the
pipeline, so it's kept as an optional/separate step.
"""

import sys
from pathlib import Path

import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.config_loader import get_config, resolve_path
from src.utils.logger import get_logger
from src.data.preprocessing import run_preprocessing

logger = get_logger(__name__)


def _lazy_import_torch():
    """Import torch only when this module actually runs, so the rest of the
    project doesn't hard-depend on a multi-GB torch install."""
    try:
        import torch
        import torch.nn as nn
        return torch, nn
    except ImportError as e:
        raise ImportError(
            "PyTorch is required for train_dl.py. Install it with:\n"
            "  pip install torch\n"
            "(kept optional/separate from requirements.txt's core install "
            "because torch is large; see README.md)"
        ) from e


def build_autoencoder(input_dim: int, cfg: dict):
    torch, nn = _lazy_import_torch()
    ae_cfg = cfg["autoencoder"]
    hidden_dims = ae_cfg["hidden_dims"]
    latent_dim = ae_cfg["latent_dim"]

    class Autoencoder(nn.Module):
        def __init__(self):
            super().__init__()
            # Encoder: input -> hidden_dims... -> latent_dim
            enc_layers = []
            dims = [input_dim] + hidden_dims + [latent_dim]
            for i in range(len(dims) - 1):
                enc_layers.append(nn.Linear(dims[i], dims[i + 1]))
                enc_layers.append(nn.ReLU())
            self.encoder = nn.Sequential(*enc_layers[:-1])  # no ReLU on latent output

            # Decoder: mirror image of the encoder
            dec_dims = [latent_dim] + hidden_dims[::-1] + [input_dim]
            dec_layers = []
            for i in range(len(dec_dims) - 1):
                dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
                if i < len(dec_dims) - 2:
                    dec_layers.append(nn.ReLU())
            # Final layer is Linear only (no activation) -- see docstring.
            self.decoder = nn.Sequential(*dec_layers)

        def forward(self, x):
            z = self.encoder(x)
            return self.decoder(z)

    return Autoencoder()


def train_autoencoder(data: dict, cfg: dict):
    torch, nn = _lazy_import_torch()
    ae_cfg = cfg["autoencoder"]
    seed = cfg["project"]["random_seed"]
    torch.manual_seed(seed)

    # Train ONLY on benign flows from the training split (y_train == 0)
    X_train_all, y_train_all = data["X_train"], data["y_train"]
    X_train_benign = X_train_all[np.array(y_train_all) == 0]

    X_val_all, y_val_all = data["X_val"], data["y_val"]
    X_val_benign = X_val_all[np.array(y_val_all) == 0]

    input_dim = X_train_benign.shape[1]
    model = build_autoencoder(input_dim, cfg)

    X_train_t = torch.tensor(X_train_benign, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_benign, dtype=torch.float32)

    optimizer = torch.optim.Adam(model.parameters(), lr=ae_cfg["learning_rate"])
    criterion = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(X_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=ae_cfg["batch_size"], shuffle=True)

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in range(ae_cfg["epochs"]):
        model.train()
        train_losses = []
        for (batch,) in loader:
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_recon = model(X_val_t)
            val_loss = criterion(val_recon, X_val_t).item()

        logger.info(f"Epoch {epoch+1}/{ae_cfg['epochs']} - train_loss={np.mean(train_losses):.5f} val_loss={val_loss:.5f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= ae_cfg["early_stopping_patience"]:
                logger.info(f"Early stopping at epoch {epoch+1} (no improvement for {patience_counter} epochs)")
                break

    model.load_state_dict(best_state)
    return model


def compute_reconstruction_error(model, X: np.ndarray):
    torch, _ = _lazy_import_torch()
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        recon = model(X_t)
        # Per-sample MSE (mean over features), not a single scalar over the batch
        errors = torch.mean((recon - X_t) ** 2, dim=1).numpy()
    return errors


def select_threshold(model, X_val_benign: np.ndarray, cfg: dict) -> float:
    errors = compute_reconstruction_error(model, X_val_benign)
    percentile = cfg["autoencoder"]["threshold_percentile"]
    threshold = np.percentile(errors, percentile)
    logger.info(f"Anomaly threshold set at {percentile}th percentile of benign val error = {threshold:.6f}")
    return threshold


def evaluate_autoencoder(model, threshold: float, X_test, y_test):
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

    errors = compute_reconstruction_error(model, X_test)
    y_pred = (errors > threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    metrics = {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "detection_rate": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "false_positive_rate": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
    }
    logger.info(f"Autoencoder TEST -> {metrics}")
    return metrics


def save_pytorch_model(model, threshold: float, cfg: dict):
    torch, _ = _lazy_import_torch()
    model_dir = resolve_path(cfg["paths"]["pytorch_model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_dir / "autoencoder.pt")
    joblib.dump({"threshold": threshold, "input_dim": next(model.encoder.parameters()).shape[1]},
                model_dir / "autoencoder_meta.joblib")
    logger.info(f"Autoencoder saved to {model_dir}/autoencoder.pt (threshold={threshold:.6f})")


if __name__ == "__main__":
    cfg = get_config()
    data = run_preprocessing(binary=True)

    model = train_autoencoder(data, cfg)

    X_val_all, y_val_all = data["X_val"], data["y_val"]
    X_val_benign = X_val_all[np.array(y_val_all) == 0]
    threshold = select_threshold(model, X_val_benign, cfg)

    evaluate_autoencoder(model, threshold, data["X_test"], data["y_test"])
    save_pytorch_model(model, threshold, cfg)
