from collections.abc import Sequence
from typing import Any
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from ml.models.schemas import TrainingConfig


class PyTorchLinearProbe(nn.Module):
    """
    Lightweight trainable linear classification head for DistilBERT representation fine-tuning.
    """

    def __init__(self, in_features: int, num_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class PyTorchNeuralClassifier(nn.Module):
    """
    Trainable multi-layer neural probe with dropout and non-linearity.
    """

    def __init__(self, in_features: int, num_classes: int, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TrainableDownstreamClassifier:
    """
    Trainable downstream text classification model compatible with DistilBERT representations.
    Executes real optimization using AdamW and Cross-Entropy loss with sample weight support.
    """

    def __init__(self, config: TrainingConfig | None = None) -> None:
        self.config = config or TrainingConfig()
        self.classes_: list[str] = []
        self._model: nn.Module | LogisticRegression | None = None
        self._single_class_default: str | None = None
        self._device: torch.device = torch.device(
            "cuda" if (self.config.device == "auto" and torch.cuda.is_available()) or self.config.device == "cuda" else "cpu"
        )

    def fit(
        self,
        X: np.ndarray,
        y: Sequence[Any],
        sample_weights: Sequence[float] | None = None,
        config: TrainingConfig | None = None,
    ) -> "TrainableDownstreamClassifier":
        cfg = config or self.config
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Training data and labels cannot be empty.")
        if len(X) != len(y):
            raise ValueError(f"Length mismatch: X={len(X)}, y={len(y)}")

        str_labels = [str(lbl) for lbl in y]
        unique_classes = sorted(list(set(str_labels)))
        self.classes_ = unique_classes

        if len(unique_classes) < 2:
            self._single_class_default = unique_classes[0]
            self._model = None
            return self

        self._single_class_default = None

        if cfg.model_type == "logistic_regression":
            clf = LogisticRegression(
                max_iter=1000,
                random_state=cfg.seed,
            )
            sw_arr = np.array(sample_weights, dtype=np.float64) if sample_weights is not None else None
            clf.fit(X, str_labels, sample_weight=sw_arr)
            self._model = clf
            return self

        # PyTorch Neural Probe Training
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

        in_features = X.shape[1]
        num_classes = len(unique_classes)
        label_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        y_indices = np.array([label_to_idx[lbl] for lbl in str_labels], dtype=np.int64)

        if cfg.model_type == "neural_head":
            model = PyTorchNeuralClassifier(in_features, num_classes).to(self._device)
        else:
            model = PyTorchLinearProbe(in_features, num_classes).to(self._device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        criterion = nn.CrossEntropyLoss(reduction="none")

        X_tensor = torch.tensor(X, dtype=torch.float32, device=self._device)
        y_tensor = torch.tensor(y_indices, dtype=torch.long, device=self._device)
        weights_tensor = (
            torch.tensor(sample_weights, dtype=torch.float32, device=self._device)
            if sample_weights is not None
            else torch.ones(len(X), dtype=torch.float32, device=self._device)
        )

        n_samples = len(X)
        batch_size = min(cfg.batch_size, n_samples)
        model.train()

        for _ in range(cfg.epochs):
            # Shuffle indices deterministically per epoch
            perm = torch.randperm(n_samples, device=self._device)
            for start_idx in range(0, n_samples, batch_size):
                batch_indices = perm[start_idx : start_idx + batch_size]
                b_x = X_tensor[batch_indices]
                b_y = y_tensor[batch_indices]
                b_w = weights_tensor[batch_indices]

                optimizer.zero_grad()
                logits = model(b_x)
                losses = criterion(logits, b_y)
                weighted_loss = torch.mean(losses * b_w)
                weighted_loss.backward()
                optimizer.step()

        model.eval()
        self._model = model
        return self

    def predict(self, X: np.ndarray) -> list[str]:
        if len(X) == 0:
            return []

        if self._single_class_default is not None:
            return [self._single_class_default] * len(X)

        if isinstance(self._model, LogisticRegression):
            return self._model.predict(X).tolist()

        if isinstance(self._model, nn.Module):
            with torch.inference_mode():
                X_tensor = torch.tensor(X, dtype=torch.float32, device=self._device)
                logits = self._model(X_tensor)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                return [self.classes_[idx] for idx in preds]

        raise RuntimeError("Classifier has not been fitted.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.empty((0, len(self.classes_)))

        if self._single_class_default is not None:
            return np.ones((len(X), 1), dtype=np.float32)

        if isinstance(self._model, LogisticRegression):
            return self._model.predict_proba(X)

        if isinstance(self._model, nn.Module):
            with torch.inference_mode():
                X_tensor = torch.tensor(X, dtype=torch.float32, device=self._device)
                logits = self._model(X_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                return probs

        raise RuntimeError("Classifier has not been fitted.")
