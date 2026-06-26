from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

EveKind = Literal["temporal_graph_eve", "sequence_transformer_eve"]


@dataclass(frozen=True)
class NeuralEveMetrics:
    auc: float
    balanced_accuracy: float
    eer: float


class NeuralEveDetector(nn.Module):
    def __init__(
        self,
        *,
        source_count: int,
        action_count: int,
        context_length: int,
        embedding_dim: int,
        hidden_dim: int,
        kind: EveKind,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.kind = kind
        self.context_length = context_length
        self.source_embedding = nn.Embedding(source_count, embedding_dim, padding_idx=0)
        self.action_embedding = nn.Embedding(action_count, embedding_dim, padding_idx=0)
        self.time_projection = nn.Linear(1, embedding_dim)
        if kind == "temporal_graph_eve":
            layers: list[nn.Module] = []
            input_dim = 3 * embedding_dim
            for _ in range(num_layers):
                layers.append(nn.Linear(input_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                input_dim = hidden_dim
            self.encoder = nn.Sequential(*layers)
            classifier_input = hidden_dim
        elif kind == "sequence_transformer_eve":
            layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=4,
                dim_feedforward=hidden_dim,
                batch_first=True,
                dropout=dropout,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            classifier_input = 3 * embedding_dim
        else:
            raise ValueError(f"Unsupported neural Eve kind: {kind}")
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        sources: torch.Tensor,
        actions: torch.Tensor,
        contexts: torch.Tensor,
        gaps: torch.Tensor,
    ) -> torch.Tensor:
        source = self.source_embedding(sources)
        action = self.action_embedding(actions)
        time = self.time_projection(gaps.unsqueeze(1))
        if self.kind == "temporal_graph_eve":
            hidden = self.encoder(torch.cat([source, action, time], dim=1))
        else:
            sequence = self.action_embedding(contexts)
            encoded = self.encoder(sequence)[:, -1, :]
            hidden = torch.cat([source, encoded, time], dim=1)
        return self.classifier(hidden).squeeze(1)


def fit_and_score_neural_eve(
    validation: pd.DataFrame,
    test: pd.DataFrame,
    *,
    kind: EveKind,
    context_length: int,
    embedding_dim: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    patience: int = 5,
) -> NeuralEveMetrics:
    torch.manual_seed(seed)
    np.random.seed(seed)
    source_to_id = _vocabulary(pd.concat([validation["source"], test["source"]], ignore_index=True))
    action_to_id = _vocabulary(pd.concat([validation["action"], test["action"]], ignore_index=True))
    train_tensors = _examples(validation, source_to_id, action_to_id, context_length)
    test_tensors = _examples(test, source_to_id, action_to_id, context_length)
    model = NeuralEveDetector(
        source_count=len(source_to_id) + 1,
        action_count=len(action_to_id) + 1,
        context_length=context_length,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        kind=kind,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(*train_tensors), batch_size=batch_size, shuffle=True)

    best_loss = float("inf")
    best_state: dict | None = None
    epochs_without_improvement = 0

    for _ in range(epochs):
        model.train()
        epoch_losses = []
        for sources, actions, contexts, gaps, labels in loader:
            optimizer.zero_grad()
            loss = criterion(model(sources, actions, contexts, gaps), labels.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        scheduler.step(np.mean(epoch_losses))

        if np.mean(epoch_losses) < best_loss:
            best_loss = np.mean(epoch_losses)
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        sources, actions, contexts, gaps, labels = test_tensors
        scores = torch.sigmoid(model(sources, actions, contexts, gaps)).numpy()
    labels_np = labels.numpy()
    predictions = (scores >= 0.5).astype(int)
    return NeuralEveMetrics(
        auc=float(roc_auc_score(labels_np, scores)),
        balanced_accuracy=float(balanced_accuracy_score(labels_np, predictions)),
        eer=_eer(labels_np, scores),
    )


def metrics_to_dict(metrics: NeuralEveMetrics) -> dict[str, float]:
    return {
        "auc": metrics.auc,
        "balanced_accuracy": metrics.balanced_accuracy,
        "eer": metrics.eer,
    }


def _vocabulary(values: pd.Series) -> dict[str, int]:
    counts = Counter(str(value) for value in values)
    return {value: index + 1 for index, (value, _) in enumerate(counts.most_common())}


def _examples(
    frame: pd.DataFrame,
    source_to_id: dict[str, int],
    action_to_id: dict[str, int],
    context_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    sources = []
    actions = []
    contexts = []
    gaps = []
    labels = []
    history_by_stream: dict[tuple[str, int], list[int]] = {}
    previous_time_by_stream: dict[tuple[str, int], float] = {}
    ordered = frame.sort_values(["timestamp", "pair_id", "label"], kind="stable")
    for source, action, timestamp, label in ordered[["source", "action", "timestamp", "label"]].itertuples(
        index=False
    ):
        source_key = str(source)
        action_id = action_to_id.get(str(action), 0)
        label_value = int(label)
        stream_key = (source_key, label_value)
        history = history_by_stream.setdefault(stream_key, [])
        contexts.append(([0] * context_length + history + [action_id])[-context_length:])
        sources.append(source_to_id.get(source_key, 0))
        actions.append(action_id)
        gap = float(timestamp) - previous_time_by_stream.get(stream_key, float(timestamp))
        gaps.append(math.log1p(max(0.0, gap)))
        labels.append(label_value)
        history.append(action_id)
        previous_time_by_stream[stream_key] = float(timestamp)
    return (
        torch.tensor(sources, dtype=torch.long),
        torch.tensor(actions, dtype=torch.long),
        torch.tensor(contexts, dtype=torch.long),
        torch.tensor(gaps, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )


def _eer(labels: np.ndarray, scores: np.ndarray) -> float:
    false_positive_rate, true_positive_rate, _ = roc_curve(labels, scores)
    false_negative_rate = 1.0 - true_positive_rate
    index = int(np.argmin(np.abs(false_positive_rate - false_negative_rate)))
    return float((false_positive_rate[index] + false_negative_rate[index]) / 2.0)
