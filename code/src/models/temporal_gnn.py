from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class TemporalGNNMetrics:
    rows: int
    mean_nll_bits: float
    perplexity: float
    top1_accuracy: float
    unknown_target_fraction: float
    expected_calibration_error: float
    brier_score: float


class SinusoidalTimeEncoding(nn.Module):
    """Learnable sinusoidal time encoding for inter-event gaps."""

    def __init__(self, embedding_dim: int, max_scale: float = 1e4) -> None:
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even for sinusoidal encoding")
        self.embedding_dim = embedding_dim
        self.max_scale = max_scale
        self.projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, gaps: torch.Tensor) -> torch.Tensor:
        # gaps: (batch,)
        position = gaps.unsqueeze(1)  # (batch, 1)
        div_term = torch.exp(
            torch.arange(0, self.embedding_dim, 2, device=gaps.device, dtype=gaps.dtype)
            * (-math.log(self.max_scale) / self.embedding_dim)
        )
        pe = torch.zeros(position.size(0), self.embedding_dim, device=gaps.device, dtype=gaps.dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return self.projection(pe)


class GraphAttentionReadout(nn.Module):
    """Single-head attention between context representation and candidate destinations."""

    def __init__(self, context_dim: int, destination_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(context_dim, context_dim)
        self.key = nn.Linear(destination_dim, context_dim)
        self.scale = math.sqrt(context_dim)

    def forward(self, context: torch.Tensor, destinations: torch.Tensor) -> torch.Tensor:
        # context: (batch, context_dim), destinations: (num_destinations, destination_dim)
        query = self.query(context)  # (batch, context_dim)
        key = self.key(destinations).T  # (context_dim, num_destinations)
        key = key.unsqueeze(0).expand(context.size(0), -1, -1)  # (batch, context_dim, num_destinations)
        scores = torch.bmm(query.unsqueeze(1), key).squeeze(1) / self.scale  # (batch, num_destinations)
        return scores


class TemporalGraphCoverModel(nn.Module):
    """Causal temporal-graph cover model with node memory and attention readout."""

    def __init__(
        self,
        *,
        sources: int,
        destinations: int,
        embedding_dim: int,
        hidden_dim: int,
        memory_dim: int | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        memory_dim = memory_dim or embedding_dim
        self.source_embedding = nn.Embedding(sources, embedding_dim, padding_idx=0)
        self.previous_destination_embedding = nn.Embedding(destinations, embedding_dim, padding_idx=0)
        self.time_encoding = SinusoidalTimeEncoding(embedding_dim)
        self.input_projection = nn.Linear(3 * embedding_dim, hidden_dim)
        self.memory_gru = nn.GRUCell(hidden_dim, memory_dim)
        self.norm1 = nn.LayerNorm(memory_dim)
        self.dropout = nn.Dropout(dropout)
        self.destination_embedding = nn.Embedding(destinations, embedding_dim)
        self.attention = GraphAttentionReadout(memory_dim, embedding_dim)
        self.destination_bias = nn.Parameter(torch.zeros(destinations))
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(
        self,
        sources: torch.Tensor,
        previous_destinations: torch.Tensor,
        gaps: torch.Tensor,
        memory: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_emb = self.source_embedding(sources)
        prev_emb = self.previous_destination_embedding(previous_destinations)
        time_emb = self.time_encoding(gaps)
        combined = torch.cat([source_emb, prev_emb, time_emb], dim=1)
        projected = torch.relu(self.input_projection(combined))
        if memory is None:
            memory = torch.zeros(projected.size(0), self.memory_gru.hidden_size, device=projected.device)
        new_memory = self.memory_gru(projected, memory)
        new_memory = self.norm1(new_memory)
        new_memory = self.dropout(new_memory)
        scores = self.attention(new_memory, self.destination_embedding.weight)
        logits = scores + self.destination_bias.unsqueeze(0)
        return logits / self.temperature, new_memory


def train_and_evaluate_temporal_gnn(
    frame: pd.DataFrame,
    *,
    embedding_dim: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    patience: int = 5,
    dropout: float = 0.2,
) -> dict[str, TemporalGNNMetrics]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = frame.loc[frame["split"] == "train"]
    source_to_id = _vocabulary(pd.concat([train["source"], train["destination"]], ignore_index=True))
    destination_to_id = _vocabulary(train["destination"])
    model = TemporalGraphCoverModel(
        sources=len(source_to_id) + 1,
        destinations=len(destination_to_id) + 1,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    train_dataset = _tensor_dataset(train, source_to_id, destination_to_id)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()

    best_val_nll = float("inf")
    best_state: dict | None = None
    epochs_without_improvement = 0

    for epoch in range(epochs):
        model.train()
        for sources, previous, gaps, targets in train_loader:
            optimizer.zero_grad()
            logits, _ = model(sources, previous, gaps)
            loss = criterion(logits, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_metrics = _evaluate(model, frame.loc[frame["split"] == "validation"], source_to_id, destination_to_id)
        val_nll = val_metrics.mean_nll_bits * math.log(2)
        scheduler.step(val_nll)

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Calibrate temperature on validation set.
    _calibrate_temperature(model, frame.loc[frame["split"] == "validation"], source_to_id, destination_to_id)

    model.eval()
    return {
        split: _evaluate(model, frame.loc[frame["split"] == split], source_to_id, destination_to_id)
        for split in ("train", "validation", "test")
        if not frame.loc[frame["split"] == split].empty
    }


def metrics_to_dict(metrics: dict[str, TemporalGNNMetrics]) -> dict[str, dict[str, float | int]]:
    return {split: vars(record) for split, record in metrics.items()}


def _vocabulary(values: pd.Series) -> dict[str, int]:
    counts = Counter(str(value) for value in values)
    return {value: index + 1 for index, (value, _) in enumerate(counts.most_common())}


def _examples(
    frame: pd.DataFrame,
    source_to_id: dict[str, int],
    destination_to_id: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    source_ids = []
    previous_ids = []
    gaps = []
    targets = []
    unknown = 0
    previous_by_source: dict[str, int] = {}
    previous_time_by_source: dict[str, float] = {}
    ordered = frame.sort_values(["timestamp"], kind="stable")
    for source, destination, timestamp in ordered[["source", "destination", "timestamp"]].itertuples(index=False):
        source_key = str(source)
        source_ids.append(source_to_id.get(source_key, 0))
        previous_ids.append(previous_by_source.get(source_key, 0))
        gap = float(timestamp) - previous_time_by_source.get(source_key, float(timestamp))
        gaps.append(math.log1p(max(0.0, gap)))
        target = destination_to_id.get(str(destination), 0)
        unknown += int(target == 0)
        targets.append(target)
        previous_by_source[source_key] = target
        previous_time_by_source[source_key] = float(timestamp)
    if not targets:
        raise ValueError("No temporal graph examples were created")
    return (
        np.asarray(source_ids, dtype=np.int64),
        np.asarray(previous_ids, dtype=np.int64),
        np.asarray(gaps, dtype=np.float32),
        np.asarray(targets, dtype=np.int64),
        unknown / len(targets),
    )


def _tensor_dataset(
    frame: pd.DataFrame,
    source_to_id: dict[str, int],
    destination_to_id: dict[str, int],
) -> TensorDataset:
    sources, previous, gaps, targets, _ = _examples(frame, source_to_id, destination_to_id)
    return TensorDataset(
        torch.from_numpy(sources),
        torch.from_numpy(previous),
        torch.from_numpy(gaps),
        torch.from_numpy(targets),
    )


def _evaluate(
    model: TemporalGraphCoverModel,
    frame: pd.DataFrame,
    source_to_id: dict[str, int],
    destination_to_id: dict[str, int],
) -> TemporalGNNMetrics:
    sources, previous, gaps, targets, unknown_fraction = _examples(frame, source_to_id, destination_to_id)
    with torch.no_grad():
        logits, _ = model(torch.from_numpy(sources), torch.from_numpy(previous), torch.from_numpy(gaps))
        probabilities = torch.softmax(logits, dim=1).numpy()
        target_tensor = torch.from_numpy(targets)
        log_probabilities = torch.log_softmax(logits, dim=1)
        nll_nats = -log_probabilities[torch.arange(len(target_tensor)), target_tensor].numpy()
        predictions = np.argmax(probabilities, axis=1)
    target_probabilities = probabilities[np.arange(len(targets)), targets]
    mean_nll_bits = float(np.mean(nll_nats) / math.log(2))
    return TemporalGNNMetrics(
        rows=len(targets),
        mean_nll_bits=mean_nll_bits,
        perplexity=float(2**mean_nll_bits),
        top1_accuracy=float(np.mean(predictions == targets)),
        unknown_target_fraction=unknown_fraction,
        expected_calibration_error=_ece(probabilities, targets),
        brier_score=float(np.mean((1.0 - target_probabilities) ** 2)),
    )


def _calibrate_temperature(
    model: TemporalGraphCoverModel,
    frame: pd.DataFrame,
    source_to_id: dict[str, int],
    destination_to_id: dict[str, int],
    *,
    lr: float = 0.05,
    max_iter: int = 20,
    batch_size: int = 1024,
) -> None:
    """Temperature scaling on validation NLL using batched gradient descent."""
    sources, previous, gaps, targets, _ = _examples(frame, source_to_id, destination_to_id)
    if len(targets) == 0:
        return
    dataset = TensorDataset(
        torch.from_numpy(sources),
        torch.from_numpy(previous),
        torch.from_numpy(gaps),
        torch.from_numpy(targets),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam([model.temperature], lr=lr)
    for _ in range(max_iter):
        epoch_loss = 0.0
        for sources_batch, previous_batch, gaps_batch, targets_batch in loader:
            optimizer.zero_grad()
            logits, _ = model(sources_batch, previous_batch, gaps_batch)
            loss = nn.CrossEntropyLoss()(logits / model.temperature, targets_batch)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.temperature.clamp_(min=0.1, max=10.0)
            epoch_loss += float(loss.item()) * len(targets_batch)
        if epoch_loss == 0:
            break


def _ece(probabilities: np.ndarray, targets: np.ndarray, *, bins: int = 10) -> float:
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = (predictions == targets).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(targets)
    error = 0.0
    for start, end in zip(edges[:-1], edges[1:]):
        mask = (confidence >= start) & (confidence < end if end < 1.0 else confidence <= end)
        if not np.any(mask):
            continue
        error += float(np.mean(mask)) * abs(float(np.mean(confidence[mask])) - float(np.mean(correct[mask])))
    return error / max(1.0, total / total)
