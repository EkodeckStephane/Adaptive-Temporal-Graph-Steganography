from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ModelKind = Literal["gru", "transformer"]


@dataclass(frozen=True)
class NeuralSequenceMetrics:
    rows: int
    mean_nll_bits: float
    perplexity: float
    top1_accuracy: float
    unknown_target_fraction: float


class NeuralSequenceModel(nn.Module):
    def __init__(
        self,
        *,
        vocabulary_size: int,
        embedding_dim: int,
        hidden_dim: int,
        context_length: int,
        kind: ModelKind,
    ) -> None:
        super().__init__()
        self.kind = kind
        self.context_length = context_length
        self.embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        if kind == "gru":
            self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
            self.projection = nn.Linear(hidden_dim, vocabulary_size)
        elif kind == "transformer":
            layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=2,
                dim_feedforward=hidden_dim,
                batch_first=True,
                dropout=0.0,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=1)
            self.projection = nn.Linear(embedding_dim, vocabulary_size)
        else:
            raise ValueError(f"Unsupported model kind: {kind}")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(inputs)
        if self.kind == "gru":
            _, hidden = self.encoder(embedded)
            representation = hidden[-1]
        else:
            encoded = self.encoder(embedded)
            representation = encoded[:, -1, :]
        return self.projection(representation)


def train_and_evaluate(
    frame: pd.DataFrame,
    *,
    kind: ModelKind,
    context_length: int,
    embedding_dim: int,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict[str, NeuralSequenceMetrics]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    train = frame.loc[frame["split"] == "train"]
    destination_to_id = _vocabulary(train["destination"])
    model = NeuralSequenceModel(
        vocabulary_size=len(destination_to_id) + 1,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        context_length=context_length,
        kind=kind,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_dataset = _tensor_dataset(train, destination_to_id, context_length)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for inputs, targets in loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()

    model.eval()
    return {
        split: _evaluate(model, frame.loc[frame["split"] == split], destination_to_id, context_length)
        for split in ("train", "validation", "test")
        if not frame.loc[frame["split"] == split].empty
    }


def metrics_to_dict(metrics: dict[str, NeuralSequenceMetrics]) -> dict[str, dict[str, float | int]]:
    return {split: vars(record) for split, record in metrics.items()}


def _vocabulary(values: pd.Series) -> dict[str, int]:
    counts = Counter(str(value) for value in values)
    return {value: index + 1 for index, (value, _) in enumerate(counts.most_common())}


def _examples(
    frame: pd.DataFrame,
    destination_to_id: dict[str, int],
    context_length: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    contexts = []
    targets = []
    unknown = 0
    history_by_source: dict[str, list[int]] = {}
    ordered = frame.sort_values(["timestamp"], kind="stable")
    for source, destination in ordered[["source", "destination"]].itertuples(index=False):
        source = str(source)
        target = destination_to_id.get(str(destination), 0)
        if target == 0:
            unknown += 1
        history = history_by_source.setdefault(source, [])
        padded = ([0] * context_length + history)[-context_length:]
        contexts.append(padded)
        targets.append(target)
        history.append(target)
    if not contexts:
        raise ValueError("No sequence examples were created")
    return (
        np.asarray(contexts, dtype=np.int64),
        np.asarray(targets, dtype=np.int64),
        unknown / len(targets),
    )


def _tensor_dataset(
    frame: pd.DataFrame,
    destination_to_id: dict[str, int],
    context_length: int,
) -> TensorDataset:
    contexts, targets, _ = _examples(frame, destination_to_id, context_length)
    return TensorDataset(torch.from_numpy(contexts), torch.from_numpy(targets))


def _evaluate(
    model: NeuralSequenceModel,
    frame: pd.DataFrame,
    destination_to_id: dict[str, int],
    context_length: int,
) -> NeuralSequenceMetrics:
    contexts, targets, unknown_fraction = _examples(frame, destination_to_id, context_length)
    with torch.no_grad():
        logits = model(torch.from_numpy(contexts))
        log_probabilities = torch.log_softmax(logits, dim=1)
        target_tensor = torch.from_numpy(targets)
        nll_nats = -log_probabilities[torch.arange(len(target_tensor)), target_tensor].numpy()
        predictions = torch.argmax(logits, dim=1).numpy()
    mean_nll_bits = float(np.mean(nll_nats) / math.log(2))
    return NeuralSequenceMetrics(
        rows=len(targets),
        mean_nll_bits=mean_nll_bits,
        perplexity=float(2**mean_nll_bits),
        top1_accuracy=float(np.mean(predictions == targets)),
        unknown_target_fraction=unknown_fraction,
    )
