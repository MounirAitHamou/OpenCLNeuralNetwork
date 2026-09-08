from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import clnn
from clnn import data, explain, nn, optim


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "CIFAR-10" / "data_batch_1.bin",
        help="CIFAR-10 binary batch to load",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=REPOSITORY_ROOT / "cifar10.clnn",
        help="model checkpoint path; optimizer state uses the .optimizer suffix",
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "opencl", "cpu"),
        default="auto",
        help="auto selects OpenCL when available and otherwise uses the CPU",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="ignore existing model and optimizer checkpoints",
    )
    return parser.parse_args()


def select_device(name: str) -> clnn.Device:
    if name == "cpu":
        return clnn.Device.cpu()
    if name == "opencl":
        if not clnn.opencl_available():
            raise RuntimeError("OpenCL was requested, but no OpenCL GPU is available")
        return clnn.Device.opencl()
    return clnn.Device.opencl() if clnn.opencl_available() else clnn.Device.cpu()


def create_model(device: clnn.Device, seed: int) -> nn.Sequential:
    model = nn.Sequential()
    model.add(nn.Conv2d(3, 32, (3, 3), padding="same", seed=seed, device=device))
    model.add(nn.ReLU())
    model.add(
        nn.Conv2d(
            32,
            64,
            (3, 3),
            stride=(2, 2),
            padding="same",
            seed=seed + 1,
            device=device,
        )
    )
    model.add(nn.ReLU())
    model.add(
        nn.Conv2d(
            64,
            128,
            (3, 3),
            stride=(2, 2),
            padding="same",
            seed=seed + 2,
            device=device,
        )
    )
    model.add(nn.ReLU())
    model.add(nn.Flatten())
    model.add(nn.Linear(128 * 8 * 8, 256, seed=seed + 3, device=device))
    model.add(nn.ReLU())
    model.add(nn.Linear(256, 10, seed=seed + 4, device=device))
    return model


def train_epoch(
    model: nn.Sequential,
    optimizer: optim.Optimizer,
    loader: data.DataLoader,
) -> float:
    accumulated_loss = 0.0
    sample_count = 0
    for batch in loader:
        optimizer.zero_grad()
        logits = model(batch.inputs)
        loss = clnn.cross_entropy(logits, batch.class_labels)
        loss_value = loss.item()
        loss.backward()
        optimizer.step()

        accumulated_loss += loss_value * batch.size
        sample_count += batch.size

    return accumulated_loss / sample_count if sample_count else float("nan")


def evaluate(model: nn.Sequential, loader: data.DataLoader) -> float:
    correct = 0
    sample_count = 0
    with clnn.no_grad():
        for batch in loader:
            logits = model(batch.inputs).numpy()
            predictions = np.argmax(logits, axis=1)
            labels = np.asarray(batch.class_labels, dtype=np.int64)
            correct += int(np.count_nonzero(predictions == labels))
            sample_count += batch.size
    return correct / sample_count if sample_count else 0.0


def main() -> None:
    arguments = parse_arguments()
    if arguments.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if arguments.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if arguments.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if arguments.weight_decay < 0.0:
        raise ValueError("--weight-decay must be non-negative")
    if not arguments.data.is_file():
        raise FileNotFoundError(f"CIFAR-10 batch not found: {arguments.data}")

    device = select_device(arguments.device)
    optimizer_path = Path(f"{arguments.checkpoint}.optimizer")
    print(f"Training on {device.name}")
    print(f"Loading CIFAR-10 from {arguments.data}")

    dataset = data.load_cifar10_batch(arguments.data)
    partitions = data.split(dataset, 0.9, 0.1, seed=arguments.seed)
    training_loader = data.DataLoader(
        partitions.training,
        arguments.batch_size,
        device=device,
        shuffle=True,
        seed=arguments.seed,
    )
    validation_loader = data.DataLoader(
        partitions.validation,
        arguments.batch_size,
        device=device,
    )

    resumed = arguments.checkpoint.is_file() and not arguments.fresh
    if resumed:
        model = nn.load_checkpoint(
            arguments.checkpoint,
            device=device,
            initialization_seed=arguments.seed,
        )
        print(f"Resumed model from {arguments.checkpoint}")
    else:
        model = create_model(device, arguments.seed)

    optimizer = optim.AdamW(
        model,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
    )
    if resumed and optimizer_path.is_file():
        optim.load_state_dict(optimizer, optimizer_path)
        print(f"Resumed optimizer state from {optimizer_path}")

    arguments.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, arguments.epochs + 1):
        mean_loss = train_epoch(model, optimizer, training_loader)
        validation_accuracy = evaluate(model, validation_loader)
        print(
            f"epoch {epoch}/{arguments.epochs}  loss {mean_loss:.6f}  "
            f"validation accuracy {validation_accuracy:.2%}"
        )

        nn.save_checkpoint(model, arguments.checkpoint)
        optim.save_state_dict(optimizer, optimizer_path)

    print(f"Saved model checkpoint to {arguments.checkpoint}")
    print(f"Saved optimizer state to {optimizer_path}")

    example_batch = next(iter(validation_loader))
    predicted_class = int(np.argmax(model(example_batch.inputs).numpy()[0]))
    attribution = explain.integrated_gradients(
        model,
        example_batch.inputs,
        target=predicted_class,
        batch_index=0,
        baseline=clnn.Tensor.zeros(example_batch.inputs.shape, device=device),
        steps=32,
    )
    channel_attribution = attribution.numpy()[0]
    heatmap = np.abs(channel_attribution).sum(axis=0)
    print(
        f"Explained class {predicted_class}: attribution {attribution.shape}, "
        f"NumPy heatmap {heatmap.shape}"
    )


if __name__ == "__main__":
    main()
