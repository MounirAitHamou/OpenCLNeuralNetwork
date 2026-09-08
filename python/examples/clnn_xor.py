from __future__ import annotations

import argparse

import numpy as np

import clnn
from clnn import explain, nn, optim


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an XOR network and explain its output logit."
    )
    parser.add_argument("--epochs", type=int, default=1_000)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "opencl", "cpu"),
        default="auto",
        help="auto selects OpenCL when available and otherwise uses the CPU",
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
    model.add(nn.Linear(2, 8, seed=seed, device=device))
    model.add(nn.Tanh())
    model.add(nn.Linear(8, 1, seed=seed + 1, device=device))
    return model


def main() -> None:
    arguments = parse_arguments()
    if arguments.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if arguments.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")

    device = select_device(arguments.device)
    inputs = clnn.Tensor(
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32),
        device=device,
    )
    targets = clnn.Tensor(
        np.array([[0], [1], [1], [0]], dtype=np.float32),
        device=device,
    )
    model = create_model(device, arguments.seed)
    optimizer = optim.Adam(model, learning_rate=arguments.learning_rate)

    for epoch in range(arguments.epochs):
        optimizer.zero_grad()
        loss = clnn.binary_cross_entropy_with_logits(model(inputs), targets)
        loss.backward()
        optimizer.step()
        if epoch % 200 == 0 or epoch + 1 == arguments.epochs:
            print(f"epoch {epoch:4d}  loss {loss.item():.6f}")

    baseline = clnn.Tensor.zeros(inputs.shape, device=device)
    saliency = explain.saliency(model, inputs, target=0)
    input_gradient = explain.input_x_gradient(model, inputs, target=0)
    integrated = explain.integrated_gradients(
        model,
        inputs,
        target=0,
        baseline=baseline,
        steps=64,
    )
    smoothed = explain.smoothgrad(
        model,
        inputs,
        target=0,
        samples=32,
        noise_stddev=0.1,
        seed=arguments.seed,
    )

    with clnn.no_grad():
        logits = model(inputs)
        probabilities = clnn.sigmoid(logits).numpy().reshape(-1)

    input_values = inputs.numpy()
    saliency_values = saliency.numpy()
    input_gradient_values = input_gradient.numpy()
    integrated_values = integrated.numpy()
    smoothed_values = smoothed.numpy()

    print(f"\nXOR predictions and attributions on {device.name}:")
    for row, (left, right) in enumerate(input_values):
        print(
            f"{int(left)} xor {int(right)} = {probabilities[row]:.4f}  "
            f"saliency={saliency_values[row]}  "
            f"input_x_gradient={input_gradient_values[row]}  "
            f"integrated_gradients={integrated_values[row]}  "
            f"smoothgrad={smoothed_values[row]}"
        )


if __name__ == "__main__":
    main()
