"""Compare the manuscript optimizers on noiseless linear regression.

All runs share 64 samples, four inputs, and the same initial weights.
Training uses full-batch MSE, gradient clipping at 1, and constant learning
rates. The all-method comparison includes MND-Muon at 0.03, 0.01, and 0.0075.
Plots are displayed only when requested; no results are written to disk.
"""

import argparse
import json
from pathlib import Path

import torch

from mnopt import CenteredAdam, MN, MND, MNDMuon


METHODS = {"centered_adam": CenteredAdam, "mn": MN, "mnd": MND, "mnd_muon": MNDMuon}


def run_regression(optimizer_name="mnd_muon", *, steps=80, device="cpu", seed=42,
                   optimizer_kwargs=None, compile=False):
    """Run the fixed regression problem and record every update's MSE.

    Each call recreates the same data and model initialization for a given seed.
    Loss index zero is the initial loss; index ``t`` is after update ``t``.
    """
    if optimizer_name not in METHODS:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    if type(steps) is not int or steps < 1:
        raise ValueError("steps must be a positive integer")
    device = torch.device(device)
    # Draw on CPU so methods share data and weights on every target device.
    # Restore the caller's CPU RNG state after constructing the problem.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        inputs = torch.randn(64, 4).to(device)
        model = torch.nn.Linear(4, 1).to(device)
    targets = inputs @ torch.tensor([[.3], [-.7], [.2], [.5]], device=device) + .1
    options = dict(optimizer_kwargs or {}, compile=compile)
    optimizer = METHODS[optimizer_name](model.parameters(), **options)

    def mse():
        return torch.nn.functional.mse_loss(model(inputs), targets)

    with torch.no_grad():
        losses = [mse().item()]
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = mse()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        with torch.no_grad():
            losses.append(mse().item())
    return {
        "optimizer": optimizer_name,
        "losses": losses,
        "weight": model.weight.detach().cpu().clone(),
        "bias": model.bias.detach().cpu().clone(),
    }


def plot_results(results):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    for label, result in results.items():
        losses = result["losses"]
        updates = range(len(losses))
        axes[0].plot(updates, losses, label=label, linewidth=1.5)
        axes[1].semilogy(updates, losses, linewidth=1.3)
    steps = len(next(iter(results.values()))["losses"]) - 1
    for ax in axes:
        ax.set_xlabel("Optimizer updates")
        ax.set_ylabel("Mean squared error")
        ax.grid(True, alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_xlim(0, min(80, steps))
    axes[0].set_ylim(bottom=0)
    axes[0].set_title(f"First {min(80, steps)} updates")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_xlim(0, steps)
    axes[1].set_title(f"All {steps} updates (log scale)")
    plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer", choices=["all", *METHODS], default="mnd_muon")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--plot", action="store_true", help="Show loss curves without saving files")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.optimizer == "all" and (args.config is not None or args.lr is not None):
        parser.error("--config and --lr require a single optimizer")

    runs = []
    if args.optimizer == "all":
        for name in METHODS:
            rates = (0.03, 0.01, 0.0075) if name == "mnd_muon" else (None,)
            for rate in rates:
                runs.append((name, {"lr": rate} if rate is not None else {}))
    else:
        config = json.loads(args.config.read_text()) if args.config else {
            "optimizer": args.optimizer, "kwargs": {},
        }
        options = dict(config["kwargs"])
        if args.lr is not None:
            options["lr"] = args.lr
        runs.append((config["optimizer"], options))

    results = {}
    for name, options in runs:
        result = run_regression(
            name, steps=args.steps, device=args.device, seed=args.seed,
            optimizer_kwargs=options, compile=args.compile,
        )
        losses = result["losses"]
        label = f"{name}[lr={options['lr']:g}]" if "lr" in options else name
        results[label] = result
        tail = losses[1:][-40:]
        print(
            f"{label}: loss {losses[0]:.6g} -> {losses[-1]:.6g} after {args.steps} steps; "
            f"best {min(losses):.6g}; mean last {len(tail)} updates {sum(tail) / len(tail):.6g}"
        )
    if args.plot:
        plot_results(results)


if __name__ == "__main__":
    main()
