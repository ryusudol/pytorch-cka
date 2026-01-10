#!/usr/bin/env python3
"""Generate all example CKA plots in the plots/ directory.

This script demonstrates all visualization functions in pytorch_cka:
- plot_cka_heatmap(): CKA similarity matrix heatmaps
- plot_cka_trend(): Line plots of CKA values
- plot_cka_comparison(): Side-by-side matrix comparison
- save_figure(): Utility for saving figures
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pytorch_cka import CKA
from pytorch_cka.viz import (
    plot_cka_comparison,
    plot_cka_heatmap,
    plot_cka_trend,
    save_figure,
)

PLOTS_DIR = Path(__file__).parent.parent / "plots"


# =============================================================================
# Model Definitions
# =============================================================================


class SimpleCNN(nn.Module):
    """Simple CNN for demonstration."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)


class WiderCNN(nn.Module):
    """Wider CNN for comparison."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.flatten(1)
        return self.fc(x)


def create_dataloader(
    batch_size: int = 16, num_samples: int = 64, image_size: int = 32
) -> DataLoader:
    """Create a dummy dataloader for demonstration."""
    data = torch.randn(num_samples, 3, image_size, image_size)
    labels = torch.randint(0, 10, (num_samples,))
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# =============================================================================
# Plot Generation Functions
# =============================================================================


def generate_heatmap_self(matrix: torch.Tensor, layers: list[str]) -> None:
    """Generate self-comparison heatmap."""
    fig, ax = plot_cka_heatmap(
        matrix,
        layers1=layers,
        layers2=layers,
        model1_name="SimpleCNN",
        model2_name="SimpleCNN",
        title="Self-Comparison Heatmap",
    )
    save_figure(fig, str(PLOTS_DIR / "heatmap_self.png"))
    print(f"  Saved: heatmap_self.png")


def generate_heatmap_cross(
    matrix: torch.Tensor, layers1: list[str], layers2: list[str]
) -> None:
    """Generate cross-model heatmap with annotations."""
    fig, ax = plot_cka_heatmap(
        matrix,
        layers1=layers1,
        layers2=layers2,
        model1_name="SimpleCNN",
        model2_name="WiderCNN",
        annot=True,
        title="Cross-Model Comparison (with annotations)",
    )
    save_figure(fig, str(PLOTS_DIR / "heatmap_cross.png"))
    print(f"  Saved: heatmap_cross.png")


def generate_heatmap_masked(matrix: torch.Tensor, layers: list[str]) -> None:
    """Generate heatmap with upper triangle masked (for symmetric matrices)."""
    fig, ax = plot_cka_heatmap(
        matrix,
        layers1=layers,
        layers2=layers,
        model1_name="SimpleCNN",
        model2_name="SimpleCNN",
        mask_upper=True,
        annot=True,
        title="Masked Upper Triangle (symmetric)",
    )
    save_figure(fig, str(PLOTS_DIR / "heatmap_masked.png"))
    print(f"  Saved: heatmap_masked.png")


def generate_trend_single(matrix: torch.Tensor) -> None:
    """Generate single-line trend plot (diagonal values)."""
    diagonal = torch.diag(matrix)
    fig, ax = plot_cka_trend(
        diagonal,
        labels=["Self-similarity"],
        xlabel="Layer Index",
        ylabel="CKA Score",
        title="Layer Self-Similarity (Diagonal Values)",
    )
    save_figure(fig, str(PLOTS_DIR / "trend_single.png"))
    print(f"  Saved: trend_single.png")


def generate_trend_multi(
    self_matrix: torch.Tensor, cross_matrix: torch.Tensor
) -> None:
    """Generate multi-line trend plot comparing different metrics."""
    # Extract different metrics from matrices
    self_diag = torch.diag(self_matrix)
    cross_diag = torch.diag(cross_matrix)
    # First row shows how layer 0 relates to other layers
    first_row = self_matrix[0, :]

    fig, ax = plot_cka_trend(
        [self_diag, cross_diag, first_row],
        labels=["Self-similarity", "Cross-model (diagonal)", "Layer 0 vs others"],
        xlabel="Layer Index",
        ylabel="CKA Score",
        title="Multiple CKA Metrics Comparison",
        colors=["#2ecc71", "#e74c3c", "#3498db"],
        markers=["o", "s", "^"],
        linestyles=["-", "--", ":"],
    )
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    save_figure(fig, str(PLOTS_DIR / "trend_multi.png"))
    print(f"  Saved: trend_multi.png")


def generate_comparison_grid(matrices: list[torch.Tensor], layers: list[str]) -> None:
    """Generate 2x2 grid comparison of multiple matrices."""
    titles = [
        "Self-comparison",
        "Cross-model",
        "Simulated Epoch 1",
        "Simulated Epoch 2",
    ]

    # Create simulated epoch matrices by adding noise
    epoch1 = matrices[0] + torch.randn_like(matrices[0]) * 0.1
    epoch1 = epoch1.clamp(0, 1)
    epoch2 = matrices[0] + torch.randn_like(matrices[0]) * 0.05
    epoch2 = epoch2.clamp(0, 1)

    all_matrices = [matrices[0], matrices[1], epoch1, epoch2]

    fig, axes = plot_cka_comparison(
        all_matrices,
        titles=titles,
        layers=layers,
        ncols=2,
        share_colorbar=True,
        annot=False,
    )
    save_figure(fig, str(PLOTS_DIR / "comparison_grid.png"))
    print(f"  Saved: comparison_grid.png")


# =============================================================================
# Main
# =============================================================================


def main():
    """Generate all example plots."""
    print("=" * 60)
    print("Generating CKA Example Plots")
    print("=" * 60)

    # Create output directory
    PLOTS_DIR.mkdir(exist_ok=True)
    print(f"\nOutput directory: {PLOTS_DIR}")

    # Create models and dataloader
    model1 = SimpleCNN()
    model2 = WiderCNN()
    dataloader = create_dataloader()

    layers = ["conv1", "conv2", "conv3", "pool"]

    # Compute CKA matrices
    print("\nComputing CKA matrices...")

    print("  Self-comparison (SimpleCNN vs SimpleCNN)")
    with CKA(model1, model1, model1_layers=layers, model2_layers=layers) as cka:
        self_matrix = cka.compare(dataloader, progress=False)

    print("  Cross-model (SimpleCNN vs WiderCNN)")
    with CKA(model1, model2, model1_layers=layers, model2_layers=layers) as cka:
        cross_matrix = cka.compare(dataloader, progress=False)

    # Generate all plots
    print("\nGenerating plots...")

    # Heatmaps
    generate_heatmap_self(self_matrix, layers)
    generate_heatmap_cross(cross_matrix, layers, layers)
    generate_heatmap_masked(self_matrix, layers)

    # Trend plots
    generate_trend_single(self_matrix)
    generate_trend_multi(self_matrix, cross_matrix)

    # Comparison grid
    generate_comparison_grid([self_matrix, cross_matrix], layers)

    print("\n" + "=" * 60)
    print("All plots generated successfully!")
    print(f"Check the '{PLOTS_DIR}' directory for output files.")
    print("=" * 60)


if __name__ == "__main__":
    main()
