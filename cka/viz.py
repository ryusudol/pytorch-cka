"""Visualization functions for CKA results.

This module provides publication-quality visualization functions that
always return (Figure, Axes) tuples for further customization.
"""

from typing import List, Literal, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_cka_heatmap(
    cka_matrix: torch.Tensor | np.ndarray,
    layers1: List[str] | None = None,
    layers2: List[str] | None = None,
    model1_name: str = "Model 1",
    model2_name: str = "Model 2",
    title: str | None = None,
    cmap: str = "magma",
    vmin: float | None = None,
    vmax: float | None = None,
    annot: bool = False,
    annot_fmt: str = ".2f",
    figsize: Tuple[float, float] | None = None,
    ax: Axes | None = None,
    colorbar: bool = True,
    mask_upper: bool = False,
    tick_fontsize: int = 8,
    label_fontsize: int = 12,
    title_fontsize: int = 14,
    annot_fontsize: int = 6,
    layer_name_depth: int | None = None,
    show: bool = False,
) -> Tuple[Figure, Axes]:
    """Plot CKA similarity matrix as a heatmap.

    Args:
        cka_matrix: CKA similarity matrix of shape (n_layers1, n_layers2).
        layers1: Layer names for y-axis (model1).
        layers2: Layer names for x-axis (model2).
        model1_name: Display name for model1.
        model2_name: Display name for model2.
        title: Plot title. If None, auto-generated.
        cmap: Matplotlib colormap name.
        vmin: Minimum value for colormap.
        vmax: Maximum value for colormap.
        annot: Show values in cells.
        annot_fmt: Format string for annotations.
        figsize: Figure size (width, height).
        ax: Existing axes to plot on.
        colorbar: Show colorbar.
        mask_upper: Mask upper triangle (for symmetric matrices).
        tick_fontsize: Font size for tick labels.
        label_fontsize: Font size for axis labels.
        title_fontsize: Font size for title.
        annot_fontsize: Font size for cell annotations.
        layer_name_depth: Number of name parts to show from end.
            E.g., 2 for "module.layer" from "encoder.module.layer".
        show: Whether to call plt.show().

    Returns:
        Tuple of (Figure, Axes).
    """
    # Convert to numpy
    if isinstance(cka_matrix, torch.Tensor):
        matrix = cka_matrix.detach().cpu().numpy()
    else:
        matrix = np.asarray(cka_matrix)

    n_layers1, n_layers2 = matrix.shape

    # Create figure if needed
    if ax is None:
        if figsize is None:
            figsize = (max(6, n_layers2 * 0.4 + 2), max(5, n_layers1 * 0.4 + 1))
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Handle masking for symmetric matrices
    if mask_upper and n_layers1 == n_layers2:
        mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)
        matrix = np.ma.masked_array(matrix, mask=mask)

    # Set colormap bounds
    if vmin is None:
        vmin = float(np.nanmin(matrix))
    if vmax is None:
        vmax = float(np.nanmax(matrix))

    # Plot heatmap
    im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    # Add annotations
    if annot:
        for i in range(n_layers1):
            for j in range(n_layers2):
                if mask_upper and n_layers1 == n_layers2 and i < j:
                    continue
                val = matrix[i, j]
                if not np.ma.is_masked(val) and not np.isnan(val):
                    text_color = "white" if val < (vmin + vmax) / 2 else "black"
                    ax.text(
                        j,
                        i,
                        format(val, annot_fmt),
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=annot_fontsize,
                    )

    # Set axis labels
    ax.set_xlabel(f"{model2_name} Layers", fontsize=label_fontsize)
    ax.set_ylabel(f"{model1_name} Layers", fontsize=label_fontsize)

    # Helper function to shorten layer names
    def shorten_name(name: str, depth: int | None) -> str:
        if depth is None:
            return name
        parts = name.split(".")
        return ".".join(parts[-depth:])

    # Set tick labels
    if layers1 is not None:
        shortened = [shorten_name(layer, layer_name_depth) for layer in layers1]
        ax.set_yticks(range(n_layers1))
        ax.set_yticklabels(shortened, fontsize=tick_fontsize)

    if layers2 is not None:
        shortened = [shorten_name(layer, layer_name_depth) for layer in layers2]
        ax.set_xticks(range(n_layers2))
        ax.set_xticklabels(shortened, fontsize=tick_fontsize, rotation=45, ha="right")

    # Title
    if title is None:
        title = f"CKA: {model1_name} vs {model2_name}"
    ax.set_title(title, fontsize=title_fontsize)

    # Colorbar
    if colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("CKA Similarity", fontsize=label_fontsize - 2)

    # Invert y-axis so layer 0 is at top
    ax.invert_yaxis()

    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_cka_trend(
    cka_values: torch.Tensor | List[torch.Tensor] | np.ndarray | List[np.ndarray],
    labels: List[str] | None = None,
    x_values: List[int | float] | None = None,
    xlabel: str = "Layer",
    ylabel: str = "CKA Similarity",
    title: str | None = None,
    figsize: Tuple[float, float] = (10, 6),
    ax: Axes | None = None,
    colors: List[str] | None = None,
    linestyles: List[str] | None = None,
    markers: List[str] | None = None,
    legend: bool = True,
    grid: bool = True,
    show: bool = False,
) -> Tuple[Figure, Axes]:
    """Plot CKA similarity trends (e.g., diagonal values or across epochs).

    Args:
        cka_values: Single array of shape (n_points,) or list of arrays.
        labels: Legend labels for each line.
        x_values: X-axis values. Defaults to 0, 1, 2, ...
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Plot title.
        figsize: Figure size.
        ax: Existing axes.
        colors: Line colors.
        linestyles: Line styles.
        markers: Marker styles.
        legend: Show legend.
        grid: Show grid.
        show: Whether to call plt.show().

    Returns:
        Tuple of (Figure, Axes).
    """
    # Normalize input to list of arrays
    if isinstance(cka_values, (torch.Tensor, np.ndarray)):
        arr = (
            cka_values.detach().cpu().numpy()
            if isinstance(cka_values, torch.Tensor)
            else cka_values
        )
        if arr.ndim == 1:
            cka_values = [arr]
        else:
            cka_values = [arr[i] for i in range(len(arr))]
    else:
        cka_values = [
            v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
            for v in cka_values
        ]

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    n_lines = len(cka_values)

    # Default styling
    if colors is None:
        cmap = plt.colormaps.get_cmap("tab10")
        colors = [cmap(i / max(n_lines - 1, 1)) for i in range(n_lines)]
    if linestyles is None:
        linestyles = ["-"] * n_lines
    if markers is None:
        markers = ["o"] * n_lines
    if labels is None:
        labels = [f"Line {i}" for i in range(n_lines)]

    # Plot each line
    for i, values in enumerate(cka_values):
        x = x_values if x_values is not None else list(range(len(values)))
        ax.plot(
            x,
            values,
            color=colors[i % len(colors)],
            linestyle=linestyles[i % len(linestyles)],
            marker=markers[i % len(markers)],
            label=labels[i] if i < len(labels) else None,
            markersize=6,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if legend and n_lines > 1:
        ax.legend()
    if grid:
        ax.grid(True, alpha=0.3)

    ax.set_ylim(0, 1.05)  # CKA is in [0, 1]

    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_cka_trend_with_range(
    mean_values: torch.Tensor | np.ndarray,
    min_values: torch.Tensor | np.ndarray,
    max_values: torch.Tensor | np.ndarray,
    label: str | None = None,
    x_values: List[int | float] | None = None,
    xlabel: str = "Layer",
    ylabel: str = "CKA Similarity",
    title: str | None = None,
    figsize: Tuple[float, float] = (10, 6),
    ax: Axes | None = None,
    color: str | None = None,
    style: Literal["fill", "errorbar"] = "fill",
    alpha: float = 0.3,
    marker: str = "o",
    linestyle: str = "-",
    legend: bool = True,
    grid: bool = True,
    show: bool = False,
) -> Tuple[Figure, Axes]:
    """Plot CKA trend line with min-max range visualization.

    Args:
        mean_values: Mean CKA values of shape (n_points,).
        min_values: Minimum CKA values of shape (n_points,).
        max_values: Maximum CKA values of shape (n_points,).
        label: Legend label for the line.
        x_values: X-axis values. Defaults to 0, 1, 2, ...
        xlabel: X-axis label.
        ylabel: Y-axis label.
        title: Plot title.
        figsize: Figure size.
        ax: Existing axes.
        color: Line and fill color. If None, uses default color.
        style: Visualization style - "fill" for shaded area, "errorbar" for error bars.
        alpha: Transparency for fill area (only used with style="fill").
        marker: Marker style.
        linestyle: Line style.
        legend: Show legend.
        grid: Show grid.
        show: Whether to call plt.show().

    Returns:
        Tuple of (Figure, Axes).
    """

    # Convert to numpy
    def to_numpy(arr):
        if isinstance(arr, torch.Tensor):
            return arr.detach().cpu().numpy()
        return np.asarray(arr)

    mean_arr = to_numpy(mean_values)
    min_arr = to_numpy(min_values)
    max_arr = to_numpy(max_values)

    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # X values
    x = x_values if x_values is not None else list(range(len(mean_arr)))

    # Default color
    if color is None:
        color = "#4e79a7"

    if style == "fill":
        # Plot mean line
        ax.plot(
            x,
            mean_arr,
            color=color,
            linestyle=linestyle,
            marker=marker,
            label=label,
            markersize=6,
        )
        # Fill between min and max
        ax.fill_between(
            x,
            min_arr,
            max_arr,
            color=color,
            alpha=alpha,
        )
    elif style == "errorbar":
        # Compute error bounds (asymmetric errors)
        yerr_lower = mean_arr - min_arr
        yerr_upper = max_arr - mean_arr
        ax.errorbar(
            x,
            mean_arr,
            yerr=[yerr_lower, yerr_upper],
            color=color,
            linestyle=linestyle,
            marker=marker,
            label=label,
            markersize=6,
            capsize=3,
            capthick=1,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if legend and label:
        ax.legend()
    if grid:
        ax.grid(True, alpha=0.3)

    ax.set_ylim(0, 1.05)  # CKA is in [0, 1]

    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_cka_comparison(
    matrices: List[torch.Tensor | np.ndarray],
    titles: List[str],
    layers: List[str] | None = None,
    ncols: int = 2,
    figsize: Tuple[float, float] | None = None,
    share_colorbar: bool = True,
    cmap: str = "magma",
    show: bool = False,
    **heatmap_kwargs,
) -> Tuple[Figure, np.ndarray]:
    """Plot multiple CKA matrices side by side for comparison.

    Args:
        matrices: List of CKA matrices.
        titles: Titles for each subplot.
        layers: Layer names (shared across all plots).
        ncols: Number of columns in subplot grid.
        figsize: Figure size. If None, auto-calculated.
        share_colorbar: Use shared colorbar with same scale.
        cmap: Colormap name.
        show: Whether to call plt.show().
        **heatmap_kwargs: Additional arguments for plot_cka_heatmap.

    Returns:
        Tuple of (Figure, array of Axes).
    """
    n_plots = len(matrices)
    nrows = (n_plots + ncols - 1) // ncols

    if figsize is None:
        figsize = (5 * ncols, 4 * nrows)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize, constrained_layout=share_colorbar
    )
    axes = np.atleast_2d(axes)

    # Find global min/max for shared colorbar
    if share_colorbar:
        all_values = []
        for m in matrices:
            if isinstance(m, torch.Tensor):
                all_values.append(m.detach().cpu().numpy().flatten())
            else:
                all_values.append(np.asarray(m).flatten())
        all_values = np.concatenate(all_values)
        vmin = float(np.nanmin(all_values))
        vmax = float(np.nanmax(all_values))
        heatmap_kwargs["vmin"] = vmin
        heatmap_kwargs["vmax"] = vmax

    for idx, (matrix, title) in enumerate(zip(matrices, titles)):
        row, col = idx // ncols, idx % ncols
        ax = axes[row, col]

        plot_cka_heatmap(
            matrix,
            layers1=layers,
            layers2=layers,
            title=title,
            ax=ax,
            cmap=cmap,
            colorbar=not share_colorbar,
            **heatmap_kwargs,
        )

    # Hide empty subplots
    for idx in range(n_plots, nrows * ncols):
        row, col = idx // ncols, idx % ncols
        axes[row, col].set_visible(False)

    # Add shared colorbar
    if share_colorbar:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02, label="CKA Similarity")
    else:
        fig.tight_layout()

    if show:
        plt.show()

    return fig, axes


def save_figure(
    fig: Figure,
    path: str,
    dpi: int = 150,
    bbox_inches: str = "tight",
    transparent: bool = False,
) -> None:
    """Save figure to file with sensible defaults.

    Args:
        fig: Matplotlib figure.
        path: Output path.
        dpi: Resolution in dots per inch.
        bbox_inches: Bounding box setting.
        transparent: Whether background should be transparent.
    """
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches, transparent=transparent)
    plt.close(fig)  # Close to free memory
