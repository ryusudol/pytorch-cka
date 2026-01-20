<div align="center">

# Centered Kernel Alignment (CKA)

[![PyPI](https://img.shields.io/pypi/v/pytorch-cka.svg)](https://pypi.org/project/pytorch-cka/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://pypi.org/project/pytorch-cka/)
[![CI](https://github.com/ryusudol/Centered-Kernel-Alignment/workflows/CI/badge.svg)](https://github.com/ryusudol/Centered-Kernel-Alignment/actions)

**Fast and Memory-efficient CKA Library for PyTorch**

</div>


<p align="center">
    <img alt="A bar chart with benchmark results." src="docs/assets/bar-chart.png" width="100%" />
</p>

<p align="center">
  <i>Computing CKA on two different ResNet-18 models.</i>
</p>


- ⚡️ **3000%** Faster than the [most popular CKA library](https://github.com/AntixK/PyTorch-Model-Compare)
- 📦 Memory-efficient minibatch CKA computation
- 🎨 Customizable visualizations: heatmaps and line charts
- 🧠 Supports HuggingFace models, DataParallel, and DDP
- 🐳 Installable via `pip` or `docker`
- 🛠️ Modern `pyproject.toml` packaging
- 🤝 Python 3.10–3.14 compatibility


## 📦 Installation

Requires Python >= 3.10.

```bash
# Using pip
pip install pytorch-cka

# Using uv
uv add pytorch-cka

# Using docker
docker pull ghcr.io/ryusudol/pytorch-cka

# From source
git clone https://github.com/ryusudol/Centered-Kernel-Alignment
cd pytorch-cka
uv sync  # or: pip install -e .
```

## 👟 Quick Start

### Basic Usage

```python
from torch.utils.data import DataLoader
from cka import CKA

pretrained_model = ...  # e.g. pretrained ResNet-18
fine_tuned_model = ...  # e.g. fine-tuned ResNet-18

layers = ["layer1", "layer2", "layer3", "fc"]

dataloader = DataLoader(..., batch_size=128)

cka = CKA(
    model1=pretrained_model,
    model2=fine_tuned_model,
    model1_name="ResNet-18 (pretrained)",
    model2_name="ResNet-18 (fine-tuned)",
    model1_layers=layers,
    model2_layers=layers,
    device="cuda"
)

# Most convenient usage (auto context manager)
cka_matrix = cka(dataloader)
cka_result = cka.export(cka_matrix)

# Or explicit control
with cka:
    cka_matrix = cka.compare(dataloader)
    cka_result = cka.export(cka_matrix)
```

### Visualization

**Heatmap**

```python
from cka import plot_cka_heatmap

fig, ax = plot_cka_heatmap(
    cka_matrix,
    layers1=layers,
    layers2=layers,
    model1_name="ResNet-18 (pretrained)",
    model2_name="ResNet-18 (random init)",
    annot=False,          # Show values in cells
    cmap="inferno",       # Colormap
    mask_upper=False,     # Mask upper triangle (symmetric matrices)
)
```

<table>
    <tr>
      <td><img src="examples/plots/heatmap_self.png" alt="Self-comparison heatmap" width="100%"/></td>
      <td><img src="examples/plots/heatmap_cross.png" alt="Cross-model comparison heatmap" width="100%"/></td>
      <!-- <td><img src="plots/heatmap_masked.png" alt="Masked upper triangle heatmap" width="100%"/></td> -->
    </tr>
    <tr>
      <td align="center">Self-comparison</td>
      <td align="center">Cross-model</td>
      <!-- <td align="center">Masked Upper</td> -->
    </tr>
</table>

**Trend Plot**

```python
from cka import plot_cka_trend

# Plot diagonal (self-similarity across layers)
diagonal = torch.diag(matrix)
fig, ax = plot_cka_trend(
    diagonal,
    labels=["Self-similarity"],
    xlabel="Layer",
    ylabel="CKA Score",
)
```

<table>
    <tr>
      <td><img src="examples/plots/line_cross_model_convergence.png" alt="Cross model CKA scores trends" width="100%"/></td>
      <td><img src="examples/plots/trend_multi.png" alt="Multiple trends comparison" width="100%"/></td>
    </tr>
    <tr>
      <td align="center">Cross Model CKA Scores Trends</td>
      <td align="center">Multiple Trends</td>
    </tr>
</table>

<!-- **Side-by-Side Comparison**

```python
from cka import plot_cka_comparison

fig, axes = plot_cka_comparison(
    matrices=[matrix1, matrix2, matrix3],
    titles=["Epoch 1", "Epoch 10", "Epoch 100"],
    layers=layers,
    share_colorbar=True,
)
```

<table>
    <tr>
      <td><img src="examples/plots/comparison_grid.png" alt="CKA comparison grid" width="100%"/></td>
    </tr>
    <tr>
      <td align="center">CKA comparison grid</td>
    </tr>
</table> -->

## 📚 References

1. Kornblith, Simon, et al. ["Similarity of Neural Network Representations Revisited."](https://arxiv.org/abs/1905.00414) _ICML 2019._

2. Nguyen, Thao, Maithra Raghu, and Simon Kornblith. ["Do Wide and Deep Networks Learn the Same Things?"](https://arxiv.org/abs/2010.15327) _arXiv 2020._ (Minibatch CKA)

3. Wang, Tinghua, Xiaolu Dai, and Yuze Liu. ["Learning with Hilbert-Schmidt Independence Criterion: A Review."](https://www.sciencedirect.com/science/article/pii/S0950705121008297) _Knowledge-Based Systems 2021._

### Related Projects

- [AntixK/PyTorch-Model-Compare](https://github.com/AntixK/PyTorch-Model-Compare)
- [RistoAle97/centered-kernel-alignment](https://github.com/RistoAle97/centered-kernel-alignment)

## 📝 License

[MIT License](LICENSE)
