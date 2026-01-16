import warnings
from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.types import Device
from torch.utils.data import DataLoader
from tqdm import tqdm

from .hsic import hsic, hsic_outer


class CKA:
    @staticmethod
    def _resolve_layers(
        model: nn.Module,
        layers: Sequence[str | int] | None,
        model_name: str,
    ) -> list[str] | None:
        if layers is None:
            return None

        all_layer_names = [name for name, _ in model.named_modules() if name]
        n_layers = len(all_layer_names)

        result: list[str] = []
        seen: set[str] = set()

        for layer in layers:
            if isinstance(layer, str):
                name = layer
            elif isinstance(layer, int):
                idx = layer if layer >= 0 else n_layers + layer

                if idx < 0 or idx >= n_layers:
                    raise IndexError(
                        f"Layer index {layer} is out of range for {model_name} "
                        f"with {n_layers} layers (valid range: {-n_layers} to {n_layers - 1})."
                    )
                name = all_layer_names[idx]
            else:
                raise TypeError(
                    f"Layer specification must be str or int, "
                    f"got {type(layer).__name__} in {model_name}_layers."
                )

            if name not in seen:
                result.append(name)
                seen.add(name)

        return result

    def __init__(
        self,
        model1: nn.Module,
        model2: nn.Module,
        model1_name: str | None = None,
        model2_name: str | None = None,
        model1_layers: Sequence[str | int] | None = None,
        model2_layers: Sequence[str | int] | None = None,
        device: Device = None,
    ) -> None:
        def unwrap(m: nn.Module) -> nn.Module:
            if isinstance(m, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                return m.module
            return m

        self.model1 = unwrap(model1)
        self.model2 = unwrap(model2)

        if device:
            self.device = torch.device(device)
        else:
            try:
                self.device = next(self.model1.parameters()).device
            except StopIteration:
                self.device = torch.device("cpu")

        model1_layers = self._resolve_layers(self.model1, model1_layers, "model1")
        model2_layers = self._resolve_layers(self.model2, model2_layers, "model2")

        if not model1_layers:
            model1_layers = [name for name, _ in self.model1.named_modules() if name]
            if len(model1_layers) > 150:
                warnings.warn(
                    f"Model1 has {len(model1_layers)} layers. "
                    "Consider specifying layers explicitly for faster computation."
                )
        if not model2_layers:
            model2_layers = [name for name, _ in self.model2.named_modules() if name]
            if len(model2_layers) > 150:
                warnings.warn(
                    f"Model2 has {len(model2_layers)} layers. "
                    "Consider specifying layers explicitly for faster computation."
                )

        self.model1_layers = model1_layers
        self.model2_layers = model2_layers

        self.model1_name = model1_name or self.model1.__class__.__name__
        self.model2_name = model2_name or self.model2.__class__.__name__

        self._is_same_model = self.model1 is self.model2

        self._features1: list[torch.Tensor | None] = [None] * len(self.model1_layers)
        self._layer_to_idx1: dict[str, int] = {
            name: i for i, name in enumerate(self.model1_layers)
        }

        if self._is_same_model:
            all_layers = list(dict.fromkeys(self.model1_layers + self.model2_layers))
            self._features_shared: list[torch.Tensor | None] = [None] * len(all_layers)
            self._layer_to_idx_shared: dict[str, int] = {
                name: i for i, name in enumerate(all_layers)
            }
        else:
            self._features2: list[torch.Tensor | None] = [None] * len(self.model2_layers)
            self._layer_to_idx2: dict[str, int] = {
                name: i for i, name in enumerate(self.model2_layers)
            }

        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []

        self._model1_training: bool | None = None
        self._model2_training: bool | None = None

    def __enter__(self) -> "CKA":
        self._register_hooks()
        self._save_training_state()
        self.model1.eval()
        self.model2.eval()
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> bool:
        self._remove_hooks()
        self._restore_training_state()
        self._clear_features()
        return False

    def _make_hook(
        self,
        features_list: list[torch.Tensor | None],
        layer_to_idx: dict[str, int],
        layer_name: str,
    ) -> Callable[[nn.Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
        idx = layer_to_idx[layer_name]

        def hook(
            module: nn.Module,
            input: Tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            if isinstance(output, tuple):
                output = output[0]
            elif hasattr(output, "last_hidden_state"):
                output = output.last_hidden_state

            if isinstance(output, torch.Tensor):
                features_list[idx] = output.detach()

        return hook

    def _register_hooks(self) -> None:
        if self._hook_handles:
            return

        if self._is_same_model:
            all_layers = set(self.model1_layers) | set(self.model2_layers)
            found = set()
            for name, module in self.model1.named_modules():
                if name in all_layers:
                    handle = module.register_forward_hook(
                        self._make_hook(
                            self._features_shared, self._layer_to_idx_shared, name
                        )
                    )
                    self._hook_handles.append(handle)
                    found.add(name)

            found1 = found & set(self.model1_layers)
            found2 = found & set(self.model2_layers)

            missing1 = set(self.model1_layers) - found1
            if missing1:
                warnings.warn(f"Layers not found in model: {sorted(missing1)}")

            if not found1:
                raise ValueError(
                    "No valid layers found in model1. "
                    "Use model.named_modules() to see available layers."
                )
            if not found2:
                raise ValueError(
                    "No valid layers found in model2. "
                    "Use model.named_modules() to see available layers."
                )
        else:
            found1 = set()
            for name, module in self.model1.named_modules():
                if name in self.model1_layers:
                    handle = module.register_forward_hook(
                        self._make_hook(self._features1, self._layer_to_idx1, name)
                    )
                    self._hook_handles.append(handle)
                    found1.add(name)

            missing1 = set(self.model1_layers) - found1
            if missing1:
                warnings.warn(f"Layers not found in model1: {sorted(missing1)}")

            found2 = set()
            for name, module in self.model2.named_modules():
                if name in self.model2_layers:
                    handle = module.register_forward_hook(
                        self._make_hook(self._features2, self._layer_to_idx2, name)
                    )
                    self._hook_handles.append(handle)
                    found2.add(name)

            missing2 = set(self.model2_layers) - found2
            if missing2:
                warnings.warn(f"Layers not found in model2: {sorted(missing2)}")

            if not found1:
                raise ValueError(
                    "No valid layers found in model1. "
                    "Use model.named_modules() to see available layers."
                )
            if not found2:
                raise ValueError(
                    "No valid layers found in model2. "
                    "Use model.named_modules() to see available layers."
                )

    def _remove_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def _save_training_state(self) -> None:
        self._model1_training = self.model1.training
        self._model2_training = self.model2.training

    def _restore_training_state(self) -> None:
        if self._model1_training is not None:
            self.model1.train(self._model1_training)
        if self._model2_training is not None:
            self.model2.train(self._model2_training)

    def _clear_features(self) -> None:
        if self._is_same_model:
            for i in range(len(self._features_shared)):
                self._features_shared[i] = None
        else:
            for i in range(len(self._features1)):
                self._features1[i] = None
            for i in range(len(self._features2)):
                self._features2[i] = None

    def compare(
        self,
        dataloader: DataLoader,
        dataloader2: DataLoader | None = None,
        progress: bool = True,
        callback: Callable[[int, int, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        if not self._hook_handles:
            raise RuntimeError(
                "Hooks not registered. Use 'with CKA(...) as cka:' context manager "
                "or call _register_hooks() first."
            )

        if dataloader2 is None:
            dataloader2 = dataloader

        n_layers1 = len(self.model1_layers)
        n_layers2 = len(self.model2_layers)

        hsic_xy = torch.zeros(n_layers1, n_layers2, device=self.device)
        hsic_xx = torch.zeros(n_layers1, device=self.device)
        hsic_yy = torch.zeros(n_layers2, device=self.device)

        total_batches = min(len(dataloader), len(dataloader2))
        iterator = zip(dataloader, dataloader2)

        if progress:
            iterator = tqdm(iterator, total=total_batches, desc="Computing CKA")

        with torch.no_grad():
            for batch_idx, (batch1, batch2) in enumerate(iterator):
                self._clear_features()

                x1 = self._extract_input(batch1)
                if x1.shape[0] <= 3:
                    raise ValueError(
                        f"HSIC requires batch size > 3, got {x1.shape[0]}. "
                        "Increase batch size to at least 4."
                    )
                x1 = x1.to(self.device)

                self.model1(x1)

                if not self._is_same_model:
                    x2 = self._extract_input(batch2)
                    x2 = x2.to(self.device)
                    self.model2(x2)

                self._accumulate_hsic(hsic_xy, hsic_xx, hsic_yy)

                if callback is not None:
                    current_cka = self._compute_cka_matrix(hsic_xy, hsic_xx, hsic_yy)
                    callback(batch_idx, total_batches, current_cka)

        return self._compute_cka_matrix(hsic_xy, hsic_xx, hsic_yy)

    def _extract_input(self, batch: Any) -> torch.Tensor:
        if isinstance(batch, torch.Tensor):
            return batch
        elif isinstance(batch, (list, tuple)):
            return batch[0]
        elif isinstance(batch, dict):
            for key in (
                "input",
                "inputs",
                "x",
                "image",
                "images",
                "input_ids",
                "pixel_values",
            ):
                if key in batch:
                    return batch[key]
            raise ValueError(
                f"Cannot find input in dict batch. Keys: {list(batch.keys())}"
            )
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

    def _accumulate_hsic(
        self,
        hsic_xy: torch.Tensor,
        hsic_xx: torch.Tensor,
        hsic_yy: torch.Tensor,
    ) -> None:
        if self._is_same_model:
            self._accumulate_hsic_same_model(hsic_xy, hsic_xx, hsic_yy)
        else:
            self._accumulate_hsic_different_models(hsic_xy, hsic_xx, hsic_yy)

    def _accumulate_hsic_same_model(
        self,
        hsic_xy: torch.Tensor,
        hsic_xx: torch.Tensor,
        hsic_yy: torch.Tensor,
    ) -> None:
        feats1: list[torch.Tensor] = []
        valid1: list[int] = []
        for i, layer in enumerate(self.model1_layers):
            idx = self._layer_to_idx_shared[layer]
            feat = self._features_shared[idx]
            if feat is not None:
                feats1.append(feat.flatten(1) if feat.dim() > 2 else feat)
                valid1.append(i)

        feats2: list[torch.Tensor] = []
        valid2: list[int] = []
        for j, layer in enumerate(self.model2_layers):
            idx = self._layer_to_idx_shared[layer]
            feat = self._features_shared[idx]
            if feat is not None:
                feats2.append(feat.flatten(1) if feat.dim() > 2 else feat)
                valid2.append(j)

        if not feats1 or not feats2:
            return

        grams1 = torch.stack([torch.mm(f, f.T) for f in feats1])
        grams2 = torch.stack([torch.mm(f, f.T) for f in feats2])

        hsic_matrix = hsic_outer(grams1, grams2)

        self_hsic1 = hsic(grams1, grams1)
        self_hsic2 = hsic(grams2, grams2)

        idx1 = torch.tensor(valid1, device=hsic_xy.device)
        idx2 = torch.tensor(valid2, device=hsic_xy.device)

        i_grid, j_grid = torch.meshgrid(idx1, idx2, indexing="ij")
        hsic_xy[i_grid, j_grid] += hsic_matrix
        hsic_xx[idx1] += self_hsic1
        hsic_yy[idx2] += self_hsic2

    def _accumulate_hsic_different_models(
        self,
        hsic_xy: torch.Tensor,
        hsic_xx: torch.Tensor,
        hsic_yy: torch.Tensor,
    ) -> None:
        feats1: list[torch.Tensor] = []
        valid1: list[int] = []
        for i, feat in enumerate(self._features1):
            if feat is not None:
                feats1.append(feat.flatten(1) if feat.dim() > 2 else feat)
                valid1.append(i)

        feats2: list[torch.Tensor] = []
        valid2: list[int] = []
        for j, feat in enumerate(self._features2):
            if feat is not None:
                feats2.append(feat.flatten(1) if feat.dim() > 2 else feat)
                valid2.append(j)

        if not feats1 or not feats2:
            return

        grams1 = torch.stack([torch.mm(f, f.T) for f in feats1])
        grams2 = torch.stack([torch.mm(f, f.T) for f in feats2])

        hsic_matrix = hsic_outer(grams1, grams2)

        self_hsic1 = hsic(grams1, grams1)
        self_hsic2 = hsic(grams2, grams2)

        idx1 = torch.tensor(valid1, device=hsic_xy.device)
        idx2 = torch.tensor(valid2, device=hsic_xy.device)

        i_grid, j_grid = torch.meshgrid(idx1, idx2, indexing="ij")
        hsic_xy[i_grid, j_grid] += hsic_matrix
        hsic_xx[idx1] += self_hsic1
        hsic_yy[idx2] += self_hsic2

    def _compute_cka_matrix(
        self,
        hsic_xy: torch.Tensor,
        hsic_xx: torch.Tensor,
        hsic_yy: torch.Tensor,
    ) -> torch.Tensor:
        denominator = torch.sqrt(
            torch.clamp(hsic_xx.unsqueeze(1) * hsic_yy.unsqueeze(0), min=0.0)
        )
        denominator = torch.where(denominator == 0, 1e-6, denominator)
        return hsic_xy / denominator

    def export(self, cka_matrix: torch.Tensor) -> Dict[str, Any]:
        return {
            "model1_name": self.model1_name,
            "model2_name": self.model2_name,
            "model1_layers": self.model1_layers,
            "model2_layers": self.model2_layers,
            "cka_matrix": cka_matrix,
        }

    def __call__(
        self,
        dataloader: DataLoader,
        dataloader2: DataLoader | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        with self:
            return self.compare(dataloader, dataloader2, **kwargs)
