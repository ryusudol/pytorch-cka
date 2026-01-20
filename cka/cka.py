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
    ) -> dict[str, nn.Module]:
        name_to_module = {name: mod for name, mod in model.named_modules() if name}
        n_layers = len(name_to_module)
        all_layer_names = list(name_to_module.keys())

        if layers is None:
            if n_layers > 150:
                warnings.warn(
                    f"{model_name} has {n_layers} layers. "
                    "Consider specifying layers explicitly for faster computation."
                )
            return name_to_module

        result: dict[str, nn.Module] = {}

        for layer in layers:
            if isinstance(layer, str):
                if layer not in name_to_module:
                    raise ValueError(
                        f"Layer '{layer}' not found in {model_name}. "
                        f"Use model.named_modules() to see available layers."
                    )
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

            if name not in result:
                result[name] = name_to_module[name]

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

        self.model1_name = model1_name or self.model1.__class__.__name__
        self.model2_name = model2_name or self.model2.__class__.__name__

        self.model1_layer_to_module = self._resolve_layers(
            self.model1, model1_layers, "model1"
        )
        self.model2_layer_to_module = self._resolve_layers(
            self.model2, model2_layers, "model2"
        )

        self.device = torch.device(
            device if device else "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.model1 = self.model1.to(self.device)
        self.model2 = self.model2.to(self.device)

        self._is_same_model = self.model1 is self.model2

        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []

        self._model1_training = self.model1.training
        self._model2_training = self.model2.training

        if self._is_same_model:
            self._shared_layer_to_module: dict[str, nn.Module] = {
                **self.model1_layer_to_module,
                **self.model2_layer_to_module,
            }
            self._shared_features: dict[str, torch.Tensor] = {}
        else:
            self._model1_layer_to_feature: dict[str, torch.Tensor] = {}
            self._model2_layer_to_feature: dict[str, torch.Tensor] = {}

    def _clear_features(self) -> None:
        if self._is_same_model:
            self._shared_features.clear()
        else:
            self._model1_layer_to_feature.clear()
            self._model2_layer_to_feature.clear()

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
            grams = []
            layer_to_idx = {}
            for idx, (layer, feat) in enumerate(self._shared_features.items()):
                layer_to_idx[layer] = idx
                grams.append(torch.mm(feat, feat.T))
            grams = torch.stack(grams)

            grams1 = grams[
                [layer_to_idx[layer] for layer in self.model1_layer_to_module]
            ]
            grams2 = grams[
                [layer_to_idx[layer] for layer in self.model2_layer_to_module]
            ]

            if self.model1_layer_to_module.keys() == self.model2_layer_to_module.keys():
                hsic_matrix = hsic_outer(grams1, grams2)
                self_hsic = torch.diagonal(hsic_matrix)
                hsic_xy += hsic_matrix
                hsic_xx += self_hsic
                hsic_yy += self_hsic
                return

            cross_hsic, self_hsic1, self_hsic2 = hsic(grams1, grams2)
        else:
            grams1 = torch.stack(
                [
                    torch.mm(
                        self._model1_layer_to_feature[layer],
                        self._model1_layer_to_feature[layer].T,
                    )
                    for layer in self.model1_layer_to_module
                ]
            )
            grams2 = torch.stack(
                [
                    torch.mm(
                        self._model2_layer_to_feature[layer],
                        self._model2_layer_to_feature[layer].T,
                    )
                    for layer in self.model2_layer_to_module
                ]
            )

            cross_hsic, self_hsic1, self_hsic2 = hsic(grams1, grams2)

        hsic_xy += cross_hsic
        hsic_xx += self_hsic1
        hsic_yy += self_hsic2

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
        return torch.clamp(hsic_xy / denominator, min=0.0, max=1.0)

    def _make_hook(
        self,
        features_dict: dict[str, torch.Tensor],
        layer_name: str,
    ) -> Callable[[nn.Module, Tuple[torch.Tensor, ...], torch.Tensor], None]:
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
                feat = output.detach()
                features_dict[layer_name] = feat.flatten(1) if feat.dim() > 2 else feat

        return hook

    def compare(
        self,
        dataloader: DataLoader,
        dataloader2: DataLoader | None = None,
        progress: bool = True,
        callback: Callable[[int, int, torch.Tensor], None] | None = None,
    ) -> torch.Tensor:
        if not self._hook_handles:
            raise RuntimeError("Hooks not registered.")

        if dataloader2 is None:
            dataloader2 = dataloader

        n_layers1 = len(self.model1_layer_to_module)
        n_layers2 = len(self.model2_layer_to_module)

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

                if self._is_same_model:
                    self.model1(x1)
                else:
                    x2 = self._extract_input(batch2).to(self.device)
                    if self.device.type == "cuda":
                        stream1 = torch.cuda.Stream(device=self.device)
                        stream2 = torch.cuda.Stream(device=self.device)

                        with torch.cuda.stream(stream1):
                            self.model1(x1)
                        with torch.cuda.stream(stream2):
                            self.model2(x2)

                        stream1.synchronize()
                        stream2.synchronize()
                    else:
                        self.model1(x1)
                        self.model2(x2)

                self._accumulate_hsic(hsic_xy, hsic_xx, hsic_yy)

                if callback is not None:
                    current_cka = self._compute_cka_matrix(hsic_xy, hsic_xx, hsic_yy)
                    callback(batch_idx, total_batches, current_cka)

        return self._compute_cka_matrix(hsic_xy, hsic_xx, hsic_yy)

    def export(self, cka_matrix: torch.Tensor) -> Dict[str, Any]:
        return {
            "model1_name": self.model1_name,
            "model2_name": self.model2_name,
            "model1_layers": list(self.model1_layer_to_module.keys()),
            "model2_layers": list(self.model2_layer_to_module.keys()),
            "cka_matrix": cka_matrix,
        }

    def __enter__(self) -> "CKA":
        if self._hook_handles:
            return

        if self._is_same_model:
            for name, module in self._shared_layer_to_module.items():
                handle = module.register_forward_hook(
                    self._make_hook(self._shared_features, name)
                )
                self._hook_handles.append(handle)
        else:
            for name, module in self.model1_layer_to_module.items():
                handle = module.register_forward_hook(
                    self._make_hook(self._model1_layer_to_feature, name)
                )
                self._hook_handles.append(handle)

            for name, module in self.model2_layer_to_module.items():
                handle = module.register_forward_hook(
                    self._make_hook(self._model2_layer_to_feature, name)
                )
                self._hook_handles.append(handle)

        self.model1.eval()
        self.model2.eval()
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: Any | None,
    ) -> bool:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

        self.model1.train(self._model1_training)
        self.model2.train(self._model2_training)

        self._clear_features()

        return False

    def __call__(
        self,
        dataloader: DataLoader,
        dataloader2: DataLoader | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        with self:
            return self.compare(dataloader, dataloader2, **kwargs)
