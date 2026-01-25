from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from cka import CKA


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 20)
        self.layer3 = nn.Linear(20, 5)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)


class LargeModel(nn.Module):
    def __init__(self, n_layers=160):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(10, 10) for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TupleOutputLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 20)

    def forward(self, x):
        out = self.linear(x)
        return (out, out)


class TupleOutputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = TupleOutputLayer()

    def forward(self, x):
        out, _ = self.layer1(x)
        return out


class TransformerLikeOutput:
    def __init__(self, tensor):
        self.last_hidden_state = tensor


class TransformerOutputLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 20)

    def forward(self, x):
        out = self.linear(x)
        return TransformerLikeOutput(out)


class TransformerOutputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = TransformerOutputLayer()

    def forward(self, x):
        out = self.layer1(x)
        return out.last_hidden_state


class Conv3DModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.fc = nn.Linear(16 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


@pytest.fixture
def dataloader():
    x = torch.randn(32, 10)
    dataset = TensorDataset(x)
    return DataLoader(dataset, batch_size=8)


@pytest.fixture
def small_batch_dataloader():
    x = torch.randn(12, 10)
    dataset = TensorDataset(x)
    return DataLoader(dataset, batch_size=3)


@pytest.fixture
def image_dataloader():
    x = torch.randn(32, 3, 8, 8)
    dataset = TensorDataset(x)
    return DataLoader(dataset, batch_size=8)


@pytest.fixture
def model1():
    return SimpleModel()


@pytest.fixture
def model2():
    return SimpleModel()


class TestResolveLayers:
    def test_layers_none_returns_all(self, model1, model2, dataloader):
        cka = CKA(model1, model2)

        assert len(cka.model1_layer_to_module) == 3
        assert "layer1" in cka.model1_layer_to_module
        assert "layer2" in cka.model1_layer_to_module
        assert "layer3" in cka.model1_layer_to_module

    def test_layers_none_large_model_warning(self, model2):
        large_model = LargeModel(151)

        with pytest.warns(UserWarning, match="has 152 layers"):
            CKA(large_model, model2)

    def test_duplicate_layers_skipped(self, model1, model2, dataloader):
        cka = CKA(
            model1,
            model2,
            model1_layers=["layer1", "layer1", "layer2"],
        )

        assert len(cka.model1_layer_to_module) == 2
        assert list(cka.model1_layer_to_module.keys()) == ["layer1", "layer2"]


class TestInit:
    def test_dataparallel_unwrapping(self, dataloader):
        model = SimpleModel()
        dp_model = nn.DataParallel(model)

        cka = CKA(dp_model, model)

        assert cka.model1 is model

    def test_distributed_dataparallel_unwrapping(self, dataloader):
        model = SimpleModel()
        ddp_mock = MagicMock(spec=nn.parallel.DistributedDataParallel)
        ddp_mock.module = model

        cka = CKA(ddp_mock, model)

        assert cka.model1 is model

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_auto_device_cpu_fallback(self, mock_mps, mock_cuda, model1, model2):
        cka = CKA(model1, model2)

        assert cka.device == torch.device("cpu")

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_auto_device_cuda(self, model1, model2):
        cka = CKA(model1, model2)

        assert cka.device == torch.device("cuda")

    def test_custom_model_names(self, model1, model2):
        cka = CKA(model1, model2, model1_name="ResNet", model2_name="VGG")

        assert cka.model1_name == "ResNet"
        assert cka.model2_name == "VGG"

    def test_default_model_names(self, model1, model2):
        cka = CKA(model1, model2)

        assert cka.model1_name == "SimpleModel"
        assert cka.model2_name == "SimpleModel"

    def test_same_model_detection(self, model1):
        cka = CKA(model1, model1)

        assert cka._is_same_model is True

    def test_different_model_detection(self, model1, model2):
        cka = CKA(model1, model2)

        assert cka._is_same_model is False

    def test_training_state_captured(self, model1, model2):
        model1.train()
        model2.eval()

        cka = CKA(model1, model2)

        assert cka._model1_training is True
        assert cka._model2_training is False

    def test_custom_device(self, model1, model2):
        cka = CKA(model1, model2, device="cpu")

        assert cka.device == torch.device("cpu")


class TestExtractInput:
    def test_tensor_input(self, model1, model2):
        cka = CKA(model1, model2)
        batch = torch.randn(8, 10)

        result = cka._extract_input(batch)

        assert torch.equal(result, batch)

    def test_list_input(self, model1, model2):
        cka = CKA(model1, model2)
        tensor = torch.randn(8, 10)
        batch = [tensor, torch.randn(8, 10)]

        result = cka._extract_input(batch)

        assert torch.equal(result, tensor)

    def test_tuple_input(self, model1, model2):
        cka = CKA(model1, model2)
        tensor = torch.randn(8, 10)
        batch = (tensor, torch.randn(8, 10))

        result = cka._extract_input(batch)

        assert torch.equal(result, tensor)

    @pytest.mark.parametrize(
        "key", ["input", "inputs", "x", "image", "images", "input_ids", "pixel_values"]
    )
    def test_dict_input_known_keys(self, model1, model2, key):
        cka = CKA(model1, model2)
        tensor = torch.randn(8, 10)
        batch = {key: tensor}

        result = cka._extract_input(batch)

        assert torch.equal(result, tensor)

    def test_dict_input_unknown_keys_raises_error(self, model1, model2):
        cka = CKA(model1, model2)
        batch = {"unknown_key": torch.randn(8, 10)}

        with pytest.raises(ValueError, match="Cannot find input in dict batch"):
            cka._extract_input(batch)

    def test_unknown_type_raises_error(self, model1, model2):
        cka = CKA(model1, model2)
        batch = "not a valid type"

        with pytest.raises(TypeError, match="Unsupported batch type"):
            cka._extract_input(batch)


class TestMakeHook:
    def test_tensor_output(self, model1, model2, dataloader):
        cka = CKA(model1, model2)

        result = cka(dataloader)

        assert result.shape == (3, 3)

    def test_tuple_output(self, model2, dataloader):
        model = TupleOutputModel()
        cka = CKA(model, model2, model1_layers=["layer1"])

        result = cka(dataloader)

        assert result.shape == (1, 3)

    def test_transformer_output(self, model2, dataloader):
        model = TransformerOutputModel()
        cka = CKA(model, model2, model1_layers=["layer1"])

        result = cka(dataloader)

        assert result.shape == (1, 3)

    def test_output_greater_than_2d_flattened(self, model2, image_dataloader):
        model = Conv3DModel()
        cka = CKA(model, model, model1_layers=["conv"], model2_layers=["conv"])

        result = cka(image_dataloader)

        assert result.shape == (1, 1)


class TestCompare:
    def test_hooks_not_registered_raises_error(self, model1, model2, dataloader):
        cka = CKA(model1, model2)

        with pytest.raises(RuntimeError, match="Hooks not registered"):
            cka.compare(dataloader)

    def test_small_batch_raises_error(self, model1, model2, small_batch_dataloader):
        cka = CKA(model1, model2)

        with pytest.raises(ValueError, match="HSIC requires batch size > 3"):
            cka(small_batch_dataloader)

    def test_dataloader2_none_uses_same(self, model1, dataloader):
        cka = CKA(model1, model1)

        result = cka(dataloader, dataloader2=None)

        diagonal = torch.diagonal(result)
        assert torch.allclose(diagonal, torch.ones_like(diagonal), atol=1e-5)

    def test_progress_false(self, model1, model2, dataloader):
        cka = CKA(model1, model2)

        result = cka(dataloader, progress=False)

        assert result.shape == (3, 3)

    def test_progress_true(self, model1, model2, dataloader):
        cka = CKA(model1, model2)

        result = cka(dataloader, progress=True)

        assert result.shape == (3, 3)

    def test_verbose_output(self, model1, model2, dataloader, capsys):
        cka = CKA(model1, model2)

        result = cka(dataloader, verbose=True, progress=False)

        captured = capsys.readouterr()
        assert "Batch 1/4" in captured.out
        assert "Batch 4/4" in captured.out
        assert "Mean CKA:" in captured.out
        assert result.shape[0] > 0

    def test_two_dataloaders(self, model1, model2, dataloader):
        x2 = torch.randn(32, 10)
        dataset2 = TensorDataset(x2)
        dataloader2 = DataLoader(dataset2, batch_size=8)

        cka = CKA(model1, model2)

        result = cka(dataloader, dataloader2=dataloader2)

        assert result.shape == (3, 3)


class TestContextManager:
    def test_idempotent_entry(self, model1, model2, dataloader):
        cka = CKA(model1, model2)

        with cka:
            initial_handles = len(cka._hook_handles)
            cka.__enter__()
            assert len(cka._hook_handles) == initial_handles

    def test_exception_removes_hooks(self, model1, model2):
        cka = CKA(model1, model2)

        with pytest.raises(ValueError):
            with cka:
                assert len(cka._hook_handles) > 0
                raise ValueError("Test exception")

        assert len(cka._hook_handles) == 0

    def test_training_mode_restored(self, model1, model2, dataloader):
        model1.train()
        model2.eval()

        cka = CKA(model1, model2)

        with cka:
            assert not model1.training
            assert not model2.training

        assert model1.training is True
        assert model2.training is False

    def test_features_cleared_after_exit(self, model1, model2, dataloader):
        cka = CKA(model1, model2)

        with cka:
            cka.compare(dataloader, progress=False)

        if cka._is_same_model:
            assert len(cka._shared_features) == 0
        else:
            assert len(cka._model1_layer_to_feature) == 0
            assert len(cka._model2_layer_to_feature) == 0


class TestCall:
    def test_call_as_context_plus_compare(self, model1, model2, dataloader):
        cka = CKA(model1, model2)

        result = cka(dataloader)

        assert result.shape == (3, 3)
        assert len(cka._hook_handles) == 0


class TestAccumulateHsic:
    def test_same_model_same_layers_uses_hsic_outer(self, model1, dataloader):
        cka = CKA(
            model1,
            model1,
            model1_layers=["layer1", "layer2"],
            model2_layers=["layer1", "layer2"],
        )

        result = cka(dataloader)

        diagonal = torch.diagonal(result)
        assert torch.allclose(diagonal, torch.ones_like(diagonal), atol=1e-5)

    def test_same_model_different_layers_uses_hsic(self, model1, dataloader):
        cka = CKA(
            model1,
            model1,
            model1_layers=["layer1", "layer2"],
            model2_layers=["layer2", "layer3"],
        )

        result = cka(dataloader)

        assert result.shape == (2, 2)

    def test_different_models_uses_hsic(self, model1, model2, dataloader):
        cka = CKA(model1, model2)

        result = cka(dataloader)

        assert result.shape == (3, 3)


class TestComputeCkaMatrix:
    def test_values_in_0_1_range(self, model1, model2, dataloader):
        cka = CKA(model1, model1)

        result = cka(dataloader)

        assert torch.all(result >= -0.1)
        assert torch.all(result <= 1.1)

    def test_zero_denominator_handling(self, model1, model2):
        cka = CKA(model1, model2)

        hsic_xy = torch.zeros(3, 3)
        hsic_xx = torch.zeros(3)
        hsic_yy = torch.zeros(3)

        result = cka._compute_cka_matrix(hsic_xy, hsic_xx, hsic_yy)

        assert not torch.any(torch.isnan(result))
        assert not torch.any(torch.isinf(result))


class TestClearFeatures:
    def test_clear_features_same_model(self, model1, dataloader):
        cka = CKA(model1, model1, device="cpu")

        with cka:
            x = torch.randn(8, 10)
            cka.model1(x)
            assert len(cka._shared_features) > 0

            cka._clear_features()
            assert len(cka._shared_features) == 0

    def test_clear_features_different_models(self, model1, model2, dataloader):
        cka = CKA(model1, model2, device="cpu")

        with cka:
            x = torch.randn(8, 10)
            cka.model1(x)
            cka.model2(x)
            assert len(cka._model1_layer_to_feature) > 0
            assert len(cka._model2_layer_to_feature) > 0

            cka._clear_features()
            assert len(cka._model1_layer_to_feature) == 0
            assert len(cka._model2_layer_to_feature) == 0


class TestExport:
    def test_export_structure(self, model1, model2, dataloader):
        cka = CKA(model1, model2, model1_name="Model1", model2_name="Model2")
        result = cka(dataloader)

        exported = cka.export(result)

        assert "model1_name" in exported
        assert "model2_name" in exported
        assert "model1_layers" in exported
        assert "model2_layers" in exported
        assert "cka_matrix" in exported
        assert exported["model1_name"] == "Model1"
        assert exported["model2_name"] == "Model2"


class TestSameModelSharedFeatures:
    def test_shared_layer_to_module_created(self, model1):
        cka = CKA(model1, model1, model1_layers=["layer1"], model2_layers=["layer2"])

        assert hasattr(cka, "_shared_layer_to_module")
        assert "layer1" in cka._shared_layer_to_module
        assert "layer2" in cka._shared_layer_to_module

    def test_shared_features_dict_used(self, model1, dataloader):
        cka = CKA(model1, model1, device="cpu")

        with cka:
            x = torch.randn(8, 10)
            cka.model1(x)

            assert hasattr(cka, "_shared_features")
            assert len(cka._shared_features) > 0
