import pytest
import torch
import torch.nn as nn
from networks.lora_flux import LoRAModule, LoRANetwork, create_network
from unittest.mock import MagicMock

def generate_synthetic_weights(org_weight, seed=42):
    generator = torch.manual_seed(seed)

    # Base random normal distribution
    weights = torch.randn_like(org_weight)

    # Add structured variance to mimic real-world weight matrices
    # Techniques to create more realistic weight distributions:

    # 1. Block-wise variation
    block_size = max(1, org_weight.shape[0] // 4)
    for i in range(0, org_weight.shape[0], block_size):
        block_end = min(i + block_size, org_weight.shape[0])
        block_variation = torch.randn(1, generator=generator) * 0.3  # Local scaling
        weights[i:block_end, :] *= 1 + block_variation

    # 2. Sparse connectivity simulation
    sparsity_mask = torch.rand(org_weight.shape, generator=generator) > 0.2  # 20% sparsity
    weights *= sparsity_mask.float()

    # 3. Magnitude decay
    magnitude_decay = torch.linspace(1.0, 0.5, org_weight.shape[0]).unsqueeze(1)
    weights *= magnitude_decay

    # 4. Add structured noise
    structural_noise = torch.randn_like(org_weight) * 0.1
    weights += structural_noise

    # Normalize to have similar statistical properties to trained weights
    weights = (weights - weights.mean()) / weights.std()

    return weights

def test_basic_linear_module_initialization():
    # Test basic Linear module initialization
    org_module = nn.Linear(10, 20)
    lora_module = LoRAModule(lora_name="test_linear", org_module=org_module, lora_dim=4)

    # Check basic attributes
    assert lora_module.lora_name == "test_linear"
    assert lora_module.lora_dim == 4

    # Check LoRA layers
    assert isinstance(lora_module.lora_down, nn.Linear)
    assert isinstance(lora_module.lora_up, nn.Linear)

    # Check input and output dimensions
    assert lora_module.lora_down.in_features == 10
    assert lora_module.lora_down.out_features == 4
    assert lora_module.lora_up.in_features == 4
    assert lora_module.lora_up.out_features == 20


def test_split_dims_initialization():
    # Test initialization with split_dims
    org_module = nn.Linear(10, 15)
    lora_module = LoRAModule(lora_name="test_split_dims", org_module=org_module, lora_dim=4, split_dims=[5, 5, 5])

    # Check split_dims specific attributes
    assert lora_module.split_dims == [5, 5, 5]
    assert isinstance(lora_module.lora_down, nn.ModuleList)
    assert isinstance(lora_module.lora_up, nn.ModuleList)

    # Check number of split modules
    assert len(lora_module.lora_down) == 3
    assert len(lora_module.lora_up) == 3

    # Check dimensions of split modules
    for down, up in zip(lora_module.lora_down, lora_module.lora_up):
        assert down.in_features == 10
        assert down.out_features == 4
        assert up.in_features == 4
        assert up.out_features in [5, 5, 5]


def test_alpha_scaling():
    # Test alpha scaling
    org_module = nn.Linear(10, 20)

    # Default alpha (should be equal to lora_dim)
    lora_module1 = LoRAModule(lora_name="test_alpha1", org_module=org_module, lora_dim=4, alpha=0)
    assert lora_module1.scale == 1.0

    # Custom alpha
    lora_module2 = LoRAModule(lora_name="test_alpha2", org_module=org_module, lora_dim=4, alpha=2)
    assert lora_module2.scale == 0.5


def test_initialization_methods():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Test different initialization methods
    org_module = nn.Linear(10, 20)
    org_module.weight.data = generate_synthetic_weights(org_module.weight)

    # Default initialization
    lora_module1 = LoRAModule(lora_name="test_init_default", org_module=org_module, lora_dim=4)

    assert lora_module1.lora_down.weight.shape == (4, 10)
    assert lora_module1.lora_up.weight.shape == (20, 4)

    # URAE initialization
    lora_module2 = LoRAModule(lora_name="test_init_urae", org_module=org_module, lora_dim=4)
    lora_module2.initialize_weights(org_module, "urae", device)
    assert hasattr(lora_module2, "_org_lora_up") and lora_module2._org_lora_down is not None
    assert hasattr(lora_module2, "_org_lora_down") and lora_module2._org_lora_down is not None

    assert lora_module2.lora_down.weight.shape == (4, 10)
    assert lora_module2.lora_up.weight.shape == (20, 4)

    # PISSA initialization
    lora_module3 = LoRAModule(lora_name="test_init_pissa", org_module=org_module, lora_dim=4)
    lora_module3.initialize_weights(org_module, "pissa", device)
    assert hasattr(lora_module3, "_org_lora_up") and lora_module3._org_lora_down is not None
    assert hasattr(lora_module3, "_org_lora_down") and lora_module3._org_lora_down is not None

    assert lora_module3.lora_down.weight.shape == (4, 10)
    assert lora_module3.lora_up.weight.shape == (20, 4)


@torch.no_grad()
def test_forward_basic_linear():
    # Create a basic linear module
    org_module = nn.Linear(10, 20)
    org_module.weight.data = torch.testing.make_tensor(
        org_module.weight.data.shape, dtype=torch.float32, device="cpu", low=0.1, high=1.0
    )

    lora_module = LoRAModule(lora_name="test_forward", org_module=org_module, lora_dim=4, alpha=4, multiplier=1.0)
    lora_module.apply_to()

    assert isinstance(lora_module.lora_down, nn.Linear)
    assert isinstance(lora_module.lora_up, nn.Linear)

    lora_module.lora_down.weight.data = torch.testing.make_tensor(
        lora_module.lora_down.weight.shape, dtype=torch.float32, device="cpu", low=0.1, high=1.0
    )
    lora_module.lora_up.weight.data = torch.testing.make_tensor(
        lora_module.lora_up.weight.shape, dtype=torch.float32, device="cpu", low=0.1, high=1.0
    )

    # Create input
    x = torch.ones(5, 10)

    # Perform forward pass
    output = lora_module.forward(x)

    # Structural assertions
    assert output is not None, "Output should not be None"
    assert isinstance(output, torch.Tensor), "Output should be a torch.Tensor"

    # Shape assertions
    assert output.shape == (5, 20), "Output shape should match expected dimensions"

    # Type and device assertions
    assert output.dtype == torch.float32, "Output should be float32"
    assert output.device == x.device, "Output should be on the same device as input"


def test_forward_module_dropout():
    # Create a basic linear module
    org_module = nn.Linear(10, 20)

    lora_module = LoRAModule(
        lora_name="test_module_dropout",
        org_module=org_module,
        lora_dim=4,
        multiplier=1.0,
        module_dropout=1.0,  # Always drop
    )

    lora_module.apply_to()

    # Create input
    x = torch.ones(5, 10)

    # Enable training mode
    lora_module.train()

    # Perform forward pass
    output = lora_module.forward(x)

    # Check if output is same as original module output
    org_output = org_module(x)
    torch.testing.assert_close(output, org_output)


def test_forward_rank_dropout():
    # Create a basic linear module
    org_module = nn.Linear(10, 20)

    lora_module = LoRAModule(
        lora_name="test_rank_dropout",
        org_module=org_module,
        lora_dim=4,
        multiplier=1.0,
        rank_dropout=0.5,  # 50% dropout
    )

    lora_module.apply_to()

    assert isinstance(lora_module.lora_down, nn.Linear)
    assert isinstance(lora_module.lora_up, nn.Linear)

    # Make lora weights predictable
    lora_module.lora_down.weight.data = torch.testing.make_tensor(
        lora_module.lora_down.weight.shape, dtype=torch.float32, device="cpu", low=0.1, high=1.0
    )
    lora_module.lora_up.weight.data = torch.testing.make_tensor(
        lora_module.lora_up.weight.shape, dtype=torch.float32, device="cpu", low=0.1, high=1.0
    )

    # Create input
    x = torch.ones(5, 10)

    # Enable training mode
    lora_module.train()

    # Perform multiple forward passes to show dropout effect
    outputs = [lora_module.forward(x) for _ in range(10)]

    # Check that outputs are not all identical due to rank dropout
    differences = [
        torch.all(torch.eq(outputs[i], outputs[j])).item() for i in range(len(outputs)) for j in range(i + 1, len(outputs))
    ]
    assert not all(differences)


def test_forward_split_dims():
    # Create a basic linear module with split dimensions
    org_module = nn.Linear(10, 15)

    lora_module = LoRAModule(lora_name="test_split_dims", org_module=org_module, lora_dim=4, multiplier=1.0, split_dims=[5, 5, 5])

    lora_module.apply_to()

    assert isinstance(lora_module.lora_down, nn.ModuleList)
    assert isinstance(lora_module.lora_up, nn.ModuleList)

    # Make lora weights predictable
    for down in lora_module.lora_down:
        assert isinstance(down, nn.Linear)
        down.weight.data = torch.testing.make_tensor(down.weight.data.shape, dtype=torch.float32, device="cpu", low=0.1, high=1.0)
    for up in lora_module.lora_up:
        assert isinstance(up, nn.Linear)
        up.weight.data = torch.testing.make_tensor(up.weight.data.shape, dtype=torch.float32, device="cpu", low=0.1, high=1.0)

    # Create input
    x = torch.ones(5, 10)

    # Perform forward pass
    output = lora_module.forward(x)

    # Check output dimensions
    assert output.shape == (5, 15)


def test_forward_dropout():
    # Create a basic linear module
    org_module = nn.Linear(10, 20)

    lora_module = LoRAModule(
        lora_name="test_dropout",
        org_module=org_module,
        lora_dim=4,
        multiplier=1.0,
        dropout=0.5,  # 50% dropout
    )

    lora_module.apply_to()

    assert isinstance(lora_module.lora_down, nn.Linear)
    assert isinstance(lora_module.lora_up, nn.Linear)

    # Make lora weights predictable
    lora_module.lora_down.weight.data = torch.testing.make_tensor(
        lora_module.lora_down.weight.shape, dtype=torch.float32, device="cpu", low=0.1, high=1.0
    )
    lora_module.lora_up.weight.data = torch.testing.make_tensor(
        lora_module.lora_up.weight.shape, dtype=torch.float32, device="cpu", low=0.1, high=1.0
    )

    # Create input
    x = torch.ones(5, 10)

    # Enable training mode
    lora_module.train()

    # Perform multiple forward passes to show dropout effect
    outputs = [lora_module.forward(x) for _ in range(10)]

    # Check that outputs are not all identical due to dropout
    differences = [
        torch.all(torch.eq(outputs[i], outputs[j])).item() for i in range(len(outputs)) for j in range(i + 1, len(outputs))
    ]
    assert not all(differences)


def test_create_network_default_parameters(mock_text_encoder, mock_flux):
    # Mock dependencies
    mock_ae = MagicMock()
    mock_text_encoders = [mock_text_encoder, mock_text_encoder]

    # Call the function with minimal parameters
    network = create_network(
        multiplier=1.0, network_dim=None, network_alpha=None, ae=mock_ae, text_encoders=mock_text_encoders, flux=mock_flux
    )

    # Assertions
    assert network is not None
    assert network.multiplier == 1.0
    assert network.lora_dim == 4  # default network_dim
    assert network.alpha == 1.0  # default network_alpha


@pytest.fixture
def mock_text_encoder():
    class CLIPAttention(nn.Module):
        def __init__(self):
            super().__init__()
            # Add some dummy layers to simulate a CLIPAttention
            self.layers = torch.nn.ModuleList([torch.nn.Linear(10, 15) for _ in range(3)])

    class MockTextEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            # Add some dummy layers to simulate a CLIPTextModel
            self.attns = torch.nn.ModuleList([CLIPAttention() for _ in range(3)])

    return MockTextEncoder()


@pytest.fixture
def mock_flux():
    class DoubleStreamBlock(nn.Module):
        def __init__(self):
            super().__init__()
            # Add some dummy layers to simulate a DoubleStreamBlock
            self.layers = torch.nn.ModuleList([torch.nn.Linear(10, 15) for _ in range(3)])

    class SingleStreamBlock(nn.Module):
        def __init__(self):
            super().__init__()
            # Add some dummy layers to simulate a SingleStreamBlock
            self.layers = torch.nn.ModuleList([torch.nn.Linear(10, 15) for _ in range(3)])

    class MockFlux(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Add some dummy layers to simulate a Flux
            self.double_blocks = torch.nn.ModuleList([DoubleStreamBlock() for _ in range(3)])
            self.single_blocks = torch.nn.ModuleList([SingleStreamBlock() for _ in range(3)])

    return MockFlux()


def test_create_network_custom_parameters(mock_text_encoder, mock_flux):
    # Mock dependencies
    mock_ae = MagicMock()
    mock_text_encoders = [mock_text_encoder, mock_text_encoder]

    # Prepare custom parameters
    custom_params = {
        "conv_dim": 8,
        "conv_alpha": 0.5,
        "img_attn_dim": 16,
        "txt_attn_dim": 16,
        "neuron_dropout": 0.1,
        "rank_dropout": 0.2,
        "module_dropout": 0.3,
        "train_blocks": "double",
        "split_qkv": "True",
        "train_t5xxl": "True",
        "in_dims": "[64, 32, 16, 8, 4]",
        "verbose": "True",
    }

    # Call the function with custom parameters
    network = create_network(
        multiplier=1.5,
        network_dim=8,
        network_alpha=2.0,
        ae=mock_ae,
        text_encoders=mock_text_encoders,
        flux=mock_flux,
        **custom_params,
    )

    # Assertions
    assert network is not None
    assert network.multiplier == 1.5
    assert network.lora_dim == 8
    assert network.alpha == 2.0
    assert network.conv_lora_dim == 8
    assert network.conv_alpha == 0.5
    assert network.train_blocks == "double"
    assert network.split_qkv is True
    assert network.train_t5xxl is True


def test_create_network_block_indices(mock_text_encoder, mock_flux):
    # Mock dependencies
    mock_ae = MagicMock()
    mock_text_encoders = [mock_text_encoder, mock_text_encoder]

    # Test block indices parsing
    network = create_network(
        multiplier=1.0,
        network_dim=4,
        network_alpha=1.0,
        ae=mock_ae,
        text_encoders=mock_text_encoders,
        flux=mock_flux,
        neuron_dropout=None,
        **{"train_double_block_indices": "0-2,4", "train_single_block_indices": "1,3"},
    )

    # Assertions would depend on the exact implementation of parsing
    assert network.train_double_block_indices is not None
    assert network.train_single_block_indices is not None

    double_block_indices = [
        True,
        True,
        True,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    single_block_indices = [
        False,
        True,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]

    assert network.train_double_block_indices == double_block_indices
    assert network.train_single_block_indices == single_block_indices


def test_create_network_loraplus_ratios(mock_text_encoder, mock_flux):
    # Mock dependencies
    mock_ae = MagicMock()
    mock_text_encoders = [mock_text_encoder, mock_text_encoder]

    # Test LoRA+ ratios
    network = create_network(
        multiplier=1.0,
        network_dim=4,
        network_alpha=1.0,
        ae=mock_ae,
        text_encoders=mock_text_encoders,
        flux=mock_flux,
        neuron_dropout=None,
        **{"loraplus_lr_ratio": 2.0, "loraplus_unet_lr_ratio": 1.5, "loraplus_text_encoder_lr_ratio": 1.0},
    )

    # Verify LoRA+ ratios were set correctly
    assert network.loraplus_lr_ratio == 2.0
    assert network.loraplus_unet_lr_ratio == 1.5
    assert network.loraplus_text_encoder_lr_ratio == 1.0


def test_create_network_loraplus_default_ratio(mock_text_encoder, mock_flux):
    # Mock dependencies
    mock_ae = MagicMock()
    mock_text_encoders = [mock_text_encoder, mock_text_encoder]

    # Test when only global LoRA+ ratio is provided
    network = create_network(
        multiplier=1.0,
        network_dim=4,
        network_alpha=1.0,
        ae=mock_ae,
        text_encoders=mock_text_encoders,
        flux=mock_flux,
        nueral_dropout=None,
        **{"loraplus_lr_ratio": 2.0},
    )

    # Verify only global ratio is set
    assert network.loraplus_lr_ratio == 2.0
    assert network.loraplus_unet_lr_ratio is None
    assert network.loraplus_text_encoder_lr_ratio is None


def test_create_network_invalid_inputs(mock_text_encoder, mock_flux):
    # Mock dependencies
    mock_ae = MagicMock()
    mock_text_encoders = [mock_text_encoder, mock_text_encoder]
    mock_flux = mock_flux

    # Test invalid train_blocks
    with pytest.raises(AssertionError):
        create_network(
            multiplier=1.0,
            network_dim=4,
            network_alpha=1.0,
            ae=mock_ae,
            text_encoders=mock_text_encoders,
            flux=mock_flux,
            neuron_dropout=None,
            **{"train_blocks": "invalid"},
        )

    # Test invalid in_dims
    with pytest.raises(AssertionError):
        create_network(
            multiplier=1.0,
            network_dim=4,
            network_alpha=1.0,
            ae=mock_ae,
            text_encoders=mock_text_encoders,
            flux=mock_flux,
            neuron_dropout=None,
            **{"in_dims": "[1,2,3]"},  # Should be 5 dimensions
        )


def test_lora_network_initialization(mock_text_encoder, mock_flux):
    # Test basic initialization with default parameters
    lora_network = LoRANetwork(text_encoders=[mock_text_encoder, mock_text_encoder], unet=mock_flux)

    # Check basic attributes
    assert lora_network.multiplier == 1.0
    assert lora_network.lora_dim == 4
    assert lora_network.alpha == 1
    assert lora_network.train_blocks == "all"

    # Check LoRA modules are created
    assert len(lora_network.text_encoder_loras) > 0
    assert len(lora_network.unet_loras) > 0


def test_lora_network_initialization_with_custom_params(mock_text_encoder, mock_flux):
    # Test initialization with custom parameters
    lora_network = LoRANetwork(
        text_encoders=[mock_text_encoder],
        unet=mock_flux,
        multiplier=0.5,
        lora_dim=8,
        alpha=2.0,
        dropout=0.1,
        rank_dropout=0.05,
        module_dropout=0.02,
        train_blocks="single",
        split_qkv=True,
    )

    # Verify custom parameters are set correctly
    assert lora_network.multiplier == 0.5
    assert lora_network.lora_dim == 8
    assert lora_network.alpha == 2.0
    assert lora_network.dropout == 0.1
    assert lora_network.rank_dropout == 0.05
    assert lora_network.module_dropout == 0.02
    assert lora_network.train_blocks == "single"
    assert lora_network.split_qkv is True


def test_lora_network_initialization_with_custom_modules_dim(mock_text_encoder, mock_flux):
    # Test initialization with custom module dimensions
    modules_dim = {"lora_te1_attns_0_layers_0": 16, "lora_unet_double_blocks_0_layers_0": 8}
    modules_alpha = {"lora_te1_attns_0_layers_0": 2, "lora_unet_double_blocks_0_layers_0": 1}

    lora_network = LoRANetwork(
        text_encoders=[mock_text_encoder, mock_text_encoder], unet=mock_flux, modules_dim=modules_dim, modules_alpha=modules_alpha
    )

    # [LoRAModule(
    #   (lora_down): Linear(in_features=10, out_features=8, bias=False)
    #   (lora_up): Linear(in_features=8, out_features=15, bias=False)
    #   (org_module): Linear(in_features=10, out_features=15, bias=True)
    # )]
    # [LoRAModule(
    #   (lora_down): Linear(in_features=10, out_features=16, bias=False)
    #   (lora_up): Linear(in_features=16, out_features=15, bias=False)
    #   (org_module): Linear(in_features=10, out_features=15, bias=True)
    # )]

    assert isinstance(lora_network.unet_loras[0].lora_down, torch.nn.Linear)
    assert isinstance(lora_network.unet_loras[0].lora_up, torch.nn.Linear)
    assert lora_network.unet_loras[0].lora_down.weight.data.shape[0] == modules_dim["lora_unet_double_blocks_0_layers_0"]
    assert lora_network.unet_loras[0].lora_up.weight.data.shape[1] == modules_dim["lora_unet_double_blocks_0_layers_0"]
    assert lora_network.unet_loras[0].alpha == modules_alpha["lora_unet_double_blocks_0_layers_0"]

    assert isinstance(lora_network.text_encoder_loras[0].lora_down, torch.nn.Linear)
    assert isinstance(lora_network.text_encoder_loras[0].lora_up, torch.nn.Linear)
    assert lora_network.text_encoder_loras[0].lora_down.weight.data.shape[0] == modules_dim["lora_te1_attns_0_layers_0"]
    assert lora_network.text_encoder_loras[0].lora_up.weight.data.shape[1] == modules_dim["lora_te1_attns_0_layers_0"]
    assert lora_network.text_encoder_loras[0].alpha == modules_alpha["lora_te1_attns_0_layers_0"]
