"""Tests for RandomFourierFeatures.

Random Fourier Features yaklaşımının doğru implement edildiğini doğrular:
  phi(x) = sqrt(2/D) * [cos(B x), sin(B x)],  B_ij ~ N(0, 1/length_scale^2)
B sabit (frozen) olmalı — Bochner teoremi yalnızca rastgele örneklenip dondurulduğunda
unbiased Gaussian kernel approximation verir.
"""
import math

import pytest
import torch

from src.models.gnot import RandomFourierFeatures


def test_output_shape():
    rff = RandomFourierFeatures(in_dim=2, out_dim=64, length_scale=0.1)
    x = torch.randn(4, 100, 2)
    y = rff(x)
    assert y.shape == (4, 100, 64)


def test_output_shape_flat():
    rff = RandomFourierFeatures(in_dim=3, out_dim=32, length_scale=0.5)
    x = torch.randn(50, 3)
    y = rff(x)
    assert y.shape == (50, 32)


def test_odd_out_dim_raises():
    with pytest.raises(ValueError, match="even"):
        RandomFourierFeatures(in_dim=2, out_dim=33, length_scale=0.1)


def test_b_is_frozen_buffer():
    """B matrisi frozen — gradyan akmamalı, parametre olarak görünmemeli."""
    rff = RandomFourierFeatures(in_dim=2, out_dim=16, length_scale=0.1)
    # B parameter listesinde olmamalı
    param_names = {n for n, _ in rff.named_parameters()}
    assert 'B' not in param_names
    # B buffer listesinde olmalı
    buffer_names = {n for n, _ in rff.named_buffers()}
    assert 'B' in buffer_names
    assert 'scale' in buffer_names


def test_no_gradient_through_b():
    rff = RandomFourierFeatures(in_dim=2, out_dim=16, length_scale=0.1)
    x = torch.randn(10, 2, requires_grad=True)
    y = rff(x)
    y.sum().backward()
    # B'nin grad'ı olmamalı
    assert rff.B.grad is None
    # x.grad olmalı (RFF bir non-trivial fonksiyon)
    assert x.grad is not None


def test_deterministic_for_same_input():
    """Aynı input → aynı output (B sabit olduğu için)."""
    torch.manual_seed(42)
    rff = RandomFourierFeatures(in_dim=2, out_dim=16, length_scale=0.1)
    x = torch.randn(5, 2)
    y1 = rff(x)
    y2 = rff(x)
    assert torch.equal(y1, y2)


def test_different_inputs_different_outputs():
    rff = RandomFourierFeatures(in_dim=2, out_dim=16, length_scale=0.1)
    x1 = torch.randn(3, 2)
    x2 = x1 + 1.0
    y1 = rff(x1)
    y2 = rff(x2)
    assert not torch.allclose(y1, y2)


def test_output_bounded():
    """Her boyutta |phi(x)| <= sqrt(2/D) (cos/sin |.| <= 1)."""
    out_dim = 32
    rff = RandomFourierFeatures(in_dim=2, out_dim=out_dim, length_scale=0.1)
    x = torch.randn(100, 2) * 10  # büyük değerler bile olsa
    y = rff(x)
    bound = math.sqrt(2.0 / out_dim)
    assert y.abs().max().item() <= bound + 1e-6


def test_normalization_factor():
    """Scale buffer'ı sqrt(2/D) olmalı."""
    out_dim = 64
    rff = RandomFourierFeatures(in_dim=2, out_dim=out_dim, length_scale=0.5)
    expected_scale = math.sqrt(2.0 / out_dim)
    assert abs(rff.scale.item() - expected_scale) < 1e-6


def test_sigma_scales_with_inverse_length_scale():
    """B'nin std'i 1/length_scale olmalı (Gaussian kernel için)."""
    torch.manual_seed(0)
    out_dim = 4096  # büyük örneklem için stable std tahmini
    ls = 0.2
    rff = RandomFourierFeatures(in_dim=2, out_dim=out_dim, length_scale=ls)
    expected_sigma = 1.0 / ls
    actual_sigma = rff.B.std().item()
    # ~5% tolerance for stochastic sampling
    assert abs(actual_sigma - expected_sigma) / expected_sigma < 0.05


def test_kernel_approximation_at_zero():
    """phi(0) = sqrt(2/D) * [cos(0), sin(0)] = sqrt(2/D) * [1, 0].
    Bu nedenle <phi(0), phi(0)> = (2/D) * (D/2) = 1 (k(0,0)=1)."""
    out_dim = 256
    rff = RandomFourierFeatures(in_dim=2, out_dim=out_dim, length_scale=0.1)
    x = torch.zeros(1, 2)
    y = rff(x)
    inner = (y * y).sum().item()
    assert abs(inner - 1.0) < 1e-5


def test_grid_dim_one():
    """1D inputs (grid_dim=1) çalışmalı."""
    rff = RandomFourierFeatures(in_dim=1, out_dim=16, length_scale=0.1)
    x = torch.randn(20, 1)
    y = rff(x)
    assert y.shape == (20, 16)
    assert torch.isfinite(y).all()
