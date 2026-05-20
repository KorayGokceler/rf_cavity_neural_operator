"""Tests for RFCavityToGNOT dataset converter.

Kritik invariantlar:
- Boundary node detection (kenarda 1 kez geçen üçgen kenarları)
- Node area normalization
- Input features shape and finiteness (val_dim = 8)
- dist_to_boundary = 0 at boundary nodes
"""
import numpy as np
import pytest

from src.data.dataset_converter import RFCavityToGNOT


@pytest.fixture
def converter():
    # Dosya yolu kullanılmıyor (h5 dosyasına dokunmadan helper'ları test ediyoruz)
    return RFCavityToGNOT(h5_filepath="dummy.h5")


def _square_mesh():
    """Basit 1x1 kare mesh: 4 köşe, 2 üçgen.
    Hepsi boundary node — iç node yok."""
    nodes = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ], dtype=np.float64)
    elements = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.int32)
    return nodes, elements


def _square_with_interior_mesh():
    """Kare + ortada bir iç node. 1 iç node, 4 köşe boundary."""
    nodes = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.5, 0.5],  # iç node
    ], dtype=np.float64)
    elements = np.array([
        [0, 1, 4],
        [1, 2, 4],
        [2, 3, 4],
        [3, 0, 4],
    ], dtype=np.int32)
    return nodes, elements


def test_find_boundary_nodes_simple_square(converter):
    nodes, elements = _square_mesh()
    bnd = converter._find_boundary_nodes(elements)
    # 4 köşenin tümü boundary olmalı
    assert set(bnd.tolist()) == {0, 1, 2, 3}


def test_find_boundary_nodes_with_interior(converter):
    nodes, elements = _square_with_interior_mesh()
    bnd = converter._find_boundary_nodes(elements)
    # İç node (4) boundary'de olmamalı
    assert 4 not in bnd
    assert set(bnd.tolist()) == {0, 1, 2, 3}


def test_node_areas_sum_proportional(converter):
    """Tüm node areas pozitif ve normalize (max=1) olmalı."""
    nodes, elements = _square_with_interior_mesh()
    areas = converter._compute_node_areas(nodes, elements)
    assert areas.shape == (5, 1)
    assert (areas >= 0).all()
    assert abs(areas.max() - 1.0) < 1e-6


def test_extract_geometry_features_shape(converter):
    nodes, elements = _square_with_interior_mesh()
    out = converter.extract_geometry_features(nodes, elements)
    n = len(nodes)
    assert out['X'].shape == (n, 2)
    assert out['Input_funcs'].shape == (n, 8)  # val_dim = 8
    assert 'principal_axis' not in out  # internal only, not exposed


def test_extract_features_input_funcs_finite(converter):
    """Tüm 8 feature kanalı sonlu olmalı (NaN/inf yok)."""
    nodes, elements = _square_with_interior_mesh()
    out = converter.extract_geometry_features(nodes, elements)
    assert np.isfinite(out['Input_funcs']).all()


def test_dist_to_boundary_zero_at_boundary(converter):
    """Boundary node'larda dist_to_boundary = 0 olmalı."""
    nodes, elements = _square_with_interior_mesh()
    out = converter.extract_geometry_features(nodes, elements)
    # 4 boundary node'un dist'i 0 olmalı
    bnd_indices = converter._find_boundary_nodes(elements)
    dist_bnd = out['Input_funcs'][:, 2]
    for idx in bnd_indices:
        assert abs(dist_bnd[idx]) < 1e-6
    # İç node'da pozitif olmalı
    interior = [i for i in range(len(nodes)) if i not in bnd_indices]
    for idx in interior:
        assert dist_bnd[idx] > 0
