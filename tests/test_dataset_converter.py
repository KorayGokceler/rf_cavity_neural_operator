"""Tests for RFCavityToGNOT dataset converter.

Kritik invariantlar:
- Boundary node detection (kenarda 1 kez geçen üçgen kenarları)
- PCA principal axis sign disambiguation (deterministic yön)
- Canonical dipole rotation: subspace span'ini koruyup orientasyonu sabitler
- Node area normalization
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
    assert out['principal_axis'].shape == (2,)


def test_principal_axis_sign_disambiguation(converter):
    """Principal axis x-bileşeni daima >= 0 olmalı (deterministic yön)."""
    rng = np.random.default_rng(0)
    for _ in range(10):
        # Rastgele dikdörtgen mesh oluştur
        a = rng.uniform(0.5, 2.0)
        b = rng.uniform(0.5, 2.0)
        nodes = rng.uniform(-1, 1, (40, 2)).astype(np.float64)
        nodes[:, 0] *= a
        nodes[:, 1] *= b
        elements = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        out = converter.extract_geometry_features(nodes, elements)
        pa = out['principal_axis']
        if abs(pa[0]) > 1e-10:
            assert pa[0] > 0
        else:
            assert pa[1] > 0


def test_extract_features_input_funcs_finite(converter):
    """Tüm 8 feature kanalı sonlu olmalı (NaN/inf yok)."""
    nodes, elements = _square_with_interior_mesh()
    out = converter.extract_geometry_features(nodes, elements)
    assert np.isfinite(out['Input_funcs']).all()


def test_canonicalize_dipole_subspace_preserves_span(converter):
    """Rotasyon span'i değiştirmemeli — döndürülmüş vektörler eski subspace içinde olmalı."""
    rng = np.random.default_rng(0)
    n = 50
    # Modes 0/1/2 için rastgele eigenvektörler
    vecs = rng.normal(size=(n, 3))
    nodes_norm = rng.uniform(-1, 1, (n, 2))
    principal_axis = np.array([1.0, 0.0])

    e1_orig = vecs[:, 1].copy()
    e2_orig = vecs[:, 2].copy()

    out = converter._canonicalize_dipole_subspace(vecs, nodes_norm, principal_axis)

    # Mode 0 değişmemeli
    assert np.allclose(out[:, 0], vecs[:, 0])

    # Yeni mode 1 ve mode 2, eski span(e1, e2) içinde olmalı
    # Yani out[:,1] = a*e1_orig + b*e2_orig şeklinde lineer kombinasyon olmalı
    # Bunu kontrol etmek için: rank([e1_orig, e2_orig, out[:,1], out[:,2]]) <= 2
    A = np.column_stack([e1_orig, e2_orig, out[:, 1], out[:, 2]])
    rank = np.linalg.matrix_rank(A, tol=1e-6)
    assert rank <= 2


def test_canonicalize_dipole_aligns_with_principal_axis(converter):
    """Mode 1, principal axis ile pozitif dipole momenti olmalı."""
    rng = np.random.default_rng(1)
    n = 100
    # nodes_norm: x ekseninde simetrik
    x = rng.uniform(-1, 1, n)
    y = rng.uniform(-1, 1, n)
    nodes_norm = np.column_stack([x, y])

    # vecs: mode 1 ve 2 ters konfigürasyon — canonicalization düzeltsin
    e1 = y.copy()  # baştan y-aligned (yani mode 2 olması gereken)
    e2 = x.copy()  # baştan x-aligned (yani mode 1 olması gereken)
    vecs = np.column_stack([np.ones(n), e1, e2])
    principal_axis = np.array([1.0, 0.0])  # x ekseni

    out = converter._canonicalize_dipole_subspace(vecs, nodes_norm, principal_axis)

    # Yeni mode 1, principal axis ile pozitif overlap olmalı
    proj_along = nodes_norm @ principal_axis
    new_dipole_1 = np.dot(out[:, 1], proj_along)
    new_dipole_2 = np.dot(out[:, 2], proj_along)
    # Mode 1 principal axis ile aligned
    assert new_dipole_1 > abs(new_dipole_2) - 1e-6


def test_canonicalize_returns_same_shape(converter):
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(30, 3))
    nodes_norm = rng.uniform(-1, 1, (30, 2))
    principal_axis = np.array([0.0, 1.0])
    out = converter._canonicalize_dipole_subspace(vecs, nodes_norm, principal_axis)
    assert out.shape == vecs.shape


def test_canonicalize_with_fewer_than_3_modes(converter):
    """Mode sayısı < 3 ise canonicalization no-op olmalı (aynı vecs döner)."""
    rng = np.random.default_rng(0)
    vecs = rng.normal(size=(20, 2))  # sadece 2 mod
    nodes_norm = rng.uniform(-1, 1, (20, 2))
    principal_axis = np.array([1.0, 0.0])
    out = converter._canonicalize_dipole_subspace(vecs, nodes_norm, principal_axis)
    assert np.array_equal(out, vecs)


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
