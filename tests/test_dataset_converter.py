"""Tests for RFCavityToGNOT dataset converter.

Kritik invariantlar:
- Boundary node detection (kenarda 1 kez geçen üçgen kenarları)
- Node area normalization
- Input features shape and finiteness (val_dim = 13)
- Boundary curvature sign is canonical (+ convex, - concave), mirror-invariant
- convert_dataset: mode ordering, mode_idx, scale, clear errors
- dist_to_boundary = 0 at boundary nodes
"""
import pickle

import h5py
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
    # val_dim = 13 (41a830d: dist_2nd, dist_3rd, curvature, convexity; + torsion)
    assert out['Input_funcs'].shape == (n, 13)
    assert 'principal_axis' not in out  # internal only, not exposed


def test_extract_features_input_funcs_finite(converter):
    """Tüm 13 feature kanalı sonlu olmalı (NaN/inf yok)."""
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


# ── Boundary loops / curvature ───────────────────────────────────────────────

def _polar_mesh(r_of_theta, n_theta=48, n_rings=4, r_inner=None):
    """Structured polar mesh. r_inner=None → disk (with center node), else annulus."""
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    r_out = r_of_theta(theta)
    nodes, rings = [], []
    if r_inner is None:
        nodes.append([0.0, 0.0])
        fracs = np.linspace(0, 1, n_rings + 1)[1:]
    else:
        fracs = np.linspace(r_inner, 1, n_rings + 1)
    for f in fracs:
        rings.append(np.arange(len(nodes), len(nodes) + n_theta))
        nodes.extend(np.c_[f * r_out * np.cos(theta), f * r_out * np.sin(theta)])
    elements = []
    if r_inner is None:
        for j in range(n_theta):
            elements.append([0, rings[0][j], rings[0][(j + 1) % n_theta]])
    for a, b in zip(rings[:-1], rings[1:]):
        for j in range(n_theta):
            j1 = (j + 1) % n_theta
            elements.append([a[j], b[j], b[j1]])
            elements.append([a[j], b[j1], a[j1]])
    return np.asarray(nodes), np.asarray(elements, dtype=np.int32)


def _l_shape_mesh():
    """L-shaped domain on a unit grid: re-entrant corner at (1, 1)."""
    pts = {}
    nodes = []
    for y in range(3):
        for x in range(3):
            if x == 2 and y == 2:
                continue
            pts[(x, y)] = len(nodes)
            nodes.append([float(x), float(y)])
    elements = []
    for (x, y) in [(0, 0), (1, 0), (0, 1)]:
        a, b, c, d = pts[(x, y)], pts[(x + 1, y)], pts[(x + 1, y + 1)], pts[(x, y + 1)]
        elements += [[a, b, c], [a, c, d]]
    return np.asarray(nodes), np.asarray(elements, dtype=np.int32), pts[(1, 1)]


def test_boundary_loops_disk_and_annulus(converter):
    nodes, el = _polar_mesh(lambda t: np.ones_like(t))
    loops = converter._boundary_loops(nodes, el)
    assert [len(l) for l in loops] == [48]
    nodes, el = _polar_mesh(lambda t: np.ones_like(t), r_inner=0.4)
    loops = converter._boundary_loops(nodes, el)
    assert sorted(len(l) for l in loops) == [48, 48]  # outer wall + inner wall


def test_curvature_sign_convex_circle_and_mirror_invariant(converter):
    """Circle: every wall node convex (+). Old greedy-NN ordering gave a random
    sign per geometry and flipped it under mirroring."""
    nodes, el = _polar_mesh(lambda t: 1.0 + 0.2 * np.cos(2 * t))  # convex ellipse-like
    bnd = converter._find_boundary_nodes(el)
    curv = converter._compute_boundary_curvature(nodes, el, bnd)
    assert (curv > 0).all()
    mirrored = nodes * np.array([-1.0, 1.0])
    f1 = converter.extract_geometry_features(nodes, el)['Input_funcs']
    f2 = converter.extract_geometry_features(mirrored, el)['Input_funcs']
    np.testing.assert_allclose(f1[:, 10], f2[:, 10], atol=1e-6)
    np.testing.assert_allclose(f1[:, 11], f2[:, 11], atol=1e-6)
    # Reversing triangle winding must not change it either
    f3 = converter.extract_geometry_features(nodes, el[:, [0, 2, 1]])['Input_funcs']
    np.testing.assert_allclose(f1[:, 10], f3[:, 10], atol=1e-6)


def test_curvature_sign_annulus_inner_wall_concave(converter):
    nodes, el = _polar_mesh(lambda t: np.ones_like(t), r_inner=0.4)
    bnd = converter._find_boundary_nodes(el)
    curv = converter._compute_boundary_curvature(nodes, el, bnd)
    radius = np.linalg.norm(nodes[bnd], axis=1)
    assert (curv[radius > 0.9] > 0).all()   # outer wall convex
    assert (curv[radius < 0.5] < 0).all()   # inner wall (hole) concave


def test_curvature_sign_l_shape_reentrant_corner(converter):
    nodes, el, corner = _l_shape_mesh()
    bnd = converter._find_boundary_nodes(el)
    curv = dict(zip(bnd.tolist(), converter._compute_boundary_curvature(nodes, el, bnd)))
    assert curv[corner] < 0                                  # re-entrant corner
    assert curv[0] > 0                                       # convex corner (0, 0)
    assert abs(curv[1]) < 1e-6                               # straight wall (1, 0)


def test_extract_features_exposes_physical_scale(converter):
    nodes, el = _square_with_interior_mesh()
    out = converter.extract_geometry_features(nodes * 0.08 + 0.01, el)
    np.testing.assert_allclose(out['X'] * out['scale'] + out['center'], nodes * 0.08 + 0.01, atol=1e-6)


# ── convert_dataset end-to-end on a synthetic raw H5 ─────────────────────────

def _write_raw_h5(path, n_geoms=3, freqs_order=(0, 1, 2), n_p2_extra=7):
    nodes, el = _polar_mesh(lambda t: 0.04 * np.ones_like(t), n_theta=16, n_rings=2)
    n = len(nodes)
    rng = np.random.default_rng(0)
    with h5py.File(path, 'w') as f:
        f.attrs['metadata'] = '{}'  # file attrs must not be mistaken for samples
        for g in range(n_geoms):
            grp = f.create_group(f"sample_{g:04d}")
            grp.create_dataset('nodes', data=nodes)
            grp.create_dataset('elements', data=el)
            base = np.array([3.0, 4.0, 5.0]) + g
            grp.create_dataset('freqs', data=base[list(freqs_order)])
            vecs = rng.normal(size=(n + n_p2_extra, 3))
            # vertex part of column j identifies its true (sorted) mode m: ramp^(m+1)
            vecs[:n, :] = np.stack([np.arange(1, n + 1) ** (m + 1.0) for m in freqs_order], axis=1)
            grp.create_dataset('vecs', data=vecs)
            grp.attrs['shape_type'] = 'circle'
    return n


def test_convert_dataset_pkl_roundtrip(tmp_path):
    h5 = tmp_path / 'raw.h5'
    n = _write_raw_h5(h5, freqs_order=(0, 2, 1))  # unsorted, like old skfem `eigs` output
    out = tmp_path / 'out.pkl'
    RFCavityToGNOT(str(h5)).convert_dataset(str(out), mode_indices=[1, 2])
    data = pickle.load(open(out, 'rb'))
    assert len(data['geometry_pool']) == 3
    geom = data['geometry_pool'][0]
    assert geom['Input_funcs'].shape == (n, 13) and geom['X'].shape == (n, 2)
    assert geom['torsion_max'] > 0
    assert geom['scale'] > 0
    s = [x for x in data['samples'] if x['geom_id'] == 0]
    # Converter sorts modes by frequency: mode 1 = 4 GHz, mode 2 = 5 GHz
    assert [x['mode_idx'] for x in s] == [1, 2]
    assert [float(x['Theta'][0]) for x in s] == [0.0, 1.0]   # slot index
    np.testing.assert_allclose([x['Theta'][1] for x in s], [4.0, 5.0])
    ramp = np.arange(1, n + 1, dtype=np.float64)
    for x in s:
        assert x['Y'].shape == (n, 1)                          # vertex (P1) part of the P2 vector
        assert np.isclose(np.abs(x['Y']).max(), 1.0)
        expected = ramp ** (x['mode_idx'] + 1.0)               # field moved together with its freq
        np.testing.assert_allclose(x['Y'][:, 0], expected / expected.max(), rtol=1e-5)


def test_convert_dataset_clear_errors(tmp_path):
    empty = tmp_path / 'empty.h5'
    with h5py.File(empty, 'w') as f:
        f.create_group('geometry_pool')  # e.g. a converted file passed by mistake
    with pytest.raises(ValueError, match="No raw FEM sample groups"):
        RFCavityToGNOT(str(empty)).convert_dataset(str(tmp_path / 'o.pkl'))

    h5 = tmp_path / 'raw.h5'
    _write_raw_h5(h5)
    with pytest.raises(ValueError, match="No samples produced"):
        RFCavityToGNOT(str(h5)).convert_dataset(str(tmp_path / 'o.pkl'), mode_indices=[5])


def test_torsion_feature_matches_disk_solution(converter):
    """Unit disk: w = (1 − r²)/4 ⇒ max w = 1/4 and j₀₁²/(4·max w) = λ₁ exactly."""
    skfem = pytest.importorskip("skfem")
    m = skfem.MeshTri.init_circle(4)
    out = converter.extract_geometry_features(m.p.T.copy(), m.t.T.copy())
    r2 = (out['X'] ** 2).sum(axis=1)                       # normalised coords
    w = out['Input_funcs'][:, 12] * out['torsion_max']
    np.testing.assert_allclose(w, np.clip((1 - r2) / 4, 0, None), atol=5e-3)
    assert abs(out['torsion_max'] - 0.25) < 5e-3
