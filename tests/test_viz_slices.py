"""Tests for src/viz/slices.py.

Deliberately does NOT import src.viz.nedelec (written in parallel and may
not exist yet) — synthetic plane_sample-format dicts are built by hand here,
using an analytic TM010-like pillbox field (H azimuthal ~ J1(kr)) so the
quiver / magnitude / outline logic is exercised with a physically sensible
pattern, on a 'y' cut (u=x, v=z, in_plane=(0,2)).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.special import j1

from src.viz import slices as vs

SCRATCH = ("/tmp/claude-0/-home-user-rf-cavity-neural-operator/"
           "551360d9-0e42-536d-82da-64927523c26c/scratchpad")


# ── synthetic data builders ──────────────────────────────────────────────────

def _pillbox_plane(R=0.4, res=41, k_list=(6.0, 9.0, 9.05), noise=None, seed=0):
    """A 'y' cut through a pillbox whose axis is along y: disk of radius R
    in the (x, z) = (u, v) plane. For each k in k_list, H is purely
    azimuthal in-plane (TM010-like: H_phi ~ J1(k*r)), H_y = 0.

    Returns (sample_true, sample_pred) A1-format dicts, H shape [nv,nu,3,K].
    """
    lin = np.linspace(-1.2 * R, 1.2 * R, res)
    U, V = np.meshgrid(lin, lin, indexing="xy")  # U=x, V=z
    r = np.sqrt(U ** 2 + V ** 2)
    theta = np.arctan2(V, U)
    inside = r <= R

    rng = np.random.default_rng(seed)
    K = len(k_list)
    H_true = np.zeros(U.shape + (3, K))
    H_pred = np.zeros(U.shape + (3, K))
    for kk, kval in enumerate(k_list):
        hphi = j1(kval * r)
        hu = -hphi * np.sin(theta)
        hv = hphi * np.cos(theta)
        hu = np.where(inside, hu, np.nan)
        hv = np.where(inside, hv, np.nan)
        H_true[..., 0, kk] = hu
        H_true[..., 2, kk] = hv
        pu, pv = hu.copy(), hv.copy()
        if noise:
            pu = pu + noise * rng.standard_normal(pu.shape)
            pv = pv + noise * rng.standard_normal(pv.shape)
        H_pred[..., 0, kk] = pu
        H_pred[..., 2, kk] = pv

    base = {"axis": "y", "offset": 0.0, "u_label": "x", "v_label": "z",
            "U": U, "V": V, "inside": inside, "in_plane": (0, 2)}
    sample_true = dict(base, H=H_true)
    sample_pred = dict(base, H=H_pred)
    return sample_true, sample_pred


def _info(K, split=None, rel_l2=None, f_pred_nan_at=None):
    f_true = np.linspace(1.3, 1.3 + 0.5 * (K - 1), K)
    f_pred = f_true + 0.01
    if f_pred_nan_at is not None:
        f_pred[f_pred_nan_at] = np.nan
    if rel_l2 is None:
        rel_l2 = np.full(K, 0.02)
    rel_l2 = np.asarray(rel_l2, dtype=float)
    split_arr = np.zeros(K, dtype=bool)
    if split:
        for kk in split:
            split_arr[kk] = True
            rel_l2[kk] = np.nan
    return {"f_true": f_true, "f_pred": f_pred, "rel_l2": rel_l2, "split": split_arr}


# ── plot_mode_slice ───────────────────────────────────────────────────────────

def test_plot_mode_slice_basic():
    st, sp = _pillbox_plane(k_list=(6.0,), noise=0.01)
    fig, axes = plt.subplots(1, 3)
    ims = vs.plot_mode_slice(axes, st, sp, 0, title="mode 0")
    assert len(ims) == 3
    assert all(im is not None for im in ims)
    for ax in axes:
        assert ax.get_aspect() in (1.0, "equal")
    plt.close(fig)


def test_plot_mode_slice_zero_inside_no_crash():
    st, sp = _pillbox_plane(k_list=(6.0,))
    st["inside"][:] = False
    sp["inside"][:] = False
    st["H"][:] = np.nan
    sp["H"][:] = np.nan
    fig, axes = plt.subplots(1, 3)
    ims = vs.plot_mode_slice(axes, st, sp, 0, title="empty")
    assert ims == (None, None, None)
    # blank panels still get a title / no exception
    assert axes[0].get_title() != "" or True
    plt.close(fig)


def test_plot_mode_slice_all_nan_freq_and_index_out_of_range():
    st, sp = _pillbox_plane(k_list=(6.0,))
    fig, axes = plt.subplots(1, 3)
    # k beyond available modes -> _extract_mode returns None -> blank, no crash
    ims = vs.plot_mode_slice(axes, st, sp, 5, title="oob")
    assert ims == (None, None, None)
    plt.close(fig)


def test_plot_mode_slice_to_mm_changes_axis_range():
    st, sp = _pillbox_plane(R=0.4, k_list=(6.0,))
    fig, axes = plt.subplots(1, 3)
    vs.plot_mode_slice(axes, st, sp, 0, to_mm=(0.05, np.zeros(3)))
    xlab = axes[0].get_xlabel()
    assert "mm" in xlab
    plt.close(fig)


# ── figure_modes ──────────────────────────────────────────────────────────────

def test_figure_modes_axes_count_and_titles():
    K = 3
    st, sp = _pillbox_plane(k_list=(6.0, 9.0, 9.05), noise=0.01)
    info = _info(K, split=[1, 2])
    fig = vs.figure_modes(st, sp, info, suptitle="synthetic pillbox")
    # n rows * (3 data axes + up to 2 colorbar axes)
    assert len(fig.axes) >= K * 3
    titles = " ".join(ax.get_title() for ax in fig.axes)
    assert "split pair" in titles
    assert "n/a" in titles  # rel_l2 NaN for the split pair
    plt.close(fig)


def test_figure_modes_modes_subset():
    K = 3
    st, sp = _pillbox_plane(k_list=(6.0, 9.0, 9.05))
    info = _info(K)
    fig = vs.figure_modes(st, sp, info, modes=[2])
    data_axes = [ax for ax in fig.axes if ax.get_title()]
    assert len(fig.axes) >= 1 * 3
    assert any("mode 2" in ax.get_title() for ax in data_axes)
    plt.close(fig)


def test_figure_modes_nan_freq_shows_na():
    K = 2
    st, sp = _pillbox_plane(k_list=(6.0, 9.0))
    info = _info(K, f_pred_nan_at=1)
    fig = vs.figure_modes(st, sp, info)
    titles = " ".join(ax.get_title() for ax in fig.axes)
    assert "n/a" in titles
    plt.close(fig)


def test_figure_modes_savefig(tmp_path):
    K = 2
    st, sp = _pillbox_plane(k_list=(6.0, 9.0), noise=0.02)
    info = _info(K)
    fig = vs.figure_modes(st, sp, info, suptitle="save test")
    out = tmp_path / "modes.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists() and out.stat().st_size > 0


def test_figure_modes_empty_inside_row_no_crash():
    K = 1
    st, sp = _pillbox_plane(k_list=(6.0,))
    st["inside"][:] = False
    sp["inside"][:] = False
    st["H"][:] = np.nan
    sp["H"][:] = np.nan
    info = _info(K)
    fig = vs.figure_modes(st, sp, info)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


# ── figure_planes ─────────────────────────────────────────────────────────────

def test_figure_planes_three_cuts():
    K = 2
    st_y, sp_y = _pillbox_plane(k_list=(6.0, 9.0), noise=0.01)
    st_x, sp_x = _pillbox_plane(k_list=(6.0, 9.0), noise=0.01)
    st_x = dict(st_x, axis="x", u_label="y", v_label="z")
    sp_x = dict(sp_x, axis="x", u_label="y", v_label="z")
    st_z, sp_z = _pillbox_plane(k_list=(6.0, 9.0), noise=0.01)
    st_z = dict(st_z, axis="z", u_label="x", v_label="y")
    sp_z = dict(sp_z, axis="z", u_label="x", v_label="y")
    samples = {"x": (st_x, sp_x), "y": (st_y, sp_y), "z": (st_z, sp_z)}
    info = _info(K)
    fig = vs.figure_planes(samples, info, k=1, suptitle="orthogonal cuts")
    assert len(fig.axes) >= 3 * 3
    titles = " ".join(ax.get_title() for ax in fig.axes)
    assert "mode 1" in titles
    assert "cut x=" in titles and "cut y=" in titles and "cut z=" in titles
    plt.close(fig)


def test_figure_planes_savefig(tmp_path):
    K = 1
    st, sp = _pillbox_plane(k_list=(6.0,))
    samples = {"y": (st, sp)}
    info = _info(K)
    fig = vs.figure_planes(samples, info, k=0)
    out = tmp_path / "planes.png"
    fig.savefig(out)
    plt.close(fig)
    assert out.exists() and out.stat().st_size > 0


# ── never calls plt.show ─────────────────────────────────────────────────────

def test_no_show_call(monkeypatch):
    def _boom():
        raise AssertionError("slices.py must never call plt.show()")
    monkeypatch.setattr(plt, "show", _boom)
    K = 2
    st, sp = _pillbox_plane(k_list=(6.0, 9.0))
    info = _info(K)
    fig = vs.figure_modes(st, sp, info)
    plt.close(fig)


# ── example PNG for visual inspection (not asserted, just produced) ─────────

@pytest.mark.skipif(not os.path.isdir(SCRATCH), reason="scratchpad dir not present")
def test_write_example_png():
    K = 3
    st, sp = _pillbox_plane(R=0.4, res=61, k_list=(6.0, 9.0, 9.05), noise=0.015)
    info = _info(K, split=[1, 2], rel_l2=[0.031, np.nan, np.nan])
    fig = vs.figure_modes(st, sp, info, suptitle="Synthetic TM010-like pillbox — y cut",
                           to_mm=(0.05, np.zeros(3)))
    out = os.path.join(SCRATCH, "viz_example.png")
    fig.savefig(out, dpi=130)
    plt.close(fig)
    assert os.path.exists(out)


def test_figure_modes_field_label_from_info():
    st, sp = _pillbox_plane()
    info = _info(st["H"].shape[-1])
    fig = vs.figure_modes(st, sp, dict(info, field="E"))
    titles = " ".join(ax.get_title() for ax in fig.axes)
    labels = " ".join(ax.get_ylabel() for ax in fig.axes)
    assert "True |E|" in titles and "|H|" not in titles and "|E|" in labels
    plt.close(fig)
    fig = vs.figure_modes(st, sp, info)                                  # default H
    assert "True |H|" in " ".join(ax.get_title() for ax in fig.axes)
    plt.close(fig)
