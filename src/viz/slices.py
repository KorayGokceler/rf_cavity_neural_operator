"""Matplotlib figures comparing true vs predicted field cross sections.

The field letter (titles, colour-bar labels) is info['field'] ('H' default,
'E' for E-formulation PKLs) or the explicit `field=` argument; the sample
dicts keep the historical key 'H' for the field array whatever it is.

Consumes the plane-sample dicts produced by ``src/viz/nedelec.py`` (A1) and
the per-geometry prediction dicts produced by ``src/viz/predict.py`` (A2),
but never imports either module — everything here is plain numpy/matplotlib
so this file can be developed, tested and used independently.

Plane-sample dict (A1 ``plane_sample``), one per (geometry, cut):
    {'axis': 'x'|'y'|'z', 'offset': float, 'u_label': str, 'v_label': str,
     'U': [nv,nu], 'V': [nv,nu],            # normalised in-plane coords
     'inside': [nv,nu] bool,
     'H': [nv,nu,3,K] (NaN outside),        # all K modes for this geometry
     'in_plane': (iu, iv)}                  # component indices (0,1,2=x,y,z)
       in U, V for this cut

Prediction-info dict (A2 ``predict``, the fields this module reads):
    {'f_true': [K] GHz, 'f_pred': [K] GHz, 'rel_l2': [K] (NaN for split
     modes), 'split': [K] bool}

Agg-safe: never calls plt.show(); all functions return Figure objects for
the caller to save/close.
"""
from __future__ import annotations

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import patheffects as _pe

matplotlib.use("Agg", force=False)  # no-op if a GUI backend is already set

FIELD_CMAP = "viridis"
ERROR_CMAP = "magma"
_AXIS_IDX = {"x": 0, "y": 1, "z": 2}


def _col_titles(field="H"):
    return (f"True |{field}|", f"Predicted |{field}|", rf"|${field}_{{pred}}-{field}_{{true}}$|")


_COL_TITLES = _col_titles("H")


def _field_of(info, field=None):
    f = field if field is not None else (info or {}).get("field", "H")
    return str(f or "H")


# ── small numeric helpers ────────────────────────────────────────────────────

def _extract_mode(sample, k):
    """sample['H'][..., :, k] -> [nv,nu,3], or None if unavailable."""
    if sample is None:
        return None
    H = sample.get("H")
    if H is None:
        return None
    H = np.asarray(H)
    if H.ndim == 4:
        if k < 0 or k >= H.shape[-1]:
            return None
        return H[..., :, k]
    if H.ndim == 3:  # already a single mode
        return H
    return None


def _magnitude(H):
    """|H| = sqrt(sum of the 3 components squared); NaN stays NaN."""
    if H is None:
        return None
    return np.sqrt(np.sum(np.asarray(H) ** 2, axis=-1))


def _finite_max(mag, inside):
    if mag is None or inside is None:
        return None
    inside = np.asarray(inside, dtype=bool)
    if not np.any(inside):
        return None
    vals = np.asarray(mag)[inside]
    vals = vals[np.isfinite(vals)]
    return float(vals.max()) if vals.size else None


def _fmt(x, fmt="{:.4f}"):
    x = float(x) if x is not None else float("nan")
    return "n/a" if not np.isfinite(x) else fmt.format(x)


def _mode_row_text(k, info):
    """'mode k  f_true=… GHz  f_pred=… GHz  relL2=…', with '(split pair)'."""
    f_true = np.asarray(info["f_true"])
    f_pred = np.asarray(info["f_pred"])
    rel_l2 = np.asarray(info["rel_l2"])
    K = f_true.shape[0]
    split = np.asarray(info.get("split", np.zeros(K, dtype=bool)))
    txt = (f"mode {k}  f_true={_fmt(f_true[k])} GHz  "
           f"f_pred={_fmt(f_pred[k])} GHz  relL2={_fmt(rel_l2[k])}")
    if bool(split[k]):
        txt += "  (split pair)"
    return txt


def _to_mm(coord, label, to_mm):
    """(coord*scale + center[axis_index]) * 1e3, or `coord` unchanged."""
    if to_mm is None:
        return coord
    scale, center = to_mm
    idx = _AXIS_IDX[label]
    return (np.asarray(coord) * scale + np.asarray(center)[idx]) * 1e3


def _axis_label(label, to_mm):
    return f"{label} [mm]" if to_mm is not None else f"{label} (normalised)"


# ── drawing helpers ──────────────────────────────────────────────────────────

def _blank_axis(ax, title, msg="no cut\n(empty inside mask)"):
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes,
             fontsize=10, color="0.5")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=9)
    return None


def _outline(ax, Ug, Vg, inside):
    inside = np.asarray(inside, dtype=bool)
    if inside.size == 0 or inside.all() or not inside.any():
        return
    try:
        ax.contour(Ug, Vg, inside.astype(float), levels=[0.5],
                   colors="black", linewidths=1.2, alpha=0.8)
    except Exception:
        pass  # e.g. degenerate/constant mask on a tiny synthetic grid


def _quiver(ax, Ug, Vg, H, inside, in_plane, vmax, n_per_side=15, color="white", min_frac=0.1):
    """In-plane arrows, length ∝ in-plane |H| / vmax (the row's colour scale), so a
    field pointing out of the cut shows no arrows; below min_frac·vmax dropped."""
    if H is None or in_plane is None:
        return
    inside = np.asarray(inside, dtype=bool)
    nv, nu = inside.shape
    sv = max(1, nv // n_per_side)
    su = max(1, nu // n_per_side)
    iu, iv = in_plane
    Uq = np.asarray(Ug)[::sv, ::su]
    Vq = np.asarray(Vg)[::sv, ::su]
    insq = inside[::sv, ::su]
    Hu = np.asarray(H)[::sv, ::su, iu]
    Hv = np.asarray(H)[::sv, ::su, iv]
    mag = np.sqrt(Hu ** 2 + Hv ** 2)
    valid = insq & np.isfinite(mag) & (mag > min_frac * vmax)
    if not np.any(valid):
        return
    un = Hu[valid] / vmax
    vn = Hv[valid] / vmax
    q = ax.quiver(Uq[valid], Vq[valid], un, vn, color=color, alpha=0.85,
                  pivot="mid", width=0.0045, scale=12, scale_units="width", minlength=0)
    try:
        q.set_path_effects([_pe.withStroke(linewidth=1.4, foreground="black",
                                            alpha=0.6)])
    except Exception:
        pass


# ── public API ───────────────────────────────────────────────────────────────

def plot_mode_slice(axes, sample_true, sample_pred, k, title="", vmax=None,
                     to_mm=None, arrows=True, field="H"):
    """Draw one mode's true | pred | |error| on 3 given Axes.

    Uses `sample_true`'s grid ('U','V','inside') for all three panels — the
    true and predicted DOFs share the same mesh (A2), so their plane samples
    share the same grid; only the field values differ.

    Returns (im_true, im_pred, im_err), each a QuadMesh or None (panel left
    blank because the inside mask was empty / all-NaN).  `field` only labels.
    """
    ax_t, ax_p, ax_e = axes
    col_titles = _col_titles(field)

    Ht = _extract_mode(sample_true, k)
    Hp = _extract_mode(sample_pred, k)
    mag_t = _magnitude(Ht)
    mag_p = _magnitude(Hp)
    inside = sample_true.get("inside") if sample_true is not None else None
    inside = np.asarray(inside, dtype=bool) if inside is not None else None

    vmax_row = vmax
    if vmax_row is None:
        cands = [m for m in (_finite_max(mag_t, inside), _finite_max(mag_p, inside))
                 if m is not None and m > 0]
        vmax_row = max(cands) if cands else None

    if vmax_row is None or inside is None:
        _blank_axis(ax_t, f"{title}\n{col_titles[0]}" if title else col_titles[0])
        _blank_axis(ax_p, col_titles[1])
        _blank_axis(ax_e, col_titles[2])
        return (None, None, None)

    u_label = sample_true.get("u_label", "u")
    v_label = sample_true.get("v_label", "v")
    Ug = _to_mm(sample_true["U"], u_label, to_mm)
    Vg = _to_mm(sample_true["V"], v_label, to_mm)
    in_plane = sample_true.get("in_plane")

    mag_e = _magnitude(Hp - Ht) if (Ht is not None and Hp is not None
                                     and np.shape(Ht) == np.shape(Hp)) else None

    ims = [None, None, None]
    panels = ((ax_t, mag_t, FIELD_CMAP, Ht), (ax_p, mag_p, FIELD_CMAP, Hp),
              (ax_e, mag_e, ERROR_CMAP, None))
    for i, (ax, mag, cmap, Hfield) in enumerate(panels):
        col_title = f"{title}\n{col_titles[i]}" if (title and i == 0) else col_titles[i]
        if mag is None or _finite_max(mag, inside) is None:
            _blank_axis(ax, col_title)
            continue
        im = ax.pcolormesh(Ug, Vg, mag, shading="auto", cmap=cmap,
                            vmin=0.0, vmax=vmax_row)
        ims[i] = im
        _outline(ax, Ug, Vg, inside)
        if arrows and Hfield is not None:
            _quiver(ax, Ug, Vg, Hfield, inside, in_plane, vmax_row)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(_axis_label(u_label, to_mm))
        if i == 0:   # shared by the row; the colorbar labels sit where later ones would
            ax.set_ylabel(_axis_label(v_label, to_mm))
        ax.set_title(col_title, fontsize=9)

    return tuple(ims)


def _add_row_colorbars(fig, axes_row, ims, field="H"):
    """One colorbar for the (shared-scale) field panels + one for the error
    panel (different colormap, same vmin/vmax so magnitudes stay comparable).
    """
    ax_t, ax_p, ax_e = axes_row
    im_field = ims[0] if ims[0] is not None else ims[1]
    if im_field is not None:
        cb = fig.colorbar(im_field, ax=[ax_t, ax_p], shrink=0.85, pad=0.02, aspect=28)
        cb.set_label(f"|{field}|  [a.u.]")
    if ims[2] is not None:
        cb2 = fig.colorbar(ims[2], ax=ax_e, shrink=0.85, pad=0.02, aspect=28)
        cb2.set_label(rf"|$\Delta {field}$|  [a.u.]")


def _row_height(sample, width=13.5):
    """Row height [in] matching the cut's aspect ratio (flat cuts → short rows)."""
    try:
        U, V = np.asarray(sample["U"]), np.asarray(sample["V"])
        r = np.ptp(V) / max(np.ptp(U), 1e-12)
    except (KeyError, TypeError, ValueError):
        r = 1.0
    return float(np.clip(width / 3.6 * r + 1.0, 2.2, 4.8))


def figure_modes(samples_true, samples_pred, info, modes=None, suptitle="",
                  to_mm=None, arrows=True, field=None):
    """One row per mode: True |F| | Predicted |F| | |F_pred - F_true|, F = field
    (default info.get('field', 'H')).

    samples_true / samples_pred: single A1 ``plane_sample`` dicts holding all
    K modes for one geometry (['H'] has shape [nv,nu,3,K]).
    info: {'f_true','f_pred','rel_l2','split'} arrays of length K (A2 dict).
    modes: iterable of mode indices to plot; default = all K modes.
    """
    H = samples_true.get("H") if samples_true is not None else None
    K = int(np.asarray(H).shape[-1]) if H is not None else len(np.asarray(info["f_true"]))
    if modes is None:
        modes = list(range(K))
    else:
        modes = list(modes)
    n = max(1, len(modes))
    field = _field_of(info, field)

    fig, axes = plt.subplots(n, 3, figsize=(13.5, _row_height(samples_true) * n), squeeze=False,
                              constrained_layout=True)
    for i, k in enumerate(modes):
        row_title = _mode_row_text(k, info)
        ims = plot_mode_slice(axes[i], samples_true, samples_pred, k,
                               title=row_title, to_mm=to_mm, arrows=arrows, field=field)
        _add_row_colorbars(fig, axes[i], ims, field)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    return fig


def figure_planes(samples, info, k, to_mm=None, arrows=True, suptitle="", field=None):
    """The same mode `k` on up to three orthogonal cuts.

    samples: dict axis -> (sample_true, sample_pred), e.g.
        {'x': (st_x, sp_x), 'y': (st_y, sp_y), 'z': (st_z, sp_z)}
    Rows = cuts (in the order `samples` is iterated), columns = true/pred/error.
    """
    axis_keys = list(samples.keys())
    n = max(1, len(axis_keys))
    hs = [_row_height(samples[a][0]) for a in axis_keys]
    fig, axes = plt.subplots(n, 3, figsize=(13.5, sum(hs)), squeeze=False,
                              constrained_layout=True, gridspec_kw={"height_ratios": hs})
    row_title = _mode_row_text(k, info)
    field = _field_of(info, field)

    for i, axis_key in enumerate(axis_keys):
        s_true, s_pred = samples[axis_key]
        offset = s_true.get("offset", float("nan")) if s_true is not None else float("nan")
        title = f"{row_title}  cut {axis_key}={offset:.3g}" if i == 0 else \
            f"cut {axis_key}={offset:.3g}"
        ims = plot_mode_slice(axes[i], s_true, s_pred, k, title=title,
                               to_mm=to_mm, arrows=arrows, field=field)
        _add_row_colorbars(fig, axes[i], ims, field)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    return fig
