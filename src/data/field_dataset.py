"""Items and batches of the learned-field model (option B, docs/28): a curved mesh with its
HCurl(p) operators (src.data_gen.highorder.field_operators) and optional high-order labels.

Item keys (one geometry): the model's tokens are the nq quadrature points of every element —
X [Nt·nq, 3] (normalised, curved geometry), Input_funcs [Nt·nq, F] (highorder.point_features),
Area [Nt·nq] (quadrature weights: the attention's mass weights); QX [Nt, nq, 3], QP [Nt, nl, nq];
scipy M, K [n×n], G [n×p], Kp [p×p], C [n×3·Nt·nl]; BndDof [n] bool; Scale; Y_field [n, K] (DOF
labels, wall rows 0) and Y_freq [K] GHz when labelled.

Batch (field_collate): padded tensors, block-diagonal CSR operators on the padded layout (sample
b's DOF d is row b·n_max + d, its L2 coefficient c column b·L_max + c), DofMask, ElemMask, Kp_diag —
the layout src.models.hcurl expects, with "edges" → DOFs and "vertices" → potentials.

FieldDataset reads the field-label H5 files of dataset_generator_3d --labels field (docs/29): per
geometry the curved mesh is rebuilt from the stored .vol + BREP (+ deformation), the model-order
operators are assembled (field_operators) and the stored projected labels are checked against them
(their Rayleigh quotients must reproduce the generator's rq_model).
"""
import glob
import os
import pickle

import h5py
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.data.dataset_3d import _block_diag, _mask, default_qoi_names


C0 = 299792458.0


def check_labels(ops, Y_field, Y_freq, rtol=1e-8):
    """The labels must be eigenvectors of THIS item's operators: Rayleigh quotients yᵀKy / yᵀMy equal
    the normalised label eigenvalues (2π f s / c)².  A label space that differs only in geometry
    order (mesh.Curve), quadrature, deformation or DOF numbering has the same DOF count, so nothing
    else would notice (curve 2 vs 3 on a pillbox: 1.4e-4)."""
    Y = np.asarray(Y_field, dtype=np.float64).reshape(ops["n_dof"], -1)
    f = np.asarray(Y_freq, dtype=np.float64).reshape(-1)[:Y.shape[1]] * 1e9
    lam = (2 * np.pi * f * float(ops["scale"]) / C0) ** 2
    rq = np.einsum("ik,ik->k", Y, ops["K"] @ Y) / np.einsum("ik,ik->k", Y, ops["M"] @ Y)
    err = np.abs(rq / lam - 1.0)
    if not np.all(err <= rtol):
        raise ValueError(f"labels are not eigenvectors of the item's HCurl(p={ops['order']}) operators "
                         f"(max |yᵀKy/yᵀMy / λ − 1| = {err.max():.2e}): labels and operators must come from "
                         "the same mesh, geometry order, polynomial order and integration rule")
    if np.abs(Y[np.asarray(ops["bnd"], dtype=bool)]).max(initial=0.0) > 1e-12 * np.abs(Y).max():
        raise ValueError("labels are nonzero on PEC wall DOFs")


def field_item(ops, Y_field=None, Y_freq=None, feature_indices=None, shape_type="", check=True):
    """Model item from a field_operators dict (normalised geometry).  Labels (DOF vectors of the
    same HCurl(p) space, frequencies in GHz) are verified against the operators (check_labels)."""
    if Y_field is not None and check:
        check_labels(ops, Y_field, Y_freq)
    nt, nq = ops["qx"].shape[:2]
    feats = np.asarray(ops["qfeat"], dtype=np.float32).reshape(nt * nq, -1)
    if feature_indices is not None:
        feats = feats[:, feature_indices]
    n = ops["n_dof"]
    bnd = np.asarray(ops["bnd"], dtype=bool)
    qx = np.asarray(ops["qx"], dtype=np.float32)
    item = {"X": torch.from_numpy(qx.reshape(-1, 3).copy()),
            "Input_funcs": torch.from_numpy(np.ascontiguousarray(feats)),
            "Area": torch.from_numpy(np.asarray(ops["qw"], dtype=np.float32).reshape(-1)),
            "QX": torch.from_numpy(qx), "QP": torch.from_numpy(np.asarray(ops["P"], dtype=np.float32)),
            "BndDof": torch.from_numpy(bnd), "Scale": torch.tensor(float(ops["scale"]), dtype=torch.float32),
            "shape_type": str(shape_type)}
    for k in ("M", "K", "G", "Kp", "C"):
        item[k] = ops[k]
    if Y_field is not None:
        Y = torch.as_tensor(np.asarray(Y_field), dtype=torch.float32).reshape(n, -1).clone()
        Y[torch.from_numpy(bnd)] = 0.0
        item["Y_field"] = Y
        item["Y_freq"] = torch.as_tensor(np.asarray(Y_freq), dtype=torch.float32).reshape(-1)
    return item


def field_collate(batch):
    """Pad and block-diagonalise a list of field_item dicts (module docstring)."""
    pad = lambda k: pad_sequence([it[k] for it in batch], batch_first=True)   # noqa: E731
    nv = [it["X"].shape[0] for it in batch]
    nt = [it["QX"].shape[0] for it in batch]
    nd = [it["M"].shape[0] for it in batch]
    npot = [it["Kp"].shape[0] for it in batch]
    nl2 = [it["C"].shape[1] for it in batch]
    Nv, Nt, Nd, Np, Nl = max(nv), max(nt), max(nd), max(npot), max(nl2)
    if len({it["QP"].shape[1:] for it in batch}) != 1:
        raise ValueError("field_collate: items of different order / quadrature")
    nq = batch[0]["QX"].shape[1]
    if Nv != Nt * nq:
        raise ValueError("field_collate: tokens must be the Nt·nq quadrature points")
    out = {k: pad(k) for k in ("X", "Input_funcs", "Area", "QX", "QP")}
    out["Mask"], out["ElemMask"], out["DofMask"] = _mask(nv, Nv), _mask(nt, Nt), _mask(nd, Nd)
    out["BndDof"] = pad("BndDof").bool()
    out["Scale"] = torch.stack([it["Scale"] for it in batch])
    out["shape_type"] = [it["shape_type"] for it in batch]
    out["M"] = _block_diag([it["M"] for it in batch], Nd, Nd)
    out["K"] = _block_diag([it["K"] for it in batch], Nd, Nd)
    out["G"] = _block_diag([it["G"] for it in batch], Nd, Np)
    out["Gt"] = _block_diag([it["G"].T for it in batch], Np, Nd)
    out["Kp"] = _block_diag([it["Kp"] for it in batch], Np, Np)
    out["Kp_diag"] = pad_sequence([torch.from_numpy(it["Kp"].diagonal().astype(np.float64)) for it in batch],
                                  batch_first=True)
    # C maps the element-major L2 coefficients (t, c, l); padded elements have QP = 0 → coefficient 0
    out["C"] = _block_diag([it["C"] for it in batch], Nd, Nl)
    if all("Y_field" in it for it in batch):
        out["Y_field"] = pad("Y_field")
        out["Y_freq"] = pad("Y_freq")
    for k in ("FreqNext", "geom_id"):
        if all(k in it for it in batch):
            out[k] = torch.stack([it[k] for it in batch])
    if all("Y_qoi" in it for it in batch):
        out["Y_qoi"] = pad_sequence([it["Y_qoi"] for it in batch], batch_first=True, padding_value=float("nan"))
    return out


def _expand(paths):
    """Sorted H5 files of a path, glob, directory or list of those."""
    if isinstance(paths, (str, os.PathLike)):
        paths = [paths]
    out = []
    for p in paths:
        p = str(p)
        if os.path.isdir(p):
            out += glob.glob(os.path.join(p, "*.h5"))
        else:
            out += glob.glob(p) or ([p] if os.path.exists(p) else [])
    out = sorted(set(out))
    if not out:
        raise FileNotFoundError(f"no field-label H5 files in {paths}")
    return out


class FieldDataset(Dataset):
    """Field-label geometries (docs/29) as field_item dicts plus Y_freq (z-scored with the TRAIN
    split's statistics), FreqNext, geom_id, Y_qoi [K, n_qoi] (the stored p3 QoI, qoi_names order).

    data_path: one H5 file, a glob ("data/field/*.h5"), a directory or a list.  The split is by
    geometry (seeded permutation of the sorted (file, group) index).  cache_dir: keep the rebuilt
    operators on disk (pickle per geometry; ~0.5 GB per 16k-tet geometry at p2) instead of
    reassembling them every epoch (~10 s per geometry at p2)."""

    def __init__(self, data_path, split="train", train_ratio=0.8, val_ratio=0.1, random_seed=42,
                 feature_indices=None, cache_dir=None, n_modes=None, rq_rtol=1e-5, augment=False):
        if augment:
            raise ValueError("FieldDataset: rotation augmentation is not supported (the DOF maps C, QP "
                             "are built for the stored orientation)")
        self.paths = _expand(data_path)
        self.split, self.feature_indices, self.rq_rtol = split, feature_indices, float(rq_rtol)
        self.cache_dir = cache_dir
        index, freqs, nxt, Ks, types = [], [], [], [], []
        self.qoi_names = tuple(default_qoi_names())
        for path in self.paths:
            with h5py.File(path, "r") as f:
                for key in sorted(k for k in f.keys() if k.startswith("sample_")):
                    g = f[key]
                    if str(g.attrs.get("labels", "")) != "field":
                        raise ValueError(f"{path}/{key}: not a field-label sample (dataset_generator_3d "
                                         "--labels field); N0 data goes through convert_3d.py + Maxwell3DDataset")
                    index.append((path, key))
                    freqs.append(np.asarray(g["freqs"], np.float64))
                    nxt.append(float(g.attrs.get("freq_next", np.nan)))
                    Ks.append(len(freqs[-1]))
                    types.append(str(g.attrs.get("shape_type", "")))
        if not index:
            raise ValueError(f"no samples in {self.paths}")
        self.n_modes = int(n_modes or min(Ks))
        self.index, self._freqs, self._next, self.shape_types = index, freqs, nxt, types
        n = len(index)
        perm = np.random.RandomState(random_seed).permutation(n)
        n_tr, n_va = int(n * train_ratio), int(n * val_ratio)
        parts = {"train": perm[:n_tr], "val": perm[n_tr:n_tr + n_va], "test": perm[n_tr + n_va:]}
        self.active_geoms = [int(i) for i in parts.get(split, parts["test"])]
        tr = np.concatenate([freqs[i][:self.n_modes] for i in parts["train"]]) if n_tr else \
            np.concatenate([fq[:self.n_modes] for fq in freqs])
        self.stats = {"mean": float(tr.mean()), "std": float(tr.std() if tr.std() > 0 else 1.0)}
        self.geom_to_samples = {i: [i] for i in range(n)}          # train.py's empty-split message
        self.field = "E"
        print(f"FieldDataset {split}: {len(self.active_geoms)} geometries × {self.n_modes} modes "
              f"({len(self.paths)} files)")

    def __len__(self):
        return len(self.active_geoms)

    def data_dims(self):
        """(val_dim, n_modes)."""
        from src.data.dataset_converter_3d import FEATURE_NAMES_3D
        nf = len(self.feature_indices) if self.feature_indices is not None else len(FEATURE_NAMES_3D)
        return nf, self.n_modes

    def raw_freqs(self, g_id):
        """Stored label frequencies [GHz] of geometry g_id (count_near_degenerate)."""
        return self._freqs[g_id][:self.n_modes]

    def _cache_path(self, g_id):
        if not self.cache_dir:
            return None
        path, key = self.index[g_id]
        tag = os.path.splitext(os.path.basename(path))[0]
        return os.path.join(self.cache_dir, f"{tag}__{key}.ops.pkl")

    def load(self, g_id):
        """(ops, record) of geometry g_id: field_operators of the rebuilt curved mesh + stored labels."""
        from src.data.dataset_converter_3d import normalise
        from src.data_gen.field_labels import read_field_group
        from src.data_gen.highorder import field_operators
        path, key = self.index[g_id]
        cp = self._cache_path(g_id)
        if cp and os.path.exists(cp):
            with open(cp, "rb") as fh:
                return pickle.load(fh)
        with h5py.File(path, "r") as f:
            mesh, rec = read_field_group(f[key])
        Xv = np.array([tuple(p.p) for p in mesh.ngmesh.Points()], dtype=np.float64)
        tets = np.array([[v.nr - 1 for v in el.vertices] for el in mesh.ngmesh.Elements3D()], dtype=np.int64)
        _, center, scale = normalise(Xv, tets)
        ops = field_operators(mesh, order=rec["model_order"], scale=scale, center=center)
        U = rec["u_model"]
        if U.shape[0] != ops["n_dof"]:
            raise ValueError(f"{path}/{key}: {U.shape[0]} label DOFs, the rebuilt HCurl(p={rec['model_order']}) "
                             f"space has {ops['n_dof']}")
        rq = np.einsum("ik,ik->k", U, ops["K"] @ U) / np.einsum("ik,ik->k", U, ops["M"] @ U)
        err = np.abs(rq / (rec["rq_model"] * ops["scale"] ** 2) - 1.0)
        if not np.all(err <= self.rq_rtol):
            raise ValueError(f"{path}/{key}: labels do not match the rebuilt operators (max Rayleigh-quotient "
                             f"mismatch {err.max():.2e}): mesh, curving or deformation differ from the generator's")
        ops = {k: v for k, v in ops.items() if k not in ("bary", "X", "tets")}
        out = (ops, rec)
        if cp:
            os.makedirs(self.cache_dir, exist_ok=True)
            tmp = f"{cp}.{os.getpid()}.tmp"
            with open(tmp, "wb") as fh:
                pickle.dump(out, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, cp)
        return out

    def __getitem__(self, idx):
        g_id = self.active_geoms[idx]
        ops, rec = self.load(g_id)
        K = self.n_modes
        item = field_item(ops, rec["u_model"][:, :K], rec["freqs"][:K], self.feature_indices,
                          rec.get("shape_type", self.shape_types[g_id]), check=False)
        m, sd = self.stats["mean"], self.stats["std"]
        item["Y_freq"] = (item["Y_freq"] - m) / sd
        item["FreqNext"] = torch.tensor((self._next[g_id] - m) / sd, dtype=torch.float32)
        item["geom_id"] = torch.tensor([g_id], dtype=torch.long)
        names = list(rec.get("qoi_names") or [])
        q = np.full((K, len(self.qoi_names)), np.nan, np.float32)
        for j, n in enumerate(self.qoi_names):
            if n in names:
                q[:, j] = np.asarray(rec["qoi"], np.float32)[:K, names.index(n)]
        item["Y_qoi"] = torch.from_numpy(q)
        return item
