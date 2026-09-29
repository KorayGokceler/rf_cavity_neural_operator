"""Load a checkpoint / dataset and produce aligned per-geometry predictions
(A2, docs/viz3d_contract.md).  Mirrors `scripts/eval_3d.py`'s loading and
metrics exactly, so `predict()`'s `rel_l2` matches `evaluate()`'s `rel_l2_k`
columns for the same (checkpoint, data_path, split, idx).

Public API (contract):
    load(checkpoint, data_path, split='test', device='cpu') -> (lm, ds)
    predict(lm, ds, idx, device='cpu') -> dict
    pick(ds, n=3, by='shape_type') -> list[int]
"""
import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data.dataset_3d import Maxwell3DDataset, maxwell3d_collate   # noqa: E402
from src.models.hcurl import mode_rel_l2                              # noqa: E402
from src.training.lightning_module import GNOTLightning                # noqa: E402
from infer import resolve_checkpoint                                    # noqa: E402


def load(checkpoint, data_path, split='test', device='cpu'):
    """(lm, ds) — same loading as scripts/eval_3d.py main(): resolve_checkpoint,
    hparams['data_cfg'] for the split ratios / seed / feature_indices, ds.stats
    handed to lm.freq_stats, lm.eval()."""
    ckpt = resolve_checkpoint(checkpoint)
    lm = GNOTLightning.load_from_checkpoint(ckpt, map_location=device)
    dc = dict(lm.hparams.get('data_cfg') or {})
    kw = dict(random_seed=dc.get('random_seed', 42), feature_indices=dc.get('feature_indices'))
    if split == 'all':
        ds = Maxwell3DDataset(data_path, split='test', train_ratio=0.0, val_ratio=0.0, **kw)
    else:
        ds = Maxwell3DDataset(data_path, split=split, train_ratio=dc.get('train_ratio', 0.8),
                              val_ratio=dc.get('val_ratio', 0.1), **kw)
    trained = (lm.hparams.get('data_cfg') or {}).get('field')     # recorded since the E switch
    if trained and getattr(ds, 'field', trained) != trained:
        raise ValueError(f"checkpoint trained on field {trained!r}, data is {ds.field!r}")
    lm.freq_stats = ds.stats
    lm.eval().to(device)
    return lm, ds


def _to(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _geom_of(ds, g_id):
    """geometry_pool entry of geometry g_id (same lookup as Maxwell3DDataset.__getitem__)."""
    s_idx = ds.geom_to_samples[g_id]
    g_key = ds.samples_metadata[s_idx[0]]['geom_id']
    return ds.geometry_pool[g_key]


def _align_modes(F, T, MT, clusters):
    """Contract alignment (A2): (pred [Ne,K] double).

    F, T, MT: torch double tensors [Ne,K] (F = Ritz field, M-orthonormal;
    T = target DOFs, unit M-norm; MT = M @ T). clusters: list of index lists.
    Isolated mode k (singleton cluster) -> sign(f_kᵀM t_k)·f_k (unit M-norm).
    Mode k inside a cluster C -> Π_F t_k = Σ_{i∈C} f_i (f_iᵀM t_k), rescaled
    to ‖t_k‖_M (Π_F is an exact M-orthogonal projector since F is
    M-orthonormal, so this is the closest combination of the predicted
    cluster basis to t_k, up to the norm-matching rescale).
    """
    F, T, MT = F.double(), T.double(), MT.double()
    C = F.T @ MT                                            # [K,K] f_iᵀ M t_j
    tt = (T * MT).sum(0).clamp(min=1e-300)                  # ‖t_k‖_M^2
    pred = torch.zeros_like(F)
    for cl in clusters:
        idx = torch.as_tensor(list(cl), dtype=torch.long, device=F.device)
        if len(cl) == 1:
            k = idx[0]
            sign = torch.sign(C[k, k])
            sign = torch.where(sign == 0, torch.ones_like(sign), sign)
            pred[:, k] = sign * F[:, k]
        else:
            Cc = C[idx][:, idx]                             # [n,n]: rows=basis i, cols=target k
            proj = F[:, idx] @ Cc                            # [Ne,n] = Π_F t_k per column k
            cap = (Cc ** 2).sum(0).clamp(min=1e-300)          # ‖Π_F t_k‖_M^2
            scale = (tt[idx] / cap).sqrt()
            pred[:, idx] = proj * scale.unsqueeze(0)
    return pred


@torch.no_grad()
def predict(lm, ds, idx, device='cpu'):
    """One geometry's aligned prediction (A2 output dict, numpy float64
    unless noted). See docs/viz3d_contract.md section A2 for the exact keys
    and the alignment rule; plus 'field' ('H' | 'E', from the PKL metadata):
    the physical field whose N0 DOFs 'true' / 'pred' / 'err' are."""
    lm.eval().to(device)
    item = ds[idx]
    batch = _to(maxwell3d_collate([item]), device)
    out = lm(batch)

    fs = lm.freq_stats
    ghz = (lambda z: z * fs['std'] + fs['mean']) if fs else (lambda z: z)   # noqa: E731

    Ne = int(batch['EdgeMask'][0].sum())
    K = out['field'].shape[-1]
    F = out['field'][0, :Ne].double().cpu()                 # [Ne,K] Ritz field
    T = batch['Y_field'][0, :Ne].double().cpu()[:, :K]       # [Ne,K] target DOFs (unit M-norm)

    M = item['M']                                            # scipy CSR float64, unpadded
    MT = torch.from_numpy(np.asarray(M @ T.numpy(), dtype=np.float64))

    inside, split = lm._clusters_3d(batch, 0, K)
    rl = mode_rel_l2(F, T, MT, inside)
    rl[split] = float('nan')

    split_mask = np.zeros(K, dtype=bool)
    split_mask[split] = True
    clusters_for_align = inside + [[k] for k in split]
    pred = _align_modes(F, T, MT, clusters_for_align)

    f_pred = ghz(out['freq'][0, :K]).double().cpu().numpy()
    f_true = ghz(batch['Y_freq'][0, :K]).double().cpu().numpy()

    g_id = ds.active_geoms[idx]
    geom = _geom_of(ds, g_id)

    true_np, pred_np = T.numpy(), pred.numpy()
    return {
        'field': str(getattr(ds, 'field', 'H')),              # 'H' | 'E': what the DOFs are
        'geom_id': int(g_id),
        'shape_type': str(item['shape_type']),
        'X': np.asarray(item['X'].numpy(), dtype=np.float64),
        'tets': np.asarray(geom['tets'], dtype=np.int64),
        'edges': np.asarray(geom['edges'], dtype=np.int64),
        'scale': float(geom['scale']),
        'center': np.asarray(geom['center'], dtype=np.float64).reshape(3),
        'true': true_np,
        'pred': pred_np,
        'err': pred_np - true_np,
        'f_true': f_true,
        'f_pred': f_pred,
        'rel_l2': rl.double().cpu().numpy(),
        'split': split_mask,
        'clusters': [[int(k) for k in cl] for cl in inside],
    }


def pick(ds, n=3, by='shape_type'):
    """Deterministic sample indices: one per distinct `by` value (geometry_pool
    attribute, default 'shape_type'), in order of first appearance, then
    filled with the remaining indices (dataset order) up to n."""
    n = min(int(n), len(ds))
    keys = [str(_geom_of(ds, ds.active_geoms[i]).get(by, '')) for i in range(len(ds))]
    chosen, seen = [], set()
    for i, k in enumerate(keys):
        if len(chosen) >= n:
            break
        if k not in seen:
            seen.add(k)
            chosen.append(i)
    if len(chosen) < n:
        chosen_set = set(chosen)
        for i in range(len(ds)):
            if len(chosen) >= n:
                break
            if i not in chosen_set:
                chosen.append(i)
                chosen_set.add(i)
    return chosen[:n]
