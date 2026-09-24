import torch
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from torch.utils.data import DataLoader

from src.data.dataset import GNOTDataset, gnot_collate_fn
from src.training.lightning_module import GNOTLightning, detect_clusters
from src.fem_refine import refine_modes


# ── Subspace postprocessing ──────────────────────────────────────────────────

def _near_degenerate_clusters(freqs, rel_threshold=0.05):
    """Return clusters of mode indices grouped by frequency proximity.

    freqs: 1-D array of ascending frequencies.
    Returns e.g. [[0], [1, 2]] for a near-degenerate pair at indices 1 & 2.
    """
    K = len(freqs)
    f_mean = float(np.mean(np.abs(freqs))) if K else 0.0
    denom = max(abs(f_mean), 1e-6)
    clusters = [[0]]
    for i in range(1, K):
        if abs(freqs[i] - freqs[i - 1]) / denom < rel_threshold:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    return clusters


def _near_degenerate_note(freqs, rel_threshold=0.05):
    """Return a human-readable note about near-degenerate frequency pairs.

    freqs: 1-D array of (ascending) frequencies in physical units.
    """
    clusters = _near_degenerate_clusters(freqs, rel_threshold) if len(freqs) else []
    pairs = [(a, b) for cl in clusters for a, b in zip(cl[:-1], cl[1:])]
    if not pairs:
        return "well-separated modes"
    return "near-degenerate: " + ", ".join(f"modes {a}&{b}" for a, b in pairs)


def _orthonormalize_np(E):
    """QR-orthonormalize columns of E (numpy, [N, n]).  Returns Q [N, n]."""
    Q, R = np.linalg.qr(E, mode='reduced')
    # Fix sign so diagonal of R is positive
    signs = np.sign(np.diag(R))
    signs[signs == 0] = 1.0
    return Q * signs[np.newaxis, :]


def _subspace_project(E_hat_orth, e_tgt):
    """Project e_tgt onto span(E_hat_orth) — closest vector in the subspace.

    E_hat_orth: [N, n] orthonormal basis (cols) of predicted subspace.
    e_tgt:      [N]    target eigenvector.
    Returns the projected vector [N].
    """
    # proj = Q Q^T e_tgt
    coeffs = E_hat_orth.T @ e_tgt        # [n]
    return E_hat_orth @ coeffs            # [N]


def _subspace_rel_l2(E_hat_orth, e_tgt):
    """Subspace projection error: ||e_tgt - proj(e_tgt, E_hat)||₂ / ||e_tgt||₂."""
    proj = _subspace_project(E_hat_orth, e_tgt)
    residual = e_tgt - proj
    return float(np.linalg.norm(residual) / (np.linalg.norm(e_tgt) + 1e-8))


def _ot_match_np(fp_norm, ft_norm, E_hat, E_tgt, freq_w=0.5):
    """Hungarian OT matching of predicted slots to target modes (numpy).

    Identical logic to lightning_module.ot_match but operates on numpy arrays.
    Returns perm[K] such that E_hat[:, perm] is optimally aligned to E_tgt,
    i.e. E_hat[:, perm[j]] is the best prediction for target mode j.

    fp_norm / ft_norm: z-scored (normalized) frequencies — same scale used
    during training so the freq_w coefficient is directly comparable.
    """
    from scipy.optimize import linear_sum_assignment
    K = len(ft_norm)
    df = fp_norm[:, None] - ft_norm[None, :]       # [Kp, Kt]
    cost_f = df ** 2
    eh2 = (E_hat ** 2).sum(0)[:, None]             # [Kp, 1]
    et2 = (E_tgt ** 2).sum(0)[None, :]             # [1, Kt]
    cross = E_hat.T @ E_tgt                        # [Kp, Kt]
    num_p = eh2 + et2 - 2.0 * cross               # ||eh_i - et_j||^2
    num_n = eh2 + et2 + 2.0 * cross               # ||eh_i + et_j||^2
    cost_field = np.minimum(num_p, num_n) / (et2 + 1e-8)
    C = freq_w * cost_f + cost_field
    row, col = linear_sum_assignment(C)
    perm = np.empty(K, dtype=np.int64)
    perm[col] = row                                # perm[true_j] = pred_slot
    return perm


# ── Checkpoint / dataset helpers (shared with scripts/infer_val_all.py) ─────

def resolve_checkpoint(path):
    """Accept a .ckpt file or a training dir (training_logs/<exp_name>).

    For a directory the best checkpoint (lowest val_rel_l2 in the file name,
    as written by train.py's ModelCheckpoint) is used, else last.ckpt.
    """
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    import glob
    import re
    ckpts = glob.glob(os.path.join(path, '**', '*.ckpt'), recursive=True)
    if not ckpts:
        raise FileNotFoundError(f"No .ckpt files under {path}")

    def _score(p):
        m = re.search(r'val_rel_l2=([0-9.]+?)(?:-v\d+)?\.ckpt$', os.path.basename(p))
        return float(m.group(1)) if m else float('inf')
    best = min(ckpts, key=lambda p: (_score(p), os.path.basename(p) != 'last.ckpt'))
    return best


def load_model(checkpoint):
    """Load GNOTLightning on CPU/GPU in eval mode."""
    ckpt = resolve_checkpoint(checkpoint)
    print(f"Loading checkpoint from: {ckpt}")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GNOTLightning.load_from_checkpoint(ckpt, map_location=device)
    model.eval().to(device)
    return model, device


def build_dataset(model, data_path, split, **overrides):
    """GNOTDataset with the SAME split / input features as at train time.

    train.py stores the dataset settings in hparams['data_cfg']; the default
    ratios/seed of GNOTDataset silently gave a different val split and wrong
    val_dim for ablation (feature_indices) checkpoints.  Node sub-sampling
    (max_nodes) is intentionally NOT reused: plots need the full mesh.
    """
    dcfg = dict(model.hparams.get('data_cfg') or {})
    kw = dict(train_ratio=dcfg.get('train_ratio', 0.8),
              val_ratio=dcfg.get('val_ratio', 0.1),
              random_seed=dcfg.get('random_seed', 42),
              feature_indices=dcfg.get('feature_indices'),
              zero_gauge_features=dcfg.get('zero_gauge_features', False),
              max_nodes=None, augment=False)
    kw.update({k: v for k, v in overrides.items() if v is not None})
    dataset = GNOTDataset(data_path, split=split, **kw)
    exp_dim = model.hparams.get('val_dim')
    if len(dataset):
        got_dim, _ = dataset.data_dims()
        if exp_dim is not None and got_dim != exp_dim:
            raise ValueError(f"Checkpoint expects val_dim={exp_dim} but dataset "
                             f"{data_path} provides {got_dim} features per node.")
    return dataset


def rescale_to_target(E_hat, E_tgt):
    """Per-column norm match of prediction to target (numpy [N, K]).

    Used for models trained with a scale-invariant field loss (SpectralNO:
    M-orthonormal output amplitude), so errors match the training metric.
    """
    n_h = np.linalg.norm(E_hat, axis=0)
    n_t = np.linalg.norm(E_tgt, axis=0)
    return E_hat * (n_t / np.maximum(n_h, 1e-8))[None, :]


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_geometry_comparison(geom_id, modes_data, save_path, elements=None, cluster_info=""):
    """Plot every mode of a geometry, ordered by ascending predicted frequency.

    Each mode row: Ground Truth | Prediction (or subspace projection) | Error.
    Near-degenerate modes are annotated and shown with subspace error.
    """
    sorted_modes = sorted(modes_data.items())
    n_modes = len(sorted_modes)

    fig, axes = plt.subplots(n_modes, 3, figsize=(18, 5 * n_modes), squeeze=False)
    title = f"Geometry ID: {geom_id}"
    if cluster_info:
        title += f"   [{cluster_info}]"
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    for i, (disp_idx, data) in enumerate(sorted_modes):
        coords  = data['coords']
        target  = data['target'].flatten()
        pred    = data['pred'].flatten()
        f_true  = data['f_true']
        f_pred  = data['f_pred']
        err_val = data['err_val']
        err_label = data['err_label']
        mode_label = data['mode_label']

        if elements is not None:
            tri = Triangulation(coords[:, 0], coords[:, 1], elements)
        else:
            tri = Triangulation(coords[:, 0], coords[:, 1])

        im1 = axes[i, 0].tripcolor(tri, target, cmap='RdBu_r', shading='gouraud', vmin=-1, vmax=1)
        axes[i, 0].set_title(
            f"{mode_label} — Ground Truth\nFreq: {f_true:.3f} GHz", fontsize=12)
        fig.colorbar(im1, ax=axes[i, 0])

        im2 = axes[i, 1].tripcolor(tri, pred, cmap='RdBu_r', shading='gouraud', vmin=-1, vmax=1)
        axes[i, 1].set_title(
            f"{mode_label} — Prediction\nFreq pred: {f_pred:.3f} GHz", fontsize=12)
        fig.colorbar(im2, ax=axes[i, 1])

        error = pred - target
        err_max = max(np.abs(error).max(), 1e-8)
        im3 = axes[i, 2].tripcolor(tri, error, cmap='RdBu_r', shading='gouraud',
                                   vmin=-err_max, vmax=err_max)
        axes[i, 2].set_title(
            f"{mode_label} — Error\n{err_label}: {err_val:.4f}   max|err|: {err_max:.3f}",
            fontsize=12)
        fig.colorbar(im3, ax=axes[i, 2])

        for ax in axes[i]:
            ax.set_aspect('equal')
            ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    model, device = load_model(args.checkpoint)

    print(f"Loading dataset from: {args.data_path}")
    dataset = build_dataset(model, args.data_path, args.split)
    scale_inv = bool(getattr(model, 'scale_invariant_field', False))

    if hasattr(dataset, 'stats') and dataset.stats:
        model.freq_stats = dataset.stats

    if args.freq_mean is not None and args.freq_std is not None:
        model.freq_stats = {'mean': args.freq_mean, 'std': args.freq_std}

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=gnot_collate_fn)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load triangle connectivity for FEM visualisation
    elements_pool = {}
    data_path_str = str(args.data_path)
    if data_path_str.endswith('.pkl'):
        import pickle as _pkl
        with open(args.data_path, 'rb') as ef:
            raw = _pkl.load(ef)
        for g_id, geom in raw['geometry_pool'].items():
            if 'elements' in geom:
                elements_pool[int(g_id)] = geom['elements']
    elif data_path_str.endswith('.h5'):
        import h5py
        with h5py.File(args.data_path, 'r') as ef:
            for g_id_str in ef['geometry_pool']:
                grp = ef['geometry_pool'][g_id_str]
                if 'elements' in grp:
                    elements_pool[int(g_id_str)] = grp['elements'][:]
    print(f"Loaded mesh connectivity for {len(elements_pool)} geometries.")

    print(f"Running inference for up to {args.num_samples} geometries...")
    geometries_results = {}
    refine_err = []   # (model, refined) mean |Δf| [GHz] per geometry

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            outputs = model(batch)
            preds   = outputs['field']       # [B, N, K]
            targets = batch['Y_field']       # [B, N, K]
            coords  = batch['X']             # [B, N, 2]
            mask    = batch.get('Mask', None)
            geom_ids = batch['geom_id'].squeeze(-1).cpu().numpy()

            f_pred = outputs['freq']         # [B, K] ascending
            f_true = batch['Y_freq']         # [B, K] ascending
            if model.freq_stats:
                f_pred_ph = f_pred * model.freq_stats['std'] + model.freq_stats['mean']
                f_true_ph = f_true * model.freq_stats['std'] + model.freq_stats['mean']
            else:
                f_pred_ph, f_true_ph = f_pred, f_true

            B, N, K = preds.shape
            for i in range(B):
                g_id = int(geom_ids[i])
                if g_id in geometries_results or len(geometries_results) >= args.num_samples:
                    continue

                m_idx = mask[i].cpu().numpy() if mask is not None else np.ones(N, dtype=bool)
                fp_i  = f_pred_ph[i].cpu().numpy()   # [K] physical GHz
                ft_i  = f_true_ph[i].cpu().numpy()   # [K] physical GHz

                # Predicted and target fields at valid nodes, normalised to unit max
                E_hat = preds[i][m_idx].cpu().numpy()     # [N_valid, K]
                E_tgt = targets[i][m_idx].cpu().numpy()   # [N_valid, K]
                xy    = coords[i][m_idx].cpu().numpy()    # [N_valid, 2]

                # ── Optional FEM refinement (--refine N) ────────────────────
                # Prediction → trial subspace → N steps of inverse iteration +
                # Rayleigh–Ritz with the mesh's P2 matrices (src/fem_refine.py).
                refined = bool(args.refine) and g_id in elements_pool and 'Scale' in batch
                if refined:
                    s_i = float(batch['Scale'][i])
                    _, E_hat, fp_ref = refine_modes(xy * s_i, elements_pool[g_id], E_hat, args.refine)
                    refine_err.append((np.abs(np.sort(fp_i) - ft_i).mean(), np.abs(fp_ref - ft_i).mean()))
                    fp_i = fp_ref

                # ── Slot-to-mode alignment ──────────────────────────────────
                # GNOT: Hungarian OT matching (ordering not guaranteed).
                # SpectralNO: ordering guaranteed by eigh; no OT needed.
                _model_type = getattr(model, 'model_type', 'gnot')
                if _model_type == 'spectral_no' or refined:
                    perm = np.arange(K, dtype=np.int64)  # identity (eigh / Ritz order)
                else:
                    fp_norm_i = f_pred[i].cpu().numpy()  # z-scored (for cost)
                    ft_norm_i = f_true[i].cpu().numpy()
                    freq_w = getattr(model, 'freq_match_weight', 0.5)
                    perm = _ot_match_np(fp_norm_i, ft_norm_i, E_hat, E_tgt,
                                        freq_w=freq_w)
                E_hat = E_hat[:, perm]    # aligned: column j = prediction for target mode j
                fp_i  = fp_i[perm]        # reorder physical freq predictions accordingly
                if scale_inv:
                    # Scale-invariant training loss (e.g. SpectralNO): the
                    # output amplitude is a gauge → norm-match before errors.
                    E_hat = rescale_to_target(E_hat, E_tgt)

                # Normalise each column to [-1, 1] for consistent colour scale
                def _norm_col(v):
                    mx = np.abs(v).max()
                    return v / mx if mx > 1e-8 else v

                # ── Subspace postprocessing ──────────────────────────────────
                # Detect near-degenerate clusters from TRUE frequencies
                # (after OT, fp_i[j] is the prediction for target mode j, so
                # we use ft_i to determine which modes form a degenerate pair)
                if args.deg_threshold is None:
                    # Same rule as the training loss/metric: absolute gap on
                    # the z-scored frequency axis (model.near_deg_threshold).
                    clusters = detect_clusters(f_true[i], model.near_deg_threshold)
                else:
                    clusters = _near_degenerate_clusters(ft_i, rel_threshold=args.deg_threshold)

                modes = {}
                cluster_parts = []
                for cl in clusters:
                    if len(cl) == 1:
                        # Non-degenerate: standard sign-aligned comparison
                        k = cl[0]
                        p = E_hat[:, k]
                        t = E_tgt[:, k]
                        rel_pos = np.linalg.norm(p - t) / (np.linalg.norm(t) + 1e-8)
                        rel_neg = np.linalg.norm(p + t) / (np.linalg.norm(t) + 1e-8)
                        if rel_neg < rel_pos:
                            p_viz, err = -p, rel_neg
                        else:
                            p_viz, err = p, rel_pos
                        modes[k] = dict(
                            coords=xy, target=_norm_col(t), pred=_norm_col(p_viz),
                            f_true=float(ft_i[k]), f_pred=float(fp_i[k]),
                            err_val=err, err_label='rel-L2',
                            mode_label=f'Mode {k} (isolated)',
                        )
                        cluster_parts.append(f'mode {k}: isolated')
                    else:
                        # Near-degenerate cluster: QR-orthonormalize predicted subspace,
                        # then measure subspace projection error per target vector.
                        E_sub_hat = E_hat[:, cl]           # [N, n]
                        E_sub_tgt = E_tgt[:, cl]           # [N, n]
                        Q_hat = _orthonormalize_np(E_sub_hat)  # canonical subspace basis

                        for rank, k in enumerate(cl):
                            t = E_sub_tgt[:, rank]
                            # Residual of t after projecting onto the predicted subspace
                            err = _subspace_rel_l2(Q_hat, t)
                            # Show the k-th orthonormal basis vector of the subspace
                            p_basis = Q_hat[:, rank]
                            modes[k] = dict(
                                coords=xy,
                                target=_norm_col(t),
                                pred=_norm_col(p_basis),    # orthonormal basis vector k
                                f_true=float(ft_i[k]),
                                f_pred=float(fp_i[k]),
                                err_val=err,
                                err_label='subspace-proj-err',
                                mode_label=f'Mode {k} (subspace basis {rank+1}/{len(cl)})',
                            )
                        cluster_parts.append(
                            f'modes {cl}: {len(cl)}D subspace  '
                            f'(proj-err={np.mean([modes[k]["err_val"] for k in cl]):.3f})')

                cluster_info = ' | '.join(cluster_parts)
                geometries_results[g_id] = dict(
                    modes=modes, elements=elements_pool.get(g_id),
                    cluster_info=cluster_info,
                )

            if len(geometries_results) >= args.num_samples:
                break

    if refine_err:
        e = np.asarray(refine_err)
        print(f"FEM refinement ({args.refine} step(s), {len(e)} geometries): mean |Δf| "
              f"model {e[:, 0].mean():.4f} GHz → refined {e[:, 1].mean():.6f} GHz")

    print(f"Plotting {len(geometries_results)} geometries...")
    for idx, (g_id, data) in enumerate(geometries_results.items()):
        save_path = os.path.join(args.output_dir, f"sample_geom_{g_id:04d}_all_modes.png")
        plot_geometry_comparison(g_id, data['modes'], save_path,
                                 data.get('elements'), data.get('cluster_info', ''))
        print(f"[{idx+1}/{len(geometries_results)}] Geometry {g_id}  "
              f"{data.get('cluster_info','')}  ->  {save_path}")

    print("Inference completed.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Infer and Visualize GNOT / SpectralNO Predictions")
    parser.add_argument("--config",       type=str, default=None,
                        help="Optional YAML config: fills --checkpoint / --data_path / "
                             "--output_dir / --num_samples from its dataset+inference sections")
    parser.add_argument("--checkpoint",   type=str, default=None,
                        help=".ckpt file or a training dir (best-*.ckpt / last.ckpt is picked)")
    parser.add_argument("--data_path",    type=str, default=None)
    parser.add_argument("--split",        type=str, default="val",
                        choices=["train", "val", "test"])
    parser.add_argument("--batch_size",   type=int, default=16)
    parser.add_argument("--num_samples",  type=int, default=None)
    parser.add_argument("--output_dir",   type=str, default=None)
    parser.add_argument("--deg_threshold", type=float, default=None,
                        help="Relative freq gap (physical units) below which modes are treated "
                             "as a subspace. Default: training rule (absolute gap on z-scored "
                             "freq, model.near_deg_threshold)")
    parser.add_argument("--refine",       type=int, default=0,
                        help="FEM refinement steps (inverse iteration + Rayleigh–Ritz "
                             "on the P2 mesh) applied to the prediction; 0 = off, 2 ≈ label accuracy")
    parser.add_argument("--freq_mean",    type=float, default=None)
    parser.add_argument("--freq_std",     type=float, default=None)
    args = parser.parse_args(argv)

    defaults = dict(data_path="data/gnot_dataset_5k.pkl", output_dir="inference_plots",
                    num_samples=5, checkpoint=None)
    if args.config:
        from src.config import load_config
        cfg = load_config(args.config)
        inf = cfg.get('inference', {}) or {}
        tr = cfg.get('training', {}) or {}
        defaults.update(
            data_path=(cfg.get('dataset', {}) or {}).get('data_path', defaults['data_path']),
            output_dir=inf.get('output_dir') or defaults['output_dir'],
            num_samples=inf.get('num_visualize') or defaults['num_samples'],
            # null → the experiment's log dir (resolve_checkpoint picks best)
            checkpoint=inf.get('checkpoint_path') or os.path.join(
                tr.get('log_dir', 'training_logs'), tr.get('exp_name', '')),
        )
    for k, v in defaults.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.checkpoint is None:
        parser.error("--checkpoint is required (or pass --config)")
    return args


if __name__ == "__main__":
    main(parse_args())

