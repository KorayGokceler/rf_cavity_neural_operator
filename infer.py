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
from src.training.lightning_module import GNOTLightning


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
    print(f"Loading checkpoint from: {args.checkpoint}")
    model = GNOTLightning.load_from_checkpoint(args.checkpoint)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    print(f"Loading dataset from: {args.data_path}")
    dataset = GNOTDataset(args.data_path, split=args.split)

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

    with torch.no_grad():
        for batch in dataloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v
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

                # ── OT matching: align predicted slots to target modes ───────
                # The model emits K (freq, field) pairs in slot order 0..K-1
                # which need not match target mode order.  Apply the same
                # Hungarian assignment used during training so every downstream
                # comparison is apples-to-apples.
                fp_norm_i = f_pred[i].cpu().numpy()  # z-scored (for cost)
                ft_norm_i = f_true[i].cpu().numpy()
                freq_w = getattr(model, 'freq_match_weight', 0.5)
                perm = _ot_match_np(fp_norm_i, ft_norm_i, E_hat, E_tgt,
                                    freq_w=freq_w)
                E_hat = E_hat[:, perm]    # aligned: column j = prediction for target mode j
                fp_i  = fp_i[perm]        # reorder physical freq predictions accordingly

                # Normalise each column to [-1, 1] for consistent colour scale
                def _norm_col(v):
                    mx = np.abs(v).max()
                    return v / mx if mx > 1e-8 else v

                # ── Subspace postprocessing ──────────────────────────────────
                # Detect near-degenerate clusters from TRUE frequencies
                # (after OT, fp_i[j] is the prediction for target mode j, so
                # we use ft_i to determine which modes form a degenerate pair)
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
                            # Best approximation of t inside predicted subspace
                            p_proj = _subspace_project(Q_hat, t)
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

    print(f"Plotting {len(geometries_results)} geometries...")
    for idx, (g_id, data) in enumerate(geometries_results.items()):
        save_path = os.path.join(args.output_dir, f"sample_geom_{g_id:04d}_all_modes.png")
        plot_geometry_comparison(g_id, data['modes'], save_path,
                                 data.get('elements'), data.get('cluster_info', ''))
        print(f"[{idx+1}/{len(geometries_results)}] Geometry {g_id}  "
              f"{data.get('cluster_info','')}  ->  {save_path}")

    print("Inference completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer and Visualize GNOT Predictions")
    parser.add_argument("--checkpoint",   type=str, required=True)
    parser.add_argument("--data_path",    type=str, default="data/gnot_dataset_5k.pkl")
    parser.add_argument("--split",        type=str, default="val",
                        choices=["train", "val", "test"])
    parser.add_argument("--batch_size",   type=int, default=16)
    parser.add_argument("--num_samples",  type=int, default=5)
    parser.add_argument("--output_dir",   type=str, default="inference_plots")
    parser.add_argument("--deg_threshold", type=float, default=0.05,
                        help="Relative freq gap below which modes are treated as a subspace")
    parser.add_argument("--freq_mean",    type=float, default=None)
    parser.add_argument("--freq_std",     type=float, default=None)
    args = parser.parse_args()
    main(args)



def plot_geometry_comparison(geom_id, modes_data, save_path, elements=None, deg_note=""):
    """Plot every mode of a geometry, ordered by ascending predicted frequency.

    Each mode is one row: Ground Truth | Prediction | Error.
    """
    # Modes are keyed by display index (0,1,2 = ascending predicted frequency)
    sorted_modes = sorted(modes_data.items())
    n_modes = len(sorted_modes)

    fig, axes = plt.subplots(n_modes, 3, figsize=(18, 5 * n_modes), squeeze=False)
    title = f"Geometry ID: {geom_id}"
    if deg_note:
        title += f"   ({deg_note})"
    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)

    for i, (disp_idx, data) in enumerate(sorted_modes):
        coords = data['coords']
        target = data['target'].flatten()
        pred = data['pred'].flatten()
        f_true = data['f_true']
        f_pred = data['f_pred']
        rel_l2 = data['rel_l2']
        sign_info = data['sign_info']

        if elements is not None:
            tri = Triangulation(coords[:, 0], coords[:, 1], elements)
        else:
            tri = Triangulation(coords[:, 0], coords[:, 1])

        im1 = axes[i, 0].tripcolor(tri, target, cmap='RdBu_r', shading='gouraud', vmin=-1, vmax=1)
        axes[i, 0].set_title(
            f"Mode {disp_idx} (by ascending frequency) - Ground Truth\nFreq: {f_true:.3f} GHz",
            fontsize=12)
        fig.colorbar(im1, ax=axes[i, 0])

        im2 = axes[i, 1].tripcolor(tri, pred, cmap='RdBu_r', shading='gouraud', vmin=-1, vmax=1)
        axes[i, 1].set_title(
            f"Mode {disp_idx} - GNOT Prediction{sign_info}\nFreq: {f_pred:.3f} GHz (Rel L2: {rel_l2:.3f})",
            fontsize=12)
        fig.colorbar(im2, ax=axes[i, 1])

        error = pred - target
        err_max = max(np.abs(error).max(), 1e-8)
        im3 = axes[i, 2].tripcolor(tri, error, cmap='RdBu_r', shading='gouraud', vmin=-err_max, vmax=err_max)
        axes[i, 2].set_title(f"Mode {disp_idx} - Error (pred - truth)\nmax |err|: {err_max:.3f}", fontsize=12)
        fig.colorbar(im3, ax=axes[i, 2])

        for ax in axes[i]:
            ax.set_aspect('equal')
            ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def _near_degenerate_note(freqs, rel_threshold=0.05):
    """Return a human-readable note about near-degenerate frequency pairs.

    freqs: 1-D array of (ascending) frequencies in physical units.
    """
    pairs = []
    K = len(freqs)
    f_mean = float(np.mean(np.abs(freqs))) if K else 0.0
    denom = max(abs(f_mean), 1e-6)
    for i in range(1, K):
        if abs(freqs[i] - freqs[i - 1]) / denom < rel_threshold:
            pairs.append((i - 1, i))
    if not pairs:
        return "well-separated modes"
    return "near-degenerate: " + ", ".join(f"modes {a}&{b}" for a, b in pairs)


def main(args):
    print(f"Loading checkpoint from: {args.checkpoint}")
    model = GNOTLightning.load_from_checkpoint(args.checkpoint)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    print(f"Loading dataset from: {args.data_path}")
    dataset = GNOTDataset(args.data_path, split=args.split)

    if hasattr(dataset, 'stats') and dataset.stats:
        model.freq_stats = dataset.stats

    if args.freq_mean is not None and args.freq_std is not None:
        model.freq_stats = {'mean': args.freq_mean, 'std': args.freq_std}
        print(f"Using manual frequency stats override: mean={args.freq_mean}, std={args.freq_std}")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=gnot_collate_fn)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load triangle connectivity directly from raw data for proper FEM viz.
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

    with torch.no_grad():
        for batch in dataloader:
            if torch.cuda.is_available():
                batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            outputs = model(batch)
            preds = outputs['field']           # [B, N, K]
            targets = batch['Y_field']         # [B, N, K]
            coords = batch['X']                # [B, N, grid]
            mask = batch.get('Mask', None)
            geom_ids = batch['geom_id'].squeeze(-1).cpu().numpy()

            f_pred = outputs['freq']           # [B, K] (model-sorted ascending)
            f_true = batch['Y_freq']           # [B, K] (dataset-sorted ascending)
            if model.freq_stats:
                f_pred_ph = f_pred * model.freq_stats['std'] + model.freq_stats['mean']
                f_true_ph = f_true * model.freq_stats['std'] + model.freq_stats['mean']
            else:
                f_pred_ph, f_true_ph = f_pred, f_true

            B, N, K = preds.shape
            for i in range(B):
                g_id = int(geom_ids[i])
                if g_id in geometries_results:
                    continue
                if len(geometries_results) >= args.num_samples:
                    continue

                m = mask[i] if mask is not None else slice(None)
                fp_i = f_pred_ph[i].cpu().numpy()    # [K] ascending
                ft_i = f_true_ph[i].cpu().numpy()    # [K] ascending

                modes = {}
                for k in range(K):  # k = display index, ascending frequency
                    p_tensor = preds[i, m, k]
                    t_tensor = targets[i, m, k]

                    rel_pos = torch.norm(p_tensor - t_tensor) / (torch.norm(t_tensor) + 1e-8)
                    rel_neg = torch.norm(p_tensor + t_tensor) / (torch.norm(t_tensor) + 1e-8)
                    if rel_neg < rel_pos:
                        rel_l2 = rel_neg.item()
                        final_pred_viz = -p_tensor.cpu().numpy()
                        sign_info = "*"
                    else:
                        rel_l2 = rel_pos.item()
                        final_pred_viz = p_tensor.cpu().numpy()
                        sign_info = ""

                    modes[k] = {
                        'coords': coords[i, m].cpu().numpy(),
                        'target': t_tensor.cpu().numpy(),
                        'pred': final_pred_viz,
                        'f_true': float(ft_i[k]),
                        'f_pred': float(fp_i[k]),
                        'rel_l2': rel_l2,
                        'sign_info': sign_info,
                    }

                deg_note = _near_degenerate_note(ft_i)
                geometries_results[g_id] = {
                    'modes': modes,
                    'elements': elements_pool.get(g_id),
                    'deg_note': deg_note,
                }

            if len(geometries_results) >= args.num_samples:
                break

    print(f"Plotting {len(geometries_results)} geometries...")
    for idx, (g_id, data) in enumerate(geometries_results.items()):
        save_path = os.path.join(args.output_dir, f"sample_geom_{g_id:04d}_all_modes.png")
        plot_geometry_comparison(g_id, data['modes'], save_path,
                                 data.get('elements'), data.get('deg_note', ""))
        note = data.get('deg_note', "")
        print(f"[{idx + 1}/{len(geometries_results)}] Geometry {g_id} ({note}) -> {save_path}")

    print("Inference and grouped visualization completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer and Visualize GNOT Predictions")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the .ckpt model checkpoint file")
    parser.add_argument("--data_path", type=str, default="data/gnot_dataset_5k.pkl", help="Path to input dataset")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split to evaluate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for inference")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of geometries to visualize and save as PNG")
    parser.add_argument("--output_dir", type=str, default="inference_plots", help="Output directory for PNG plots")

    parser.add_argument("--freq_mean", type=float, default=None, help="Manual override for frequency mean")
    parser.add_argument("--freq_std", type=float, default=None, help="Manual override for frequency std")

    args = parser.parse_args()
    main(args)
