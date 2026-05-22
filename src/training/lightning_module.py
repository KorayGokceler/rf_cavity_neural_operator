import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics
import math
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from src.models.gnot import GNOTModel
from src.training.physics_losses import (
    rayleigh_quotient,
    eigenvalue_ordering_loss,
    parametric_expected_rayleigh,
    PhysicsCurriculum,
)


# Speed of light (m/s) — used to convert predicted frequency (GHz) to k² = (2πf/c)²
SPEED_OF_LIGHT = 299_792_458.0


# ════════════════════════════════════════════════════════════════════════════
#  Set-prediction loss helpers (OT matching + soft-Grassmannian subspace)
# ════════════════════════════════════════════════════════════════════════════

def ot_match(f_pred, f_true, E_hat, E_tgt, mask, freq_w):
    """Optimal (Hungarian) assignment of predicted slots to target modes.

    The cost combines normalized-frequency distance and a sign-agnostic
    relative field L2.  Near-degenerate modes therefore get matched within
    their own subspace automatically — no cluster threshold, no assumption
    about which modes are degenerate.  Matching is solved on *detached*
    tensors (no gradient through the argmin, as in DETR); the returned
    permutation only re-orders the differentiable field/freq tensors.

    Args:
        f_pred, f_true: [K] frequencies (any order).
        E_hat, E_tgt:   [N, K] predicted / target mode fields.
        mask:           [N] boolean (True = valid node) or None.
        freq_w:         scalar weight of the frequency term in the cost.
    Returns:
        perm: LongTensor [K] — predicted slot index for each target mode j,
              i.e. ``E_hat[:, perm]`` is aligned column-wise to ``E_tgt``.
    """
    K = f_true.shape[-1]
    with torch.no_grad():
        if mask is not None:
            m = mask.to(E_hat.dtype).unsqueeze(-1)         # [N, 1]
            Eh = E_hat * m
            Et = E_tgt * m
        else:
            Eh, Et = E_hat, E_tgt

        df = f_pred.view(-1, 1) - f_true.view(1, -1)        # [Kp, Kt]
        cost_f = df ** 2

        eh2 = (Eh ** 2).sum(0).view(-1, 1)                  # [Kp, 1]
        et2 = (Et ** 2).sum(0).view(1, -1)                  # [1, Kt]
        cross = Eh.transpose(-2, -1) @ Et                   # [Kp, Kt]  <eh_i, et_j>
        num_p = eh2 + et2 - 2.0 * cross                     # ||eh - et||^2
        num_n = eh2 + et2 + 2.0 * cross                     # ||eh + et||^2
        num = torch.minimum(num_p, num_n)                   # sign-agnostic
        cost_field = num / (et2 + 1e-8)                     # [Kp, Kt] rel L2^2

        C = (freq_w * cost_f + cost_field).detach().cpu().numpy()
        row, col = linear_sum_assignment(C)                 # row sorted 0..K-1
        perm = np.empty(K, dtype=np.int64)
        perm[col] = row                                     # perm[true j] = pred slot
    return torch.as_tensor(perm, device=E_hat.device)


def detect_clusters(f_true_matched, threshold):
    """Group mode indices by absolute frequency proximity (for 'hard' mode).

    Two consecutive (sorted) modes belong to the same cluster when their
    absolute gap on the (normalized) frequency scale is small:

        |f_i - f_{i-1}| < threshold.

    Frequencies are standardized (z-scored) upstream, so an absolute gap is
    well-defined and scale-consistent across samples; a relative gap divided
    by a near-zero standardized mean is numerically unstable and was the prior
    bug. ``threshold`` (default 0.05) is therefore an absolute distance on the
    normalized frequency axis.

    Args:
        f_true_matched: [K] target frequencies aligned to prediction order.
        threshold: absolute gap threshold on the normalized frequency scale.
    Returns:
        list[list[int]] e.g. [[0], [1, 2]] or [[0], [1], [2]].
    """
    f = f_true_matched.detach().float().tolist()
    K = len(f)
    clusters = [[0]]
    for i in range(1, K):
        if abs(f[i] - f[i - 1]) < threshold:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    return clusters


def count_near_degenerate(dataset, threshold):
    """How many train geometries contain a near-degenerate mode cluster.

    Mirrors exactly what training sees: same per-geometry normalized,
    ascending-sorted frequencies as GNOTDataset.__getitem__ and the same
    detect_clusters threshold used by the loss/metric.  Cheap — reads only
    per-mode Theta[1] (raw freq), never node fields.

    Returns (n_deg_geoms, n_deg_modes, total_geoms).
    """
    n_deg_geo = 0
    n_deg_modes = 0
    geoms = dataset.active_geoms
    use_h5 = getattr(dataset, 'is_h5', False)
    h5 = dataset._get_h5_handle() if use_h5 else None
    stats = dataset.stats
    for g_id in geoms:
        s_idx = dataset.geom_to_samples[g_id]
        if use_h5:
            raw = [float(h5['samples'][str(j)]['Theta'][1]) for j in s_idx]
        else:
            raw = [float(dataset.samples_metadata[j]['Theta'][1])
                   for j in s_idx]
        if stats:
            fn = [(r - stats['mean']) / stats['std'] for r in raw]
        else:
            fn = list(raw)
        fn.sort()  # __getitem__ orders modes by ascending normalized freq
        cl = detect_clusters(torch.tensor(fn, dtype=torch.float32), threshold)
        deg = [c for c in cl if len(c) > 1]
        if deg:
            n_deg_geo += 1
            n_deg_modes += sum(len(c) for c in deg)
    return n_deg_geo, n_deg_modes, len(geoms)


def _masked_orthonormalize(E, mask=None):
    """Return an orthonormal basis (over the masked rows) for the column span
    of E using a numerically stable reduced QR.

    Args:
        E: [N, n] matrix whose columns span the subspace.
        mask: [N] boolean (True = valid node) or None.
    Returns:
        Q: [N, n] with masked rows zeroed and Q^T Q = I over valid rows.
    """
    if mask is not None:
        m = mask.to(E.dtype).unsqueeze(-1)  # [N, 1]
        E = E * m
    # Reduced QR; add tiny ridge for stability when columns are near-collinear
    # (degenerate/near-degenerate eigenvectors).
    Q, R = torch.linalg.qr(E, mode='reduced')
    # Fix sign ambiguity of QR (not required for Grassmannian but keeps R* sane)
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    sign = torch.sign(diag)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    Q = Q * sign.unsqueeze(-2)
    if mask is not None:
        Q = Q * mask.to(Q.dtype).unsqueeze(-1)
    return Q


def grassmannian_loss(E_hat, E_tgt, mask=None):
    """Grassmannian subspace distance:  n - ||Q_hat^T Q_tgt||_F^2.

    Equals 0 when the two subspaces coincide and n when fully orthogonal.
    For n == 1 this reduces exactly to 1 - cos^2(theta) between the two
    vectors, i.e. the standard sign-invariant field similarity.

    Args:
        E_hat, E_tgt: [N, n] predicted / target subspace bases.
        mask: [N] boolean or None.
    """
    n = E_hat.shape[-1]
    Qh = _masked_orthonormalize(E_hat, mask)
    Qt = _masked_orthonormalize(E_tgt, mask)
    M = Qh.transpose(-2, -1) @ Qt          # [n, n]
    return n - (M ** 2).sum()


def soft_procrustes_loss(E_hat, E_tgt, f_matched, sigma, mask=None):
    """Frequency-weighted soft-subspace alignment loss.

    Only near-degenerate modes (small |f_i - f_j| relative to ``sigma``) are
    allowed to mix through an orthogonal Procrustes rotation; well-separated
    modes fall back to a per-mode sign-invariant relative L2.

    The previous formulation built ``M = (E_hat^T E_tgt) * W`` and took its
    SVD.  Element-wise masking of a Gram matrix does NOT yield a valid
    Procrustes rotation once three or more modes are involved (the off-block
    couplings of a 3-way degeneracy are silently dropped), so the recovered
    ``R`` was not orthogonal-optimal.  Instead we now:

        1. compute the *un-weighted* orthogonal Procrustes rotation ``R`` from
           the full Gram matrix and **detach** it — it is an alignment
           *target*, not a differentiable shortcut that could collapse the
           prediction onto a rotated copy of the target;
        2. gate rotation freedom per-mode by how strongly that mode couples to
           any other one:  ``alpha_j = max_{i != j} W[i, j]``;
        3. blend the rotated target (degenerate limit, alpha→1) with the
           sign-aligned raw target (separated limit, alpha→0).

    For well-separated modes this is exactly the standard sign-invariant
    relative L2; for a degenerate block it is the Grassmannian subspace
    distance realised through the optimal within-block rotation.

    Args:
        E_hat, E_tgt: [N, K] columns are the (mask-applied) mode fields.
        f_matched: [K] target frequencies (prediction-aligned order).
        sigma: scalar coupling bandwidth (ABSOLUTE, on the z-scored freq axis).
        mask: [N] boolean or None.
    """
    K = f_matched.shape[-1]
    if mask is not None:
        m = mask.to(E_hat.dtype).unsqueeze(-1)  # [N, 1]
        E_hat = E_hat * m
        E_tgt = E_tgt * m

    fi = f_matched.view(-1, 1)
    fj = f_matched.view(1, -1)
    sigma2 = (sigma ** 2) + 1e-12
    W = torch.exp(-((fi - fj) ** 2) / sigma2)  # [K, K], diag == 1

    with torch.no_grad():
        # Un-weighted orthogonal Procrustes:  argmin_R ||E_hat - E_tgt R||_F.
        C = E_hat.transpose(-2, -1) @ E_tgt          # [K, K]
        U, _, Vh = torch.linalg.svd(C)
        R = Vh.transpose(-2, -1) @ U.transpose(-2, -1)   # [K, K], R = V U^T

        # Per-mode mixing gate: strongest coupling of mode j to any other mode.
        eye = torch.eye(K, device=W.device, dtype=W.dtype)
        alpha = (W * (1.0 - eye)).max(dim=0).values  # [K] in [0, 1]

        # Sign-aligned raw target (sign-invariant for well-separated modes).
        s = torch.sign((E_hat * E_tgt).sum(dim=0, keepdim=True))  # [1, K]
        s = torch.where(s == 0, torch.ones_like(s), s)

    E_tgt_rot = E_tgt @ R                            # [N, K] subspace-aligned
    E_tgt_sgn = E_tgt * s                            # [N, K] sign-aligned
    E_eff = alpha * E_tgt_rot + (1.0 - alpha) * E_tgt_sgn

    num = ((E_hat - E_eff) ** 2).sum()
    den = (E_eff ** 2).sum() + 1e-8
    return num / den

class GNOTLightning(pl.LightningModule):
    def __init__(self, val_dim=6, grid_dim=2, hidden_dim=256,
                 n_shared_layers=2, n_mode_layers=2, n_field_head_layers=2,
                 n_heads=4, num_experts=4, num_field_modes=3,
                 lr=1e-3, freq_weight=0.5, smoothness_weight=0.0,
                 mode_loss_weights=None,
                 lr_mode_specific=None, lr_freq_heads=None,
                 scheduler='onecycle', weight_decay=1e-4, use_checkpoint=False,
                 reducelr_patience=10, reducelr_factor=0.5,
                 predict_frequency=True,
                 onecycle_pct_start=0.3, onecycle_div_factor=25, onecycle_final_div_factor=1e4,
                 cosine_eta_min=1e-6,
                 gradient_clip_val=None,
                 dropout=0.0,
                 rff_dim=64, rff_length_scale=0.1,
                 n_basis=16,
                 degeneracy_mode='soft',
                 near_deg_threshold=0.05,
                 deg_sigma_rel=0.5,
                 deg_sigma_abs=0.3,
                 slot_ortho_weight=0.1,
                 freq_match_weight=0.5,
                 # ── Physics-informed losses (Rayleigh / Gram-Schmidt) ──
                 enable_physics_loss=False,
                 enable_curriculum=False,
                 curriculum_e1=20,
                 curriculum_e2=80,
                 rayleigh_weight=0.0,
                 rayleigh_mode='autograd',
                 param_rayleigh_weight=0.0,
                 use_gram_schmidt=False,
                 order_weight=0.0,
                 order_margin=0.01,
                 length_scale=0.1):
        super().__init__()
        # Suppress harmless DDP + gradient checkpointing stream mismatch warning
        torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
        self.lr_mode_specific = lr_mode_specific
        self.lr_freq_heads = lr_freq_heads
        self.gradient_clip_val = gradient_clip_val
        
        self.save_hyperparameters()
        
        self.model = GNOTModel(
            val_dim=val_dim,
            grid_dim=grid_dim,
            embed_dim=hidden_dim,
            n_shared_layers=n_shared_layers,
            n_mode_layers=n_mode_layers,
            n_field_head_layers=n_field_head_layers,
            n_heads=n_heads,
            num_experts=num_experts,
            num_field_modes=num_field_modes,
            use_checkpoint=use_checkpoint,
            predict_frequency=predict_frequency,
            dropout=dropout,
            rff_dim=rff_dim,
            rff_length_scale=rff_length_scale,
            n_basis=n_basis,
            use_gram_schmidt=use_gram_schmidt,
        )
        self.freq_weight = freq_weight
        self.smoothness_weight = smoothness_weight
        self.predict_frequency = predict_frequency
        self.num_field_modes = num_field_modes
        self.freq_stats = None
        # Set-prediction / degeneracy handling
        assert degeneracy_mode in ('soft', 'hard'), \
            f"degeneracy_mode must be 'soft' or 'hard', got {degeneracy_mode}"
        self.degeneracy_mode = degeneracy_mode
        self.near_deg_threshold = near_deg_threshold
        # deg_sigma_rel kept only for checkpoint/back-compat; the soft loss now
        # uses the absolute deg_sigma_abs (see _compute_loss for the rationale).
        self.deg_sigma_rel = deg_sigma_rel
        self.deg_sigma_abs = deg_sigma_abs
        self.slot_ortho_weight = slot_ortho_weight
        self.freq_match_weight = freq_match_weight

        # ── Physics-informed losses (Rayleigh / Gram-Schmidt / ordering) ──
        # Defaults keep the legacy supervised-only behaviour exactly.
        self.enable_physics_loss = enable_physics_loss
        self.rayleigh_weight = rayleigh_weight
        self.rayleigh_mode = rayleigh_mode
        self.param_rayleigh_weight = param_rayleigh_weight
        self.use_gram_schmidt = use_gram_schmidt
        self.order_weight = order_weight
        self.order_margin = order_margin
        # Length scale used by the data-gen pipeline (`L=0.1 m` in
        # dataset_generator.py). Required to convert predicted GHz frequencies
        # into the same units as the Rayleigh quotient computed in normalised
        # geometry coordinates.
        self.length_scale = length_scale
        # Curriculum scheduler — three phases (physics warmup → ramp → supervised).
        self.curriculum = PhysicsCurriculum(
            e1=curriculum_e1,
            e2=curriculum_e2,
            w_field=1.0,                 # field-loss weight in Phase C
            w_freq=freq_weight,
            w_smooth=smoothness_weight,
            w_slot_ortho=slot_ortho_weight,
            w_phys_rayleigh=1.0,
            w_phys_order=max(order_weight, 0.1),
            w_phys_param=max(param_rayleigh_weight, 0.2),
            w_rayleigh_anchor=rayleigh_weight,
            w_order_anchor=order_weight,
            w_param_anchor=param_rayleigh_weight,
            enabled=enable_curriculum,
        )

        # Optional: Metrics to evaluate and measure model's success
        self.train_r2 = torchmetrics.R2Score()
        self.val_r2 = torchmetrics.R2Score()
        self.test_r2 = torchmetrics.R2Score()
        
        self.val_mae = torchmetrics.MeanAbsoluteError()
        self.test_mae = torchmetrics.MeanAbsoluteError()
        self._val_rl2_buffer: list = []   # per-geometry per-mode rel L2, for histograms

    def forward(self, batch):
        return self.model(batch)

    def _compute_loss(self, batch, prefix):
        # ── Physics-loss prep: enable autograd through node coordinates so
        # that Rayleigh quotient ∂E/∂x can be computed downstream. We only
        # turn this on when the autograd Rayleigh path is actually active,
        # because it inflates memory ~2x (see docs/13_RAYLEIGH_ANALYSIS.md).
        w_sched = self.curriculum.weights(self.current_epoch)
        need_rayleigh = (
            self.enable_physics_loss
            and (w_sched['rayleigh'] > 0 or w_sched['param'] > 0)
        )
        need_autograd = need_rayleigh and self.rayleigh_mode == 'autograd'

        # Lightning wraps validation/test under torch.no_grad() by default,
        # which would prevent autograd.grad from finding a graph for ∇E.
        # When autograd Rayleigh is active we must re-enable grad tracking
        # for the entire forward + loss computation, regardless of prefix.
        import contextlib
        grad_ctx = torch.enable_grad() if need_autograd else contextlib.nullcontext()

        with grad_ctx:
            if need_autograd:
                batch['X'] = batch['X'].detach().requires_grad_(True)

            return self._compute_loss_inner(batch, prefix, w_sched, need_rayleigh)

    def _compute_loss_inner(self, batch, prefix, w_sched, need_rayleigh):
        outputs = self.model(batch)
        mask = batch.get('Mask', None)               # [B, N] boolean

        pred_field = outputs['field']                # [B, N, K]
        true_field = batch['Y_field']                # [B, N, K]
        f_pred = outputs['freq']                     # [B, K] (model-sorted)
        f_true = batch['Y_freq']                     # [B, K] (dataset-sorted)
        B, N, K = pred_field.shape
        eps = 1e-8

        # --- PINN BOUNDARY CONSTRAINT (applied to every mode column) -------
        dist_bnd = batch['Input_funcs'][:, :, 2]     # [B, N]
        valid_mask = mask if mask is not None else torch.ones(B, N, dtype=torch.bool, device=pred_field.device)
        bnd_mask = (dist_bnd < 1e-4) & valid_mask    # [B, N]
        bnd_mask_f = bnd_mask.unsqueeze(-1).float()  # [B, N, 1]

        if bnd_mask.any():
            sq_diff = (pred_field ** 2) * bnd_mask_f                  # [B, N, K]
            loss_bnd = sq_diff.sum() / (bnd_mask_f.sum() * K).clamp(min=1.0)
        else:
            loss_bnd = torch.tensor(0.0, device=pred_field.device)
        self.log(f'{prefix}/loss_bnd', loss_bnd, prog_bar=True, batch_size=B, sync_dist=True)
        # NOTE: no hard zeroing of pred_field at boundary nodes.  The FEM target
        # field is NOT identically zero on the detected boundary band
        # (mean|Y_bnd| ~= 0.5), so multiplying predictions by (1 - bnd_mask)
        # forced the network toward a physically wrong field and produced a
        # systematically corrupted gradient.  Boundary behaviour is instead a
        # *soft* penalty (loss_bnd, weighted by smoothness_weight).

        # --- PER-SAMPLE LOSS ------------------------------------------------
        # Predicted slots are assigned to target modes per-sample by an optimal
        # (Hungarian) transport on a frequency + sign-agnostic-field cost.  This
        # behaves like a numerical solver: the network just emits K (freq,
        # field) pairs and the loss discovers the correspondence — no ascending-
        # sort assumption, no cluster threshold, scales to any K.
        device = pred_field.device
        loss_freq = torch.zeros((), device=device)
        loss_field = torch.zeros((), device=device)
        loss_ortho = torch.zeros((), device=device)
        rel_l2_per_mode = torch.zeros(K, device=device)
        rel_l2_count = torch.zeros(K, device=device)
        # Detach for metric bookkeeping — no gradient needed past this point.
        aligned_pred_field = pred_field.detach().clone()     # [B, N, K] — for R2/MAE

        for b in range(B):
            fp_b = f_pred[b]                          # [K]
            ft_b = f_true[b]                          # [K]
            E_hat = pred_field[b]                     # [N, K]
            E_tgt = true_field[b]                     # [N, K]
            m_b   = valid_mask[b]                     # [N]

            # OT (Hungarian) assignment: predicted slot -> target mode j.
            # perm/index_select is differentiable w.r.t. the field/freq values;
            # only the assignment itself is detached.
            perm = ot_match(fp_b, ft_b, E_hat, E_tgt, m_b, self.freq_match_weight)
            E_hat = E_hat[:, perm]                    # align predicted columns
            fp_b = fp_b[perm]
            aligned_pred_field[b] = aligned_pred_field[b][:, perm]  # R2/MAE consistent

            # Frequency regression loss (OT-matched MSE)
            loss_freq = loss_freq + F.mse_loss(fp_b, ft_b)

            if self.degeneracy_mode == 'hard':
                clusters = detect_clusters(ft_b, self.near_deg_threshold)
                fl_b = torch.zeros((), device=device)
                for cl in clusters:
                    cl_t = torch.tensor(cl, device=device, dtype=torch.long)
                    g = grassmannian_loss(E_hat[:, cl_t], E_tgt[:, cl_t], m_b)
                    fl_b = fl_b + g
                loss_field = loss_field + fl_b
            else:  # 'soft'
                # CRITICAL: sigma is an ABSOLUTE bandwidth on the z-scored
                # frequency axis, NOT deg_sigma_rel * mean_gap.  The mean gap
                # between standardized frequencies is ~O(1), so the old scaling
                # pinned every off-diagonal coupling at exp(-1/deg_sigma_rel^2)
                # regardless of the actual degeneracy — the soft subspace
                # coupling never activated.  A fixed sigma (in normalized-freq
                # units) makes W genuinely degeneracy-sensitive.
                sigma = self.deg_sigma_abs
                loss_field = loss_field + soft_procrustes_loss(
                    E_hat, E_tgt, ft_b, sigma, m_b)

                # Slot-collapse guard: the subspace loss is invariant to
                # within-block rotations, so two near-degenerate slots could
                # collapse onto the same direction (subspace loses a dimension).
                # Penalize squared cosine between slot fields, weighted by how
                # strongly the two modes couple (W).  Well-separated modes
                # (W ~= 0) are left free; degenerate ones are pushed apart.
                fi = ft_b.view(-1, 1)
                fj = ft_b.view(1, -1)
                W_b = torch.exp(-((fi - fj) ** 2) / (sigma ** 2 + 1e-12))  # [K,K]
                m_col = m_b.to(E_hat.dtype).unsqueeze(-1)                  # [N,1]
                Eh_n = F.normalize(E_hat * m_col, dim=0, eps=eps)          # [N,K]
                G = Eh_n.transpose(-2, -1) @ Eh_n                          # [K,K]
                eye_k = torch.eye(K, device=device, dtype=W_b.dtype)
                # 0.5 * sum over all i!=j  ==  sum over i<j  (G, W symmetric)
                loss_ortho = loss_ortho + 0.5 * (
                    (W_b * (1.0 - eye_k)) * (G ** 2)).sum()

            # --- relative L2 (reporting only) --------------------------------
            # Singleton modes: per-mode sign-agnostic rel L2 (unchanged).
            # Near-degenerate clusters: a fixed slot↔mode comparison is unfair
            # because any within-block rotation is a physically equivalent
            # eigenbasis, so it inflated the logged metric while the loss
            # (Grassmannian/Procrustes) was already subspace-correct.  Mirror
            # the evaluation convention (infer_val_all.py near_deg_subspace_relL2):
            # project each target column onto the predicted cluster subspace and
            # report the projection-residual rel L2.
            with torch.no_grad():
                m_f = m_b.float().unsqueeze(-1)       # [N, 1]
                sample_rl = torch.zeros(K, device=device)
                clusters = detect_clusters(ft_b, self.near_deg_threshold)
                for cl in clusters:
                    if len(cl) == 1:
                        k = cl[0]
                        eh = E_hat[:, k:k + 1]
                        et = E_tgt[:, k:k + 1]
                        num_p = (((eh - et) ** 2) * m_f).sum()
                        num_n = (((eh + et) ** 2) * m_f).sum()
                        den = ((et ** 2) * m_f).sum() + eps
                        rl = torch.sqrt(torch.minimum(num_p, num_n) / den)
                        rel_l2_per_mode[k] = rel_l2_per_mode[k] + rl
                        rel_l2_count[k] = rel_l2_count[k] + 1.0
                        sample_rl[k] = rl
                    else:
                        cl_t = torch.tensor(cl, device=device, dtype=torch.long)
                        Q = _masked_orthonormalize(E_hat[:, cl_t], m_b)  # [N,|cl|]
                        for k in cl:
                            et = E_tgt[:, k:k + 1] * m_f          # [N, 1]
                            coeff = Q.transpose(-2, -1) @ et       # [|cl|, 1]
                            et_proj = Q @ coeff                    # [N, 1]
                            num = ((et - et_proj) ** 2).sum()
                            den = (et ** 2).sum() + eps
                            rl = torch.sqrt(num / den)
                            rel_l2_per_mode[k] = rel_l2_per_mode[k] + rl
                            rel_l2_count[k] = rel_l2_count[k] + 1.0
                            sample_rl[k] = rl
                if prefix == 'val':
                    self._val_rl2_buffer.append(sample_rl.cpu())

        loss_freq = loss_freq / max(B, 1)
        loss_field = loss_field / max(B, 1)
        loss_ortho = loss_ortho / max(B, 1)

        if not (self.predict_frequency and outputs.get('freq') is not None):
            loss_freq = torch.zeros((), device=device)

        # ── Physics-driven loss terms (Rowan et al. arXiv:2506.04375) ──
        loss_rayleigh = torch.zeros((), device=device)
        loss_param_rayleigh = torch.zeros((), device=device)
        loss_order = torch.zeros((), device=device)
        if self.enable_physics_loss:
            node_area = batch['Input_funcs'][..., 5]               # [B, N]

            # K, M sparse caches if available on the batch (rayleigh_mode='fem')
            K_sparse = batch.get('K_sparse', None)
            M_sparse = batch.get('M_sparse', None)

            # Rayleigh quotient R[E_k] of predicted fields. Per-mode [B, K].
            if w_sched['rayleigh'] > 0 or w_sched['param'] > 0:
                R_pred = rayleigh_quotient(
                    pred_field,
                    coords=batch['X'] if self.rayleigh_mode == 'autograd' else None,
                    area=node_area,
                    mask=valid_mask,
                    mode=self.rayleigh_mode,
                    K_sparse=K_sparse,
                    M_sparse=M_sparse,
                )                                                   # [B, K]

                # Convert predicted frequency → squared wavenumber in the
                # same (normalised-coordinate) units as R_pred:
                #   f_GHz  = f_norm·σ + μ
                #   k²_phys = (2π·f_GHz·1e9 / c)²      (1/m²)
                #   k²_norm = k²_phys · L²              (dimensionless, L=length_scale)
                if self.freq_stats is not None and self.predict_frequency:
                    mu = float(self.freq_stats['mean'])
                    sigma = float(self.freq_stats['std'])
                    f_GHz = f_pred * sigma + mu
                    k2_phys = ((2.0 * math.pi * f_GHz * 1e9) / SPEED_OF_LIGHT) ** 2
                    lam_pred = k2_phys * (self.length_scale ** 2)   # [B, K]
                    # Relative MSE so the loss is scale-agnostic across modes
                    rel = (R_pred - lam_pred) / (lam_pred.abs() + 1e-6)
                    loss_rayleigh = (rel ** 2).mean()
                else:
                    # No frequency head: Rayleigh becomes the eigenvalue target
                    # itself (paper Section 4) — directly minimise R averaged
                    # over modes, scaled by mode index to encourage ordering.
                    loss_rayleigh = R_pred.mean()

                # Parametric expected Rayleigh — paper Section 5 (Eq. 16/18).
                # Mini-batch mean of Rayleigh; geometry distribution ρ(a) is
                # uniform by construction (random shapes per sample).
                loss_param_rayleigh = R_pred.mean()

            # Eigenvalue ordering hinge: encourages λ_k ≤ λ_{k+1}. Operates on
            # predicted frequencies (already sorted by the model, so this is a
            # cheap safety net rather than the primary mechanism).
            if w_sched['order'] > 0 and outputs.get('freq') is not None:
                loss_order = eigenvalue_ordering_loss(f_pred, margin=self.order_margin)

        # ── Total loss: supervised + physics, curriculum-weighted ──────────
        total_loss = (
            w_sched['field']      * loss_field
            + w_sched['freq']     * loss_freq
            + w_sched['smooth']   * loss_bnd
            + w_sched['slot_ortho'] * loss_ortho
            + w_sched['rayleigh'] * loss_rayleigh
            + w_sched['order']    * loss_order
            + w_sched['param']    * loss_param_rayleigh
        )

        # --- logging -----------------------------------------------------
        self.log(f'{prefix}/loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)
        self.log(f'{prefix}/field_loss', loss_field, on_step=False, on_epoch=True, prog_bar=False, batch_size=B, sync_dist=True)
        self.log(f'{prefix}/freq_loss', loss_freq, on_step=False, on_epoch=True, prog_bar=False, batch_size=B, sync_dist=True)
        self.log(f'{prefix}/ortho_loss', loss_ortho, on_step=False, on_epoch=True, prog_bar=False, batch_size=B, sync_dist=True)
        if self.enable_physics_loss:
            self.log(f'{prefix}/rayleigh_loss', loss_rayleigh, on_step=False, on_epoch=True, prog_bar=False, batch_size=B, sync_dist=True)
            self.log(f'{prefix}/param_rayleigh_loss', loss_param_rayleigh, on_step=False, on_epoch=True, prog_bar=False, batch_size=B, sync_dist=True)
            self.log(f'{prefix}/order_loss', loss_order, on_step=False, on_epoch=True, prog_bar=False, batch_size=B, sync_dist=True)
            # Log curriculum weights so we can see the phase transition in TB
            self.log(f'{prefix}/w_field', float(w_sched['field']), on_step=False, on_epoch=True, batch_size=B, sync_dist=True)
            self.log(f'{prefix}/w_rayleigh', float(w_sched['rayleigh']), on_step=False, on_epoch=True, batch_size=B, sync_dist=True)

        rel_l2_count = rel_l2_count.clamp(min=1.0)
        rel_l2_mean_per_mode = rel_l2_per_mode / rel_l2_count
        rel_l2 = rel_l2_mean_per_mode.mean()
        self.log(f'{prefix}/field_rel_l2', rel_l2, on_step=True, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)
        for k in range(K):
            self.log(f'{prefix}/mode_{k}_rel_l2', rel_l2_mean_per_mode[k],
                     on_step=False, on_epoch=True, prog_bar=False, batch_size=B, sync_dist=True)

        if self.predict_frequency and outputs.get('freq') is not None and self.freq_stats:
            with torch.no_grad():
                fp_ghz = f_pred * self.freq_stats['std'] + self.freq_stats['mean']
                ft_ghz = f_true * self.freq_stats['std'] + self.freq_stats['mean']
                mae_ghz = F.l1_loss(fp_ghz, ft_ghz)
                self.log(f'{prefix}/freq_mae_ghz', mae_ghz, on_step=False, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)

        # Tensors for torchmetrics (R2, MAE).
        # Sign is physically arbitrary for eigenvectors (loss is sign-invariant),
        # so flip each mode column to the sign that minimises squared error before
        # handing off to R2/MAE — otherwise a valid -sign prediction gives R2≈-3.
        with torch.no_grad():
            sign_aligned = aligned_pred_field.clone()
            m_exp = valid_mask.float().unsqueeze(-1)          # [B, N, 1]
            pos_err = ((sign_aligned - true_field) ** 2 * m_exp).sum(dim=1)   # [B, K]
            neg_err = ((sign_aligned + true_field) ** 2 * m_exp).sum(dim=1)   # [B, K]
            flip = (neg_err < pos_err).float().unsqueeze(1)   # [B, 1, K]  1=flip
            sign_aligned = sign_aligned * (1.0 - 2.0 * flip)

            mask_km = valid_mask.unsqueeze(-1).expand_as(sign_aligned)        # [B, N, K]
            preds_valid = sign_aligned[mask_km].contiguous()
            targets_valid = true_field[mask_km].contiguous()

        return total_loss, preds_valid, targets_valid

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self._compute_loss(batch, "train")
        self.train_r2(preds, targets)
        self.log('train/r2', self.train_r2, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        
        # Öğrenme oranını (learning rate) progress bar'a yansıt
        opt = self.optimizers()
        current_lr = opt.param_groups[0]['lr']
        self.log('lr', current_lr, prog_bar=True, on_step=True, on_epoch=False)
        
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._compute_loss(batch, "val")
        self.val_r2(preds, targets)
        self.val_mae(preds, targets)
        self.log('val/r2', self.val_r2, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/mae', self.val_mae, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        if hasattr(self.model, 'reset_expert_calls'):
            self.model.reset_expert_calls()
        self._val_rl2_buffer = []

    def on_validation_epoch_end(self):
        # Only print on rank 0 to avoid duplicate output in DDP
        if self.global_rank != 0:
            return

        # Log Expert Load Balancing per block
        if hasattr(self.model, 'get_expert_calls_per_block'):
            calls_dict = self.model.get_expert_calls_per_block()
            for block_name, calls in calls_dict.items():
                calls = calls.float()
                total_calls = calls.sum()
                if total_calls > 0:
                    percentages = (calls / total_calls) * 100
                    for i, p in enumerate(percentages):
                        self.logger.experiment.add_scalar(f"Experts_{block_name}/E{i}", p, self.current_epoch)
        
        metrics = self.trainer.logged_metrics
        epoch = self.trainer.current_epoch

        # Collect mode-specific errors for both Train and Val
        # We use a dict of {mode_idx: value}
        train_errors = {
            k.split('mode_')[1].split('_')[0]: v.item() if torch.is_tensor(v) else v
            for k, v in metrics.items() if k.startswith('train/mode_') and k.endswith('_rel_l2')
        }
        val_errors = {
            k.split('mode_')[1].split('_')[0]: v.item() if torch.is_tensor(v) else v
            for k, v in metrics.items() if k.startswith('val/mode_') and k.endswith('_rel_l2')
        }

        mode_indices = sorted(list(set(train_errors.keys()) | set(val_errors.keys())))
        if not mode_indices:
            return

        # Header
        print(f"\n{'━'*64}")
        print(f"  Epoch {epoch:>3d} │ Mode-Specific Relative L2 Field Error")
        print(f"{'─'*64}")
        print(f"  Mode      │   Train Rel   │    Val Rel    │ Progress")
        print(f"{'─'*64}")

        for m_idx in mode_indices:
            t_val = train_errors.get(m_idx, 0.0)
            v_val = val_errors.get(m_idx, 0.0)
            
            # Status indicators
            status = '✅' if v_val < 0.1 else ('⚠️ ' if v_val < 0.3 else '❌')
            if v_val == 0: status = '??'

            # Mini bar chart for Val error (Safe for NaN)
            if not np.isfinite(v_val):
                bar_len = 0
                bar = '?' * 15
            else:
                bar_len = int(min(v_val * 15, 15))
                bar = '█' * bar_len + '░' * (15 - bar_len)
            
            t_str = f"{t_val:10.4f}" if m_idx in train_errors else "   -      "
            v_str = f"{v_val:10.4f}" if m_idx in val_errors else "   -      "
            
            print(f"  Mode {m_idx:2s}   │  {t_str}   │  {v_str}   │ [{bar}] {status}")

        # Global Summary
        print(f"{'─'*64}")
        # Try both direct name and _epoch suffix (Lightning adds _epoch when on_step=True)
        global_val_l2 = metrics.get('val/field_rel_l2_epoch', metrics.get('val/field_rel_l2', 0.0))
        global_val_r2 = metrics.get('val/r2_epoch', metrics.get('val/r2', 0.0))
        
        print(f"  GLOBAL VAL  │  Rel L2: {global_val_l2:.4f}  │  R²: {global_val_r2:.4f}")
        print(f"{'━'*64}\n")

        # Per-geometry rel L2 error distribution histograms (every 5 epochs).
        # Each entry in _val_rl2_buffer is a [K] tensor (one geometry), so
        # stacking gives [N_val, K]; one histogram per mode.
        _HIST_EVERY = 5
        if self._val_rl2_buffer and self.current_epoch % _HIST_EVERY == 0:
            rl_all = torch.stack(self._val_rl2_buffer)  # [N_val, K]
            for k in range(rl_all.shape[1]):
                self.logger.experiment.add_histogram(
                    f'val/mode_{k}_rel_l2_dist',
                    rl_all[:, k],
                    global_step=self.current_epoch)

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self._compute_loss(batch, "test")
        self.test_r2(preds, targets)
        self.test_mae(preds, targets)
        self.log('test/r2', self.test_r2, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/mae', self.test_mae, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        base_lr = self.hparams.lr
        param_groups = []
        handled_param_ids = set()

        # Mode Specific LRs
        if self.lr_mode_specific is not None:
            for mode_idx, mode_lr in enumerate(self.lr_mode_specific):
                if mode_lr is not None and mode_idx < len(self.model.mode_field_blocks):
                    # Gather parameters for this mode's blocks and field head
                    mode_params = []
                    # Mode Field Blocks
                    for p in self.model.mode_field_blocks[mode_idx].parameters():
                        mode_params.append(p)
                        handled_param_ids.add(id(p))
                    # Mode branch heads (DeepONet coefficient heads)
                    for p in self.model.branch_net[mode_idx].parameters():
                        mode_params.append(p)
                        handled_param_ids.add(id(p))
                    
                    if len(mode_params) > 0:
                        param_groups.append({"params": mode_params, "lr": mode_lr})
                        print(f"Optimizer: Mode {mode_idx} branch trained with lr={mode_lr}")

        # Freq Head LR (single global head now)
        if self.lr_freq_heads is not None and self.predict_frequency:
            freq_params = []
            for p in self.model.freq_head_global.parameters():
                if id(p) not in handled_param_ids:
                    freq_params.append(p)
                    handled_param_ids.add(id(p))
            if len(freq_params) > 0:
                param_groups.append({"params": freq_params, "lr": self.lr_freq_heads})
                print(f"Optimizer: Frequency head trained with lr={self.lr_freq_heads}")

        # Base param group (General trunk, embeddings, unhandled parts)
        base_params = []
        for p in self.parameters():
            if id(p) not in handled_param_ids:
                base_params.append(p)
        
        if len(base_params) > 0:
            param_groups.append({"params": base_params, "lr": base_lr})
            print(f"Optimizer: Shared trunk & remaining params trained with lr={base_lr}")

        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.hparams.weight_decay)
        
        if self.hparams.scheduler == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, 
                max_lr=self.hparams.lr, 
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=self.hparams.onecycle_pct_start,
                div_factor=self.hparams.onecycle_div_factor,
                final_div_factor=self.hparams.onecycle_final_div_factor
            )
            return {
                "optimizer": optimizer, 
                "lr_scheduler": {
                    "scheduler": scheduler, 
                    "interval": "step"
                }
            }
        elif self.hparams.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=self.trainer.max_epochs, 
                eta_min=self.hparams.cosine_eta_min
            )
            return {
                "optimizer": optimizer, 
                "lr_scheduler": {
                    "scheduler": scheduler, 
                    "interval": "epoch"
                }
            }
        elif self.hparams.scheduler == 'reducelr':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.hparams.reducelr_factor,
                patience=self.hparams.reducelr_patience,
                min_lr=1e-7
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/field_rel_l2",
                    "interval": "epoch",
                    "frequency": 1
                }
            }
        elif self.hparams.scheduler == 'custom_cosine':
            # Phase 1 (epochs 0-9):  constant at base_lr (warm-up / stable start)
            # Phase 2 (epoch 10+):   cosine decay from base_lr down to cosine_eta_min
            def lr_lambda(epoch):
                base_lr = self.hparams.lr
                eta_min = self.hparams.cosine_eta_min
                warmup_epochs = 10
                if epoch < warmup_epochs:
                    return 1.0
                total_cos_epochs = self.trainer.max_epochs - warmup_epochs
                if total_cos_epochs <= 0:
                    return eta_min / base_lr
                progress = min(1.0, (epoch - warmup_epochs) / total_cos_epochs)
                cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
                target_lr = eta_min + (base_lr - eta_min) * cosine_factor
                return target_lr / base_lr

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch"
                }
            }
        else:
            return optimizer
