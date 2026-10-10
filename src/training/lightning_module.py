"""Lightning module of the 3D cavity eigenmode model (EigenspaceOperator3D, E field, Whitney N0).

Loss (docs/20): NEO span / self-supervised compliance / basis-orthogonality terms on the raw trial
space, + freq_weight · z-MSE of the Ritz frequencies, + qoi_weight · cavity figure-of-merit loss
(docs/24 §5). Metrics: per-mode M-norm rel-L2 (subspace error inside near-degenerate clusters),
frequency MAE / relative error, QoI relative errors.
"""
import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics


def detect_clusters(f_true_matched, threshold, rel_threshold=None, freq_stats=None):
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
        rel_threshold: if set (and freq_stats given), use the scale-free
            relative gap in GHz instead, (f_i − f_{i−1}) / f_{i−1} < rel_threshold
            — the same cluster for a small and a large cavity (Davis–Kahan:
            eigenvector error ∝ error / relative gap).
    Returns:
        list[list[int]] e.g. [[0], [1, 2]] or [[0], [1], [2]].
    """
    f = f_true_matched.detach().float().tolist()
    K = len(f)
    if rel_threshold is not None and freq_stats:
        g = [v * freq_stats['std'] + freq_stats['mean'] for v in f]
        close = [abs(g[i] - g[i - 1]) < rel_threshold * abs(g[i - 1]) for i in range(1, K)]
    else:
        close = [abs(f[i] - f[i - 1]) < threshold for i in range(1, K)]
    clusters = [[0]]
    for i in range(1, K):
        if close[i - 1]:
            clusters[-1].append(i)
        else:
            clusters.append([i])
    return clusters


def count_near_degenerate(dataset, threshold, rel_threshold=None):
    """(geometries with a near-degenerate cluster, modes in such clusters, geometries) of a
    Maxwell3DDataset: the same normalized, ascending frequencies and detect_clusters threshold as
    the metric. Cheap: reads only Theta[1] (raw freq), never fields."""
    n_deg_geo = n_deg_modes = 0
    stats = dataset.stats
    for g_id in dataset.active_geoms:
        if hasattr(dataset, 'raw_freqs'):                      # FieldDataset
            raw = [float(v) for v in dataset.raw_freqs(g_id)]
        else:
            raw = [float(dataset.samples_metadata[j]['Theta'][1]) for j in dataset.geom_to_samples[g_id]]
        fn = sorted((r - stats['mean']) / stats['std'] for r in raw) if stats else sorted(raw)
        deg = [c for c in detect_clusters(torch.tensor(fn, dtype=torch.float32), threshold, rel_threshold, stats)
               if len(c) > 1]
        if deg:
            n_deg_geo += 1
            n_deg_modes += sum(len(c) for c in deg)
    return n_deg_geo, n_deg_modes, len(dataset.active_geoms)


def _basis_conditioning_loss(M_mat: torch.Tensor) -> torch.Tensor:
    """Collinearity penalty of the learned basis: mean squared off-diagonal mass correlation
    corr_mn = M_mn / √(M_mm M_nn) (the cosine between ψ_m and ψ_n) of the Gram M_mat [B, m, m];
    keeps the trial space well spread. Scalar in [0, 1]."""
    Msz = M_mat.shape[-1]
    d = torch.diagonal(M_mat, dim1=-2, dim2=-1).clamp(min=1e-8).rsqrt()  # [B, M]
    corr = M_mat * d.unsqueeze(-1) * d.unsqueeze(-2)                     # [B, M, M]
    eye = torch.eye(Msz, device=M_mat.device, dtype=M_mat.dtype)
    off = corr * (1.0 - eye)
    denom = max(Msz * (Msz - 1), 1)
    return (off ** 2).sum(dim=(-1, -2)).mean() / denom


def _jacobi_cholesky(G, ridge):
    """(C, d): Cholesky C of D G D + ridge·I with D = diag(G)^{-1/2} (dead
    columns → 0), i.e. a ridge relative to each column's own norm.  A column
    is dead when its diagonal is ≤ 1e-12 × the largest one: e.g. an exact
    gradient in 3D has rounding-level (≈1e-16) projected mass AND curl–curl
    entries, which the Jacobi scaling would otherwise blow up to O(1) noise."""
    diag = torch.diagonal(G, dim1=-2, dim2=-1)
    alive = diag > 1e-12 * diag.amax(-1, keepdim=True).clamp(min=0)
    d = torch.where(alive & (diag > 0), diag.clamp(min=1e-300).rsqrt(), torch.zeros_like(diag))
    eye = torch.eye(G.shape[-1], device=G.device, dtype=G.dtype)
    Gs = G * d.unsqueeze(-1) * d.unsqueeze(-2)
    Gs = 0.5 * (Gs + Gs.transpose(-1, -2))
    C, info = torch.linalg.cholesky_ex(Gs + ridge * eye)
    if bool((info > 0).any()):                     # numerically indefinite → bigger ridge
        C = torch.linalg.cholesky(Gs + max(ridge, 1e-9) * 1e3 * eye)
    return C, d


def span_residual(G, m, ridge=1e-9):
    """Fraction of each target NOT captured by span(V) in the norm of G.

        r_k = 1 − ‖P_V t_k‖²_G / ‖t_k‖²_G,    ‖P_V t‖²_G = g_kᵀ G_VV⁻¹ g_k,  g_k = VᵀG t_k

    with P_V the G-orthogonal projector onto span(V) (G = mass → L²(Ω),
    G = stiffness → energy/H¹₀ norm).  Invariant to each target's sign and
    to any invertible mixing of the basis columns; 0 for every target inside
    span(V) however a degenerate cluster is rotated (the mean over
    G-orthonormal targets is rotation invariant).  Needs no eigensolve and no
    near-degenerate cluster threshold.
    G_VV⁻¹ is applied by a Cholesky solve of the Jacobi-scaled Gram plus a
    relative ridge (the ridge only lowers the captured energy → r ≥ 0).

    Args:
        G: [B, m+n, m+n] Gram of [V | T] (hcurl_grams).
        m: number of basis columns.
    Returns:
        r [B, n] in [0, 1];  sqrt(r_k) is the best-approximation rel-L2 error
        of t_k from span(V) in that norm.
    """
    C, d = _jacobi_cholesky(G[:, :m, :m], ridge)
    Gvt = G[:, :m, m:] * d.unsqueeze(-1)                          # D VᵀG T
    W = torch.linalg.solve_triangular(C, Gvt, upper=False)       # C⁻¹ D VᵀG T
    energy = (W * W).sum(dim=-2)                                 # [B, n]
    tt = torch.diagonal(G[:, m:, m:], dim1=-2, dim2=-1)
    r = 1.0 - energy / tt.clamp(min=1e-300)
    return torch.where(tt > 0, r.clamp(min=0.0, max=1.0), torch.zeros_like(r))


def ritz_compliance(G_M, G_A, ridge=1e-9):
    """Expected compliance of the Galerkin solution for random loads,
    E_b[(Mb)ᵀ V (VᵀAV)⁻¹ VᵀMb],  b ~ N(0, M⁻¹)  (Cov(Mb) = M), in closed form:

        tr(G_A⁻¹ G_M) = Σ_i 1/θ_i      (θ_i: Ritz values of span(V)),

    evaluated by a Cholesky solve — no eigensolve.  Invariant to any
    invertible change of basis V → VS (so computed on the Jacobi-scaled
    pencil); by Ky Fan / Cauchy interlacing it is maximised exactly by the
    lowest-m eigenspace of the pencil (A, M).  Returns [B].
    """
    C, d = _jacobi_cholesky(G_A, ridge)
    Gm = G_M * d.unsqueeze(-1) * d.unsqueeze(-2)
    Z = torch.cholesky_solve(Gm, C)                               # (D G_A D)⁻¹ D G_M D
    return torch.diagonal(Z, dim1=-2, dim2=-1).sum(-1)


def ritz_logdet(G_M, G_A, ridge=1e-9):
    """(1/m) Σ_i log θ_i = (log det G_A − log det G_M) / m  (Cholesky, no
    eigensolve).  Minimised by the lowest-m eigenspace (θ_i ≥ λ_i for every i)
    with equal relative weight 1/m on every Ritz value; −log tr(G_A⁻¹G_M)
    puts weight (1/θ_i)/Σ_j(1/θ_j) on θ_i (mostly θ_1).  Both weights sum to
    1, so the two forms have the same gradient scale.  [B]"""
    Ca, d = _jacobi_cholesky(G_A, ridge)
    Gm = G_M * d.unsqueeze(-1) * d.unsqueeze(-2)
    eye = torch.eye(Gm.shape[-1], device=Gm.device, dtype=Gm.dtype)
    Cm = torch.linalg.cholesky(0.5 * (Gm + Gm.transpose(-1, -2)) + ridge * eye)
    return (_chol_logdet(Ca) - _chol_logdet(Cm)) / Gm.shape[-1]


def _chol_logdet(C):
    return 2.0 * torch.log(torch.diagonal(C, dim1=-2, dim2=-1)).sum(-1)


# Cavity QoI (docs/24): absolute R/Q floor [Ω] below which a mode counts as
# non-accelerating whatever the other modes of its geometry do (all-TE batches).
_QOI_RQ_ABS = 1e-6


def _qoi_term_tuple(terms):
    """qoi_terms as a validated tuple of qoi_torch keys ('a,b' strings accepted)."""
    from src.qoi.torch_qoi import QOI_KEYS
    if terms is None:
        return ()
    if isinstance(terms, str):
        terms = [t for t in (x.strip() for x in terms.split(',')) if t]
    terms = tuple(str(t) for t in terms)
    bad = [t for t in terms if t not in QOI_KEYS or t in ('U_J', 'L_acc_m', 'f_Hz', 'Rs_ohm')]
    if bad:
        raise ValueError(f"qoi_terms {bad} not usable; choose from "
                         f"{[k for k in QOI_KEYS if k not in ('U_J', 'L_acc_m', 'f_Hz', 'Rs_ohm')]}")
    return terms


class CavityLightning(pl.LightningModule):
    """Training wrapper of EigenspaceOperator3D. Argument names follow the checkpoints written so far
    (hidden_dim = embed_dim, eigenspace_kwargs = model.eigenspace); arguments of the removed 2D models
    in older checkpoints are accepted and ignored."""

    def __init__(self, val_dim, num_field_modes=6, hidden_dim=128, n_heads=4, n_basis=24,
                 rff_dim=64, rff_length_scale=0.1, dropout=0.0, eigenspace_kwargs=None,
                 lr=2e-4, weight_decay=1e-4, scheduler='custom_cosine', cosine_eta_min=1e-6,
                 onecycle_pct_start=0.3, onecycle_div_factor=25, onecycle_final_div_factor=1e4,
                 reducelr_patience=10, reducelr_factor=0.5, gradient_clip_val=None,
                 freq_weight=0.1, span_weight=1.0, selfsup_weight=0.01, ortho_weight=0.01,
                 span_norm='both', span_root=True, span_ridge=1e-9, selfsup_form='compliance',
                 near_deg_threshold=0.05, near_deg_rel_threshold=None,
                 qoi_weight=0.0, qoi_terms=('Q0', 'R_over_Q_ohm', 'G_ohm'), qoi_peak_p=None,
                 qoi_rq_floor=1e-2, data_cfg=None, model_type='eigenspace3d', **legacy):
        """
        span_norm: 'mass' (L²), 'energy' (curl–curl, controls the Ritz eigenvalue error) or 'both'.
        span_root: use √residual (= best-approximation rel-L2, the metric) instead of the residual.
        span_ridge: relative ridge of the Jacobi-scaled Gram solves.
        selfsup_form: 'compliance' (−log Σ 1/θ_i, NEO) or 'logdet' (mean log θ_i).
        qoi_weight: weight of the cavity figure-of-merit loss (docs/24 §5): mean squared log-ratio
            of the QoI of the Ritz fields at the predicted frequency vs the QoI of the FE targets
            (same per-geometry operators, src/qoi/torch_qoi.py), over isolated modes. 0 = off; the
            qoi_<name>_rel_err metrics are logged whenever the batch carries the QoI operators.
        qoi_terms: qoi_torch keys in the loss (list/tuple or 'a,b,c').
        qoi_peak_p: None = exact surface peaks in the loss, else a p-norm soft max.
        qoi_rq_floor: voltage-based quantities only count for modes whose FE R/Q exceeds
            qoi_rq_floor · max_k R/Q of the same geometry (non-accelerating modes: V ≈ 0).
        data_cfg: dataset settings at train time (split, seed, feature_indices, field ...), stored in
            the checkpoint so evaluation rebuilds the same split and inputs.
        model_type: 'eigenspace3d' (Whitney N0 edge model, Maxwell3DDataset) or 'field3d' (learned
            field in HCurl(p) on a curved mesh, EigenspaceOperatorField + FieldDataset, docs/28–29).
        """
        super().__init__()
        if model_type not in ('eigenspace3d', 'field3d'):
            raise ValueError(f"model_type {model_type!r}: 'eigenspace3d' or 'field3d' "
                             "(checkpoints of the removed 2D models: branch legacy-2d)")
        if model_type == 'field3d' and float(qoi_weight or 0.0) > 0:
            raise ValueError("qoi_weight > 0 is not available for field3d (the QoI operators are N0); "
                             "the p3 QoI labels are stored in the data (Y_qoi)")
        field = str((data_cfg or {}).get('field') or 'E').upper()
        if field != 'E':
            raise ValueError("checkpoint of the removed H formulation (field 'H', branch legacy-h): "
                             "only E-field models are supported")
        assert span_norm in ('mass', 'energy', 'both'), f"span_norm: {span_norm!r}"
        assert selfsup_form in ('compliance', 'logdet'), f"selfsup_form: {selfsup_form!r}"
        hp = {k: v for k, v in locals().items()
              if k not in ('self', 'legacy', 'field', '__class__')}
        self.save_hyperparameters(hp)
        self.model_type = model_type

        if model_type == 'field3d':
            from src.models.eigenspace_operator_field import EigenspaceOperatorField as Model
        else:
            from src.models.eigenspace_operator_3d import EigenspaceOperator3D as Model
        self.model = Model(val_dim=val_dim, embed_dim=hidden_dim, n_heads=n_heads,
                           n_basis=n_basis, num_field_modes=num_field_modes,
                           rff_dim=rff_dim, rff_length_scale=rff_length_scale,
                           dropout=dropout, **(eigenspace_kwargs or {}))
        self.num_field_modes = num_field_modes
        self.gradient_clip_val = gradient_clip_val
        self.freq_weight = freq_weight
        self.span_weight, self.selfsup_weight, self.ortho_weight = span_weight, selfsup_weight, ortho_weight
        self.span_norm, self.span_root, self.span_ridge = span_norm, bool(span_root), span_ridge
        self.selfsup_form = selfsup_form
        self.near_deg_threshold, self.near_deg_rel_threshold = near_deg_threshold, near_deg_rel_threshold
        self.qoi_weight = float(qoi_weight or 0.0)
        self.qoi_terms = _qoi_term_tuple(qoi_terms)
        self.qoi_peak_p = None if qoi_peak_p in (None, 0) else float(qoi_peak_p)
        self.qoi_rq_floor = float(qoi_rq_floor)
        if self.qoi_weight > 0 and not self.qoi_terms:
            raise ValueError("qoi_weight > 0 but qoi_terms is empty")
        self.freq_stats = None

        self.train_r2, self.val_r2, self.test_r2 = (torchmetrics.R2Score() for _ in range(3))
        self.val_mae, self.test_mae = torchmetrics.MeanAbsoluteError(), torchmetrics.MeanAbsoluteError()
        self._val_rl2_buffer: list = []   # per-geometry per-mode rel L2, for histograms

    # freq_stats is handed to the wrapped model (λ → z-scored GHz)
    @property
    def freq_stats(self):
        return self.__dict__.get('_freq_stats')

    @freq_stats.setter
    def freq_stats(self, value):
        self.__dict__['_freq_stats'] = value
        if 'model' in self._modules:
            self.model.freq_stats = value

    def forward(self, batch):
        return self.model(batch)

    def _clusters(self, f_true):
        """Near-degenerate mode groups of one sample (metric, evaluation)."""
        return detect_clusters(f_true, self.near_deg_threshold, self.near_deg_rel_threshold, self.freq_stats)

    def _span_terms(self, G_M, G_A, m, prefix):
        """NEO-style terms on the raw basis V (before Rayleigh–Ritz) from the mass / curl–curl
        Grams G_M, G_A [B, m+n, m+n] of [PV | T] (hcurl_grams; T = ALL stored target modes):

          span    mean_k r_k, r_k = 1 − ‖P_V t_k‖²/‖t_k‖² in the mass and/or energy norm
                  (span_residual; √r_k if span_root)
          selfsup −log tr(G_A⁻¹ G_M) = −log Σ 1/θ_i (label-free expected compliance) or
                  (1/m) Σ log θ_i ('logdet')
          ortho   off-diagonal mass correlations of V (_basis_conditioning_loss)
        Returns span_weight·span + selfsup_weight·selfsup + ortho_weight·ortho."""
        B = G_M.shape[0]
        r_M = span_residual(G_M, m, self.span_ridge)            # [B, n]
        r_A = span_residual(G_A, m, self.span_ridge)
        r = {'mass': r_M, 'energy': r_A, 'both': 0.5 * (r_M + r_A)}[self.span_norm]
        loss_span = (torch.sqrt(r + 1e-8) if self.span_root else r).mean()

        Gm_vv, Ga_vv = G_M[:, :m, :m], G_A[:, :m, :m]
        if self.selfsup_form == 'logdet':
            loss_ss = ritz_logdet(Gm_vv, Ga_vv, self.span_ridge).mean()
        else:
            loss_ss = -torch.log(ritz_compliance(Gm_vv, Ga_vv, self.span_ridge)
                                 .clamp(min=1e-300)).mean()
        loss_ortho = _basis_conditioning_loss(Gm_vv)

        total = self.span_weight * loss_span
        if self.selfsup_weight > 0.0:
            total = total + self.selfsup_weight * loss_ss
        if self.ortho_weight > 0.0:
            total = total + self.ortho_weight * loss_ortho

        kw = dict(on_step=False, on_epoch=True, batch_size=B, sync_dist=True)
        with torch.no_grad():
            self.log(f'{prefix}/span_loss', loss_span.float(), prog_bar=True, **kw)
            self.log(f'{prefix}/selfsup_loss', loss_ss.float(), **kw)
            self.log(f'{prefix}/ortho_loss', loss_ortho.float(), **kw)
            # Best-approximation rel-L2 of each stored target mode from span(V)
            # (mass norm ≈ lower bound of the post-Ritz mode_k_rel_l2, since the
            # Ritz modes lie in span(V); energy norm ~ the Ritz eigenvalue error).
            rl_M, rl_A = r_M.clamp(min=0).sqrt(), r_A.clamp(min=0).sqrt()
            self.log(f'{prefix}/span_rel_l2', rl_M.mean().float(), **kw)
            self.log(f'{prefix}/span_rel_l2_energy', rl_A.mean().float(), **kw)
            for k in range(rl_M.shape[-1]):
                self.log(f'{prefix}/span_mode_{k}_rel_l2', rl_M[:, k].mean().float(), **kw)
        return total

    def _compute_loss_3d(self, batch, prefix):
        """Loss of EigenspaceOperator3D (E-field N0 edge DOFs).

        Loss = the span / selfsup / ortho terms (_span_terms) on the Grams
        of [PV | T] from hcurl_grams: mass block VV = M_div (kernel-projected,
        so gradient content neither helps nor hurts), curl–curl blocks
        unchanged (K G = 0), T = ALL stored modes; + freq_weight · z-MSE of
        the K Ritz frequencies.  Metrics (no grad): per-mode M-norm rel-L2 of
        the Ritz fields vs the K lowest targets (sign-agnostic, subspace error
        inside near-degenerate clusters; NaN = excluded for a cluster that the
        K-th output splits, see _clusters_3d), freq MAE [GHz] and relative
        error, the basis' gradient mass fraction 1 − diag(M_div)/diag(VᵀMV),
        CG iterations.  Cavity figures of merit (batches with the QoI_* operator
        keys, docs/24 §5): + qoi_weight · QoI log-ratio loss when qoi_weight > 0,
        qoi_<name>_rel_err metrics always (_qoi_terms_3d).
        """
        from src.models.hcurl import KpSolve, hcurl_grams, mode_rel_l2
        out = self.model(batch)
        V, T = out['basis'], batch['Y_field']
        B, _, m = V.shape
        G_M, G_A, MT = hcurl_grams(V, T, out, batch)
        total = self._span_terms(G_M, G_A, m, prefix)
        K = out['field'].shape[-1]
        f_pred, f_true = out['freq'], batch['Y_freq'][:, :K]
        loss_freq = F.mse_loss(f_pred, f_true)
        if self.freq_weight > 0.0:
            total = total + self.freq_weight * loss_freq.to(total.dtype)
        loss_qoi = self._qoi_terms_3d(batch, out, T[..., :K], f_pred, f_true, prefix)
        if loss_qoi is not None:
            total = total + self.qoi_weight * loss_qoi.to(total.dtype)
        total = total.float()

        kw = dict(on_step=False, on_epoch=True, batch_size=B, sync_dist=True)
        with torch.no_grad():
            Fd, Tk, MTk = out['field'].double(), T[..., :K].double(), MT[..., :K]
            rl = []
            for b in range(B):
                inside, split = self._clusters_3d(batch, b, K)
                r = mode_rel_l2(Fd[b], Tk[b], MTk[b], inside).float()
                r[split] = float('nan')
                rl.append(r)
            rl = torch.stack(rl)                                            # [B, K]
            if prefix == 'val':
                self._val_rl2_buffer.extend(rl.cpu())
            grad_frac = 1.0 - (torch.diagonal(out['M_div'], dim1=-2, dim2=-1)
                               / torch.diagonal(out['M_V'], dim1=-2, dim2=-1).clamp(min=1e-300))
            self.log(f'{prefix}/freq_loss', loss_freq.float(), **kw)
            self.log(f'{prefix}/grad_frac', grad_frac.mean().float(), **kw)
            self.log(f'{prefix}/kp_cg_iters', float(KpSolve.last_iters), **kw)
            # NaN (split cluster) entries are excluded by weight, not by value:
            # each key is logged on every step with batch_size = its number of
            # valid entries (0 → contributes nothing), so the epoch value is the
            # mean over all valid (geometry, mode) entries.  A NaN value would
            # poison the epoch mean (→ EarlyStopping stops, ModelCheckpoint
            # never improves); skipping keys per step would differ across DDP
            # ranks (sync_dist).
            valid = torch.isfinite(rl)
            rl0 = torch.where(valid, rl, torch.zeros_like(rl))
            for k in range(K):
                n_k = int(valid[:, k].sum())
                self.log(f'{prefix}/mode_{k}_rel_l2', rl0[:, k].sum() / max(n_k, 1),
                         **dict(kw, batch_size=n_k))
            n_valid = int(valid.sum())
            field_rl = rl0.sum() / max(n_valid, 1)
            if self.freq_stats:
                fp = f_pred * self.freq_stats['std'] + self.freq_stats['mean']
                ft = f_true * self.freq_stats['std'] + self.freq_stats['mean']
                self.log(f'{prefix}/freq_mae_ghz', F.l1_loss(fp, ft), prog_bar=True, **kw)
                self.log(f'{prefix}/freq_rel_err', ((fp - ft).abs() / ft.abs()).mean(), **kw)
            # R2 bookkeeping: unit-M-norm targets vs sign-aligned predictions
            t_unit = Tk / (Tk * MTk).sum(1, keepdim=True).clamp(min=1e-300).sqrt()
            sgn = torch.sign(torch.einsum('bek,bek->bk', Fd, MTk)).unsqueeze(1)
            dof_mask = batch['EdgeMask'] if 'EdgeMask' in batch else batch['DofMask']     # N0 / field3d
            emask = dof_mask.unsqueeze(-1).expand_as(Fd)
            preds, targets = (Fd * sgn)[emask].float(), t_unit[emask].float()
        self.log(f'{prefix}/loss', total, on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=B, sync_dist=True)
        self.log(f'{prefix}/field_rel_l2', field_rl, on_step=True, on_epoch=True,
                 prog_bar=True, batch_size=n_valid, sync_dist=True)
        return total, preds, targets

    def _to_ghz(self, f):
        """z-scored frequency → GHz (float64); unchanged when there are no freq_stats."""
        f = f.double()
        if self.freq_stats:
            f = f * float(self.freq_stats['std']) + float(self.freq_stats['mean'])
        return f

    def _isolated_3d(self, batch, K):
        """bool [B, K]: output k of sample b is an isolated mode (a singleton
        cluster of _clusters_3d), i.e. neither inside a near-degenerate cluster
        (per-mode QoI depend on the arbitrary rotation within the eigenspace)
        nor in a cluster that the K-th output splits."""
        B = batch['Y_freq'].shape[0]
        iso = torch.zeros(B, K, dtype=torch.bool)
        for b in range(B):
            inside, _ = self._clusters_3d(batch, b, K)
            for c in inside:
                if len(c) == 1:
                    iso[b, c[0]] = True
        return iso

    def _qoi_terms_3d(self, batch, out, Tk, f_pred, f_true, prefix):
        """Cavity figures of merit (docs/24 §5) of the Ritz fields out['field'] at
        the predicted frequency vs those of the FE targets Tk [B, Ne, K] at the FE
        frequency, both evaluated on the fly with the batch's QoI operators
        (src/qoi/torch_qoi.qoi_torch), never with the stored labels.

        Mask (per sample b, output k): isolated modes only (_isolated_3d); the
        voltage-based quantities (torch_qoi.VOLTAGE_KEYS) additionally only for
        accelerating modes, FE R/Q > max(qoi_rq_floor · max_k R/Q_b, _QOI_RQ_ABS);
        non-finite / non-positive FE values are dropped.

        Loss (qoi_weight > 0): mean over qoi_terms of the per-term mean over the
        valid entries of (log q_pred − log q_true)²; returns None when off.
        Metrics (no grad, whenever the batch has the operators): for every label
        quantity {prefix}/qoi_<name>_rel_err = mean |q_pred/q_true − 1| over its
        valid entries (exact peaks), logged with batch_size = #valid (0 → no
        contribution; never NaN, as mode_k_rel_l2); {prefix}/qoi_loss and
        {prefix}/qoi_<term>_log_mse when the loss is on."""
        from src.qoi.torch_qoi import QOI_LABELS, VOLTAGE_KEYS, has_qoi_ops, qoi_torch
        if not has_qoi_ops(batch):
            if self.qoi_weight > 0:
                raise ValueError("qoi_weight > 0 but the batch has no QoI operators: build the "
                                 "datasets with Maxwell3DDataset(..., qoi_ops=True) "
                                 "(train.py does when training.qoi_weight > 0).")
            return None
        B, K = f_pred.shape
        fp, ft = self._to_ghz(f_pred), self._to_ghz(f_true)
        tiny = 1e-300
        with torch.no_grad():
            q_true = qoi_torch(batch, Tk.detach(), ft)
            iso = self._isolated_3d(batch, K).to(fp.device)
            rq = q_true['R_over_Q_ohm']
            acc = rq > torch.clamp(self.qoi_rq_floor * rq.amax(1, keepdim=True), min=_QOI_RQ_ABS)

            def valid(name, q):
                v = iso & torch.isfinite(q[name]) & (q[name] > 0)
                return v & acc if name in VOLTAGE_KEYS else v

        loss = q_pred = None
        if self.qoi_weight > 0:
            q_pred = qoi_torch(batch, out['field'], fp, peak_p=self.qoi_peak_p)
            with torch.no_grad():
                q_ref = q_true if self.qoi_peak_p is None else \
                    qoi_torch(batch, Tk.detach(), ft, peak_p=self.qoi_peak_p)
            terms = []
            for name in self.qoi_terms:
                v = valid(name, q_ref)
                d = torch.log(q_pred[name].clamp(min=tiny)) - torch.log(q_ref[name].clamp(min=tiny))
                d2 = torch.where(v, d * d, torch.zeros_like(d))
                terms.append(d2.sum() / max(int(v.sum()), 1))
            loss = torch.stack(terms).mean()

        kw = dict(on_step=False, on_epoch=True, sync_dist=True)
        with torch.no_grad():
            if loss is not None:
                self.log(f'{prefix}/qoi_loss', loss.detach().float(), **dict(kw, batch_size=B))
                for name, t in zip(self.qoi_terms, terms, strict=True):
                    self.log(f'{prefix}/qoi_{name}_log_mse', t.detach().float(), **dict(kw, batch_size=B))
            if q_pred is None or self.qoi_peak_p is not None:
                q_pred_m = qoi_torch(batch, out['field'].detach(), fp.detach())
            else:
                q_pred_m = {k: v.detach() for k, v in q_pred.items()}
            for name in QOI_LABELS:
                rel = (q_pred_m[name] / q_true[name].clamp(min=tiny) - 1.0).abs()
                v = valid(name, q_true) & torch.isfinite(rel)
                rel = torch.where(v, rel, torch.zeros_like(rel))
                n = int(v.sum())
                self.log(f'{prefix}/qoi_{name}_rel_err', (rel.sum() / max(n, 1)).float(),
                         **dict(kw, batch_size=n))
        return loss

    def _clusters_3d(self, batch, b, K):
        """(clusters inside the K outputs, split indices) of sample b, from ALL
        stored target frequencies plus FreqNext (mode K_data+1).  A near-degenerate
        cluster that continues past output K−1 is cut by the Ritz output: the
        K-th Ritz vector may be any combination of the pair, so its per-mode
        error is undefined → those indices are reported as split (NaN)."""
        f = batch['Y_freq'][b]
        nxt = batch.get('FreqNext', None)
        if nxt is not None and torch.isfinite(nxt[b]):
            f = torch.cat([f, nxt[b:b + 1]])
        clusters = self._clusters(f)
        inside = [c for c in clusters if c[-1] < K]
        split = [k for c in clusters if c[0] < K <= c[-1] for k in c if k < K]
        return inside, split

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self._compute_loss_3d(batch, "train")
        self.train_r2(preds, targets)
        self.log('train/r2', self.train_r2, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('lr', self.optimizers().param_groups[0]['lr'], prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._compute_loss_3d(batch, "val")
        self.val_r2(preds, targets)
        self.val_mae(preds, targets)
        self.log('val/r2', self.val_r2, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val/mae', self.val_mae, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def on_validation_epoch_start(self):
        self._val_rl2_buffer = []

    def on_validation_epoch_end(self):
        # Only print on rank 0 to avoid duplicate output in DDP
        if self.global_rank != 0:
            return

        # TensorBoard-only calls (add_scalar/add_histogram) are skipped when the
        # logger is CSV / absent (e.g. tensorboard not installed).
        exp = getattr(self.logger, 'experiment', None) if self.logger is not None else None
        tb = exp if hasattr(exp, 'add_histogram') else None

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
        print("  Mode      │   Train Rel   │    Val Rel    │ Progress")
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
        if tb is not None and self._val_rl2_buffer and self.current_epoch % _HIST_EVERY == 0:
            rl_all = torch.stack(self._val_rl2_buffer)  # [N_val, K]
            for k in range(rl_all.shape[1]):
                vals = rl_all[:, k][torch.isfinite(rl_all[:, k])]   # 3D: NaN = split cluster
                if vals.numel():
                    tb.add_histogram(f'val/mode_{k}_rel_l2_dist', vals,
                                     global_step=self.current_epoch)

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self._compute_loss_3d(batch, "test")
        self.test_r2(preds, targets)
        self.test_mae(preds, targets)
        self.log('test/r2', self.test_r2, on_step=False, on_epoch=True, sync_dist=True)
        self.log('test/mae', self.test_mae, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        param_groups = [{"params": list(self.parameters()), "lr": self.hparams.lr}]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.hparams.weight_decay)
        
        if self.hparams.scheduler == 'onecycle':
            # max_lr per param group: a scalar would overwrite lr_mode_specific
            # / lr_freq_heads group LRs with the base lr.
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[g['lr'] for g in optimizer.param_groups],
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
                    # Step only on epochs that actually ran validation, else
                    # Lightning raises "monitor val/field_rel_l2 not available"
                    # when check_val_every_n_epoch > 1.
                    "frequency": max(1, int(getattr(self.trainer, 'check_val_every_n_epoch', 1) or 1)),
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


GNOTLightning = CavityLightning      # name used by older checkpoints, scripts and notebooks
