"""Lightning module of the 3D cavity eigenmode model (EigenspaceOperatorField, E field, HCurl(p) on
curved meshes, docs/28).

Loss: NEO span / self-supervised compliance / basis-orthogonality terms on the raw trial space,
+ freq_weight · z-MSE of the Ritz frequencies (per-sample fidelity weights, docs/34). Metrics: per-mode
M-norm rel-L2 (subspace error inside near-degenerate clusters), frequency MAE / relative error.

projection=False (docs/38): the TRAIN and VALIDATION steps skip the kernel projection and the Rayleigh–Ritz solve;
both compute the same loss (_compute_loss_3d_plain: span on the unprojected mass Gram and the energy Gram + ortho),
and val/span_loss is the monitored metric (monitor_metric).  The TEST step stays projected (Ritz frequencies, fields) and
adds test/loss_plain, the training loss on the test set.
"""
import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics

from src.data import gpu_transfer


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
    FieldDataset: the same normalized, ascending frequencies and detect_clusters threshold as
    the metric. Cheap: reads only the stored frequencies (raw_freqs), never fields."""
    n_deg_geo = n_deg_modes = 0
    stats = dataset.stats
    for g_id in dataset.active_geoms:
        raw = [float(v) for v in dataset.raw_freqs(g_id)]
        fn = sorted((r - stats['mean']) / stats['std'] for r in raw) if stats else sorted(raw)
        deg = [c for c in detect_clusters(torch.tensor(fn, dtype=torch.float32), threshold, rel_threshold, stats)
               if len(c) > 1]
        if deg:
            n_deg_geo += 1
            n_deg_modes += sum(len(c) for c in deg)
    return n_deg_geo, n_deg_modes, len(dataset.active_geoms)


def monitor_metric(projection):
    """The metric of ModelCheckpoint, EarlyStopping and the ReduceLROnPlateau scheduler: the Ritz field error
    val/field_rel_l2 (projection: true), or val/span_loss (projection: false: the validation step has no Ritz
    fields; val/span_loss is the span term of the training loss on the validation set, an upper bound of its
    projected span residual, docs/38 §2.3; the ortho term of val/loss is a regulariser, not a quality measure,
    and stays logged only)."""
    return 'val/field_rel_l2' if projection else 'val/span_loss'


def _basis_conditioning_loss(M_mat: torch.Tensor, rel_floor=None) -> torch.Tensor:
    """Collinearity penalty of the learned basis: mean squared off-diagonal mass correlation
    corr_mn = M_mn / √(M_mm M_nn) (the cosine between ψ_m and ψ_n) of the Gram M_mat [B, m, m];
    keeps the trial space well spread. Scalar in [0, 1].
    rel_floor (projection-free step, docs/38 §2.5): clamp the diagonal at rel_floor · (its sample maximum)
    instead of the absolute 1e-8, for a Gram whose columns may be (numerically) null, e.g. the curl–curl Gram
    of a gradient column; None = the absolute floor (the projected path, unchanged)."""
    Msz = M_mat.shape[-1]
    diag = torch.diagonal(M_mat, dim1=-2, dim2=-1)
    if rel_floor is None:
        d = diag.clamp(min=1e-8).rsqrt()                                  # [B, M]
    else:
        d = torch.maximum(diag, rel_floor * diag.amax(-1, keepdim=True)).clamp(min=1e-300).rsqrt()
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
    entries, which the Jacobi scaling would otherwise blow up to O(1) noise.

    No host synchronisation (docs/38 §11): a numerically indefinite Gram (any sample of the batch) falls back to
    a 1e3 × bigger ridge for the WHOLE batch, as before, but chosen by torch.where on a device-side flag instead
    of a Python `if` on `(info > 0).any()` (a GPU→host sync in every loss evaluation).  The fallback factorisation
    runs always, on `Gs + big·I` where the flag is set and on `big·I` (trivially positive definite: no NaN in the
    unused branch's backward) where it is not; where it is not taken C is bit for bit the plain factorisation.
    If the bigger ridge fails too the result is NaN (previously `torch.linalg.cholesky` raised): the loss is
    then NaN, EarlyStopping (check_finite) stops the run at the next validation."""
    diag = torch.diagonal(G, dim1=-2, dim2=-1)
    alive = diag > 1e-12 * diag.amax(-1, keepdim=True).clamp(min=0)
    d = torch.where(alive & (diag > 0), diag.clamp(min=1e-300).rsqrt(), torch.zeros_like(diag))
    eye = torch.eye(G.shape[-1], device=G.device, dtype=G.dtype)
    Gs = G * d.unsqueeze(-1) * d.unsqueeze(-2)
    Gs = 0.5 * (Gs + Gs.transpose(-1, -2))
    C, info = torch.linalg.cholesky_ex(Gs + ridge * eye)
    bad = (info > 0).any()                         # 0-d bool tensor on the device: numerically indefinite → bigger ridge
    C2, info2 = torch.linalg.cholesky_ex(torch.where(bad, Gs, torch.zeros_like(Gs)) + max(ridge, 1e-9) * 1e3 * eye)
    C = torch.where(bad, C2, C)
    return torch.where(bad & (info2 > 0).any(), torch.full_like(C, float('nan')), C), d


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


class CavityLightning(pl.LightningModule):
    """Training wrapper of EigenspaceOperatorField. Argument names follow the checkpoints written so far
    (hidden_dim = embed_dim, eigenspace_kwargs = model.eigenspace); arguments of removed features in
    older checkpoints (2D models, the N0 QoI loss: qoi_weight, qoi_terms, ...) are accepted and ignored."""

    def __init__(self, val_dim, num_field_modes=6, hidden_dim=128, n_heads=4, n_basis=24,
                 rff_dim=64, rff_length_scale=0.1, dropout=0.0, eigenspace_kwargs=None,
                 lr=2e-4, weight_decay=1e-4, scheduler='custom_cosine', cosine_eta_min=1e-6,
                 onecycle_pct_start=0.3, onecycle_div_factor=25, onecycle_final_div_factor=1e4,
                 reducelr_patience=10, reducelr_factor=0.5, gradient_clip_val=None,
                 freq_weight=0.1, span_weight=1.0, selfsup_weight=0.01, ortho_weight=0.01,
                 span_norm='both', span_root=True, span_ridge=1e-9, selfsup_form='compliance',
                 near_deg_threshold=0.05, near_deg_rel_threshold=None,
                 fidelity_weights=(1.0, 1.0), fidelity_freq_scale=(1.0, 1.0),
                 data_cfg=None, model_type='field3d', projection=True, diag_metrics_every_n_steps=1, **_legacy):
        """
        span_norm: 'mass' (L²), 'energy' (curl–curl, controls the Ritz eigenvalue error) or 'both'.
        span_root: use √residual (= best-approximation rel-L2, the metric) instead of the residual.
        span_ridge: relative ridge of the Jacobi-scaled Gram solves.
        selfsup_form: 'compliance' (−log Σ 1/θ_i, NEO) or 'logdet' (mean log θ_i).
        fidelity_weights: (LF, HF) per-sample weights of the span and frequency terms of field3d batches
            that carry item["Fidelity"] (FieldDataset, docs/34); the batch loss is mean_b(w_b · loss_b).
        fidelity_freq_scale: (LF, HF) extra factors on the frequency term: LF frequencies are the discrete
            eigenvalues of the model space (the Ritz values can reach them), HF ones lie below its floor.
        projection: True (default) = the training loss is computed on the kernel-projected Ritz problem (all terms
            above); validation is the same projected step (monitor val/field_rel_l2).  False = projection-free
            TRAINING and VALIDATION (docs/38): the forward builds V only (no Kp CG, no Rayleigh–Ritz) and the
            loss is span_weight · span + ortho_weight · ortho on the UNPROJECTED mass Gram (VᵀMV, VᵀMT, TᵀMT)
            and the energy Gram, computed by one function for both steps (monitor val/span_loss).  Needs
            selfsup_weight = freq_weight = 0 (ValueError otherwise: the compliance term rewards gradient content
            without the projection, the frequency term needs Ritz values) and span_norm 'mass' or 'both' (ValueError for
            'energy': nothing in an energy-norm loss sees gradient content).  The test step and prediction always
            project; the test step adds test/loss_plain.
        diag_metrics_every_n_steps: projection=False only (docs/38 §11).  1 (default) = every diagnostic metric of the
            training step is computed and logged on every step.  N > 1 = the expensive ones (train/span_rel_l2,
            span_rel_l2_energy, span_mode_k_rel_l2, mass_diag_ratio, grad_frac_lb, grad_frac_lb_max) only on the
            batches with batch_idx % N == 0 (the same batches on every DDP rank, so the same keys are logged on the
            same steps); their epoch values are then means over those batches only.  The loss, train/span_loss,
            train/ortho_loss and the gradients are the same on every step.  Validation always logs everything.
        data_cfg: dataset settings at train time (split, seed, feature_indices, field ...), stored in
            the checkpoint so evaluation rebuilds the same split and inputs.
        model_type: 'field3d' (learned field in HCurl(p) on a curved mesh, EigenspaceOperatorField +
            FieldDataset, docs/28–29), the only model; checkpoints of the removed models are rejected.
        """
        super().__init__()
        if model_type != 'field3d':
            raise ValueError(f"model_type {model_type!r}: only 'field3d' is supported (checkpoints of the removed "
                             "N0 edge model 'eigenspace3d' and of the removed 2D models (branch legacy-2d) "
                             "cannot be loaded)")
        if float(_legacy.get('qoi_weight', 0.0) or 0.0) > 0:
            raise ValueError("qoi_weight > 0: the QoI training loss of the removed N0 model is not available; "
                             "the p3 QoI labels are stored in the data (Y_qoi)")
        field = str((data_cfg or {}).get('field') or 'E').upper()
        if field != 'E':
            raise ValueError("checkpoint of the removed H formulation (field 'H', branch legacy-h): "
                             "only E-field models are supported")
        assert span_norm in ('mass', 'energy', 'both'), f"span_norm: {span_norm!r}"
        assert selfsup_form in ('compliance', 'logdet'), f"selfsup_form: {selfsup_form!r}"
        if not isinstance(projection, bool):
            raise ValueError(f"projection must be true or false, not {projection!r}")
        if not projection and (float(selfsup_weight) > 0.0 or float(freq_weight) > 0.0):
            raise ValueError(
                f"projection=False needs selfsup_weight = 0 and freq_weight = 0 (got selfsup_weight={selfsup_weight}, "
                f"freq_weight={freq_weight}): without the kernel projection the self-supervised compliance term "
                "rewards gradient content (docs/18 §3.4) and the frequency term needs the Ritz values "
                "(configs/field_3d_noproj.yaml)")
        if not projection and span_norm == 'energy':
            raise ValueError("projection=False with span_norm='energy': the energy norm and the energy-Gram ortho are blind "
                             "to the gradient content of V (K G = 0), so nothing would keep V free of gradients "
                             "(docs/38 §2.3); use span_norm 'mass' or 'both' (default)")
        diag_metrics_every_n_steps = int(diag_metrics_every_n_steps)
        if diag_metrics_every_n_steps < 1:
            raise ValueError(f"diag_metrics_every_n_steps must be >= 1, got {diag_metrics_every_n_steps}")
        if diag_metrics_every_n_steps > 1 and projection:
            raise ValueError("diag_metrics_every_n_steps > 1 is an option of the projection-free step "
                             "(training.projection: false); the projected step logs the Ritz metrics on every step")
        hp = {k: v for k, v in locals().items()
              if k not in ('self', '_legacy', 'field', '__class__')}
        self.save_hyperparameters(hp)
        self.model_type = model_type
        self.projection = projection
        self.diag_every = diag_metrics_every_n_steps
        self.gpu_prefetch = False            # train.py: training.gpu_prefetch (src/data/gpu_transfer.py, docs/38 §11)
        self._copy_stream = None
        self._clock = None                   # training.profile_steps: StepProfiler sets a StepClock (docs/38 §11)
        self._tables = {}                    # device-resident constants (fidelity weight tables): built once
        self._lr_cache = (None, None)

        from src.models.eigenspace_operator_field import EigenspaceOperatorField
        self.model = EigenspaceOperatorField(val_dim=val_dim, embed_dim=hidden_dim, n_heads=n_heads,
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
        self.fidelity_weights = tuple(float(w) for w in fidelity_weights)
        self.fidelity_freq_scale = tuple(float(w) for w in fidelity_freq_scale)
        if len(self.fidelity_weights) != 2 or len(self.fidelity_freq_scale) != 2:
            raise ValueError("fidelity_weights / fidelity_freq_scale: (LF, HF) pairs")
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

    def on_load_checkpoint(self, checkpoint):
        """Every field3d checkpoint stores hyper_parameters['model_type'] (it was an __init__ argument when
        field3d was added, with default 'eigenspace3d'); one without it is an N0 or 2D model, whose weights
        would otherwise load silently into the field3d default with strict=False (scripts/eval_fidelity.py)."""
        hp = checkpoint.get('hyper_parameters')
        if hp is not None and 'model_type' not in hp:
            raise ValueError("checkpoint without hyper_parameters['model_type']: a model older than field3d "
                             "(the removed N0 edge model or a 2D model), which cannot be loaded")

    def forward(self, batch):
        return self.model(batch)

    # ── host-sync-free helpers (docs/38 §11) ────────────────────────────────────────────────────────
    def _table(self, values, device):
        """torch.tensor(values, device=device) built once per (values, device): `torch.tensor(list, device=cuda)` is a
        pageable host→device copy that synchronises the stream, so it must not run in every loss evaluation."""
        key = (tuple(values), str(device))
        t = self._tables.get(key)
        if t is None:
            t = self._tables[key] = torch.tensor(list(values), device=device)
        return t

    def _lr_tensor(self, lr):
        """The learning rate as a device scalar for self.log (a Python float would be copied to the device on
        every call, a stream sync); rebuilt only when the value changes (per epoch with the epoch schedules)."""
        key = (lr, str(self.device))
        if self._lr_cache[0] != key:
            self._lr_cache = (key, torch.tensor(lr, device=self.device, dtype=torch.get_default_dtype()))
        return self._lr_cache[1]

    def _mark(self, stage):
        """Step-time breakdown (training.profile_steps): charge the time since the last mark to `stage`."""
        if self._clock is not None:
            self._clock.mark(stage)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """Default Lightning transfer, or with gpu_prefetch the sparse-parts path of src/data/gpu_transfer.py
        (pinned dense parts → non_blocking copy on a side stream → CSR rebuilt on the device; the step waits for the
        copy event, wait_ready).  The timing (if a step profile runs) is charged to 'transfer'."""
        if self._clock is not None:
            self._clock.begin_transfer()
        if self.gpu_prefetch:
            if self._copy_stream is None and torch.device(device).type == 'cuda':
                self._copy_stream = torch.cuda.Stream(device)
            batch = gpu_transfer.to_device(batch, device, self._copy_stream)
        else:
            batch = super().transfer_batch_to_device(batch, device, dataloader_idx)
        if self._clock is not None:
            self._clock.end_transfer()
        return batch

    def _clusters(self, f_true):
        """Near-degenerate mode groups of one sample (metric, evaluation)."""
        return detect_clusters(f_true, self.near_deg_threshold, self.near_deg_rel_threshold, self.freq_stats)

    def _span_terms(self, G_M, G_A, m, prefix, w=None, selfsup=True, ortho='mass', diag=True):
        """NEO-style terms on the raw basis V (before Rayleigh–Ritz) from the mass / curl–curl
        Grams G_M, G_A [B, m+n, m+n] of [PV | T] (hcurl_grams; T = ALL stored target modes):

          span    mean_k r_k, r_k = 1 − ‖P_V t_k‖²/‖t_k‖² in the mass and/or energy norm
                  (span_residual; √r_k if span_root)
          selfsup −log tr(G_A⁻¹ G_M) = −log Σ 1/θ_i (label-free expected compliance) or
                  (1/m) Σ log θ_i ('logdet')
          ortho   off-diagonal mass correlations of V (_basis_conditioning_loss)
        w [B]: per-sample weights of the span term (fidelity, docs/34); None = plain mean.
        prefix None: nothing is logged (the loss only).
        selfsup=False (projection-free step, selfsup_weight = 0): the compliance of the UNPROJECTED Gram is
        unbounded, so the term is neither computed nor logged.
        ortho='energy' (projection-free step): the collinearity of the columns is measured on the curl–curl
        Gram G_A[:m, :m] (gradient-blind, = the projected basis's energy Gram since K G = 0) instead of the mass
        Gram, whose correlations gradient content dilutes at constant span (docs/38 §2.5).
        diag=False (projection-free training step, diag_metrics_every_n_steps): the best-approximation diagnostics
        (span_rel_l2, span_rel_l2_energy, span_mode_k_rel_l2) are neither computed nor logged; span_loss, selfsup_loss
        and ortho_loss are.
        Returns span_weight·span + selfsup_weight·selfsup + ortho_weight·ortho."""
        B = G_M.shape[0]
        r_M = span_residual(G_M, m, self.span_ridge)            # [B, n]
        r_A = span_residual(G_A, m, self.span_ridge)
        r = {'mass': r_M, 'energy': r_A, 'both': 0.5 * (r_M + r_A)}[self.span_norm]
        loss_span = torch.sqrt(r + 1e-8) if self.span_root else r
        loss_span = loss_span.mean() if w is None else (loss_span.mean(-1) * w.to(loss_span.dtype)).mean()

        Gm_vv, Ga_vv = G_M[:, :m, :m], G_A[:, :m, :m]
        if not selfsup:
            loss_ss = None
        elif self.selfsup_form == 'logdet':
            loss_ss = ritz_logdet(Gm_vv, Ga_vv, self.span_ridge).mean()
        else:
            loss_ss = -torch.log(ritz_compliance(Gm_vv, Ga_vv, self.span_ridge)
                                 .clamp(min=1e-300)).mean()
        loss_ortho = _basis_conditioning_loss(Gm_vv) if ortho == 'mass' else \
            _basis_conditioning_loss(Ga_vv, rel_floor=1e-12)

        total = self.span_weight * loss_span
        if selfsup and self.selfsup_weight > 0.0:
            total = total + self.selfsup_weight * loss_ss
        if self.ortho_weight > 0.0:
            total = total + self.ortho_weight * loss_ortho

        if prefix is None:
            return total
        kw = dict(on_step=False, on_epoch=True, batch_size=B, sync_dist=True)
        with torch.no_grad():
            self.log(f'{prefix}/span_loss', loss_span.float(), prog_bar=True, **kw)
            if selfsup:
                self.log(f'{prefix}/selfsup_loss', loss_ss.float(), **kw)
            self.log(f'{prefix}/ortho_loss', loss_ortho.float(), **kw)
            if not diag:
                return total
            # Best-approximation rel-L2 of each stored target mode from span(V)
            # (mass norm ≈ lower bound of the post-Ritz mode_k_rel_l2, since the
            # Ritz modes lie in span(V); energy norm ~ the Ritz eigenvalue error).
            rl_M, rl_A = r_M.clamp(min=0).sqrt(), r_A.clamp(min=0).sqrt()
            self.log(f'{prefix}/span_rel_l2', rl_M.mean().float(), **kw)
            self.log(f'{prefix}/span_rel_l2_energy', rl_A.mean().float(), **kw)
            for k in range(rl_M.shape[-1]):
                self.log(f'{prefix}/span_mode_{k}_rel_l2', rl_M[:, k].mean().float(), **kw)
        return total

    def _compute_loss_3d(self, batch, prefix, basis_out=None):
        """Loss of EigenspaceOperatorField (E-field HCurl(p) DOFs).

        Loss = the span / selfsup / ortho terms (_span_terms) on the Grams
        of [PV | T] from hcurl_grams: mass block VV = M_div (kernel-projected,
        so gradient content neither helps nor hurts), curl–curl blocks
        unchanged (K G = 0), T = ALL stored modes; + freq_weight · z-MSE of
        the K Ritz frequencies (fidelity-weighted per sample, docs/34).  Metrics
        (no grad): per-mode M-norm rel-L2 of the Ritz fields vs the K lowest
        targets (sign-agnostic, subspace error inside near-degenerate clusters;
        NaN = excluded for a cluster that the K-th output splits, see
        _clusters_3d), freq MAE [GHz] and relative error, the basis' gradient
        mass fraction 1 − diag(M_div)/diag(VᵀMV), CG iterations.
        basis_out (dict or None): receives the basis V of this pass under 'basis' (test_step of a projection-free run
        reuses it for test/loss_plain instead of running the trunk again).
        """
        from src.models.hcurl import KpSolve, hcurl_grams, mode_rel_l2
        out = self.model(batch)
        V, T = out['basis'], batch['Y_field']
        if basis_out is not None:
            basis_out['basis'] = V
        B, _, m = V.shape
        G_M, G_A, MT = hcurl_grams(V, T, out, batch)
        fid = batch.get('Fidelity', None)                 # mixed LF / HF labels (docs/34)
        w_fid = w_freq = None
        if fid is not None:
            fid = fid.reshape(-1).to(V.device)
            w_fid = self._table(self.fidelity_weights, V.device)[fid]
            w_freq = w_fid * self._table(self.fidelity_freq_scale, V.device)[fid]
        total = self._span_terms(G_M, G_A, m, prefix, w_fid)
        K = out['field'].shape[-1]
        f_pred, f_true = out['freq'], batch['Y_freq'][:, :K]
        loss_freq = F.mse_loss(f_pred, f_true)
        if self.freq_weight > 0.0:
            lf = loss_freq if w_freq is None else ((f_pred - f_true) ** 2).mean(-1).mul(w_freq.to(f_pred.dtype)).mean()
            total = total + self.freq_weight * lf.to(total.dtype)
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
            if fid is not None and self.freq_stats:             # per-fidelity metrics
                fp_g, ft_g = self._to_ghz(f_pred), self._to_ghz(f_true)
                rel_b = ((fp_g - ft_g).abs() / ft_g.abs()).mean(-1)
                for name, code in (('lf', 0), ('hf', 1)):
                    sel = (fid == code).cpu()
                    if bool(sel.any()):
                        self.log(f'{prefix}/freq_rel_err_{name}', rel_b[sel.to(rel_b.device)].mean().float(),
                                 **dict(kw, batch_size=int(sel.sum())))
                        v = torch.isfinite(rl)[sel.to(rl.device)]
                        n = int(v.sum())
                        r_sel = rl[sel.to(rl.device)]
                        self.log(f'{prefix}/field_rel_l2_{name}', torch.where(v, r_sel, torch.zeros_like(r_sel)).sum() / max(n, 1),
                                 **dict(kw, batch_size=n))
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
            emask = batch['DofMask'].unsqueeze(-1).expand_as(Fd)
            preds, targets = (Fd * sgn)[emask].float(), t_unit[emask].float()
        self.log(f'{prefix}/loss', total, on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=B, sync_dist=True)
        self.log(f'{prefix}/field_rel_l2', field_rl, on_step=True, on_epoch=True,
                 prog_bar=True, batch_size=n_valid, sync_dist=True)
        return total, preds, targets

    # a V column whose mass norm² is below this fraction of the curl-bearing columns' largest one is dead in the
    # unprojected loss (amplitude < 1e-3 of them): its loss gradient on V would grow as 1/amplitude otherwise
    COLUMN_FLOOR = 1e-6

    @staticmethod
    def _label_lambda1(G_M, G_A, m):
        """[B, 1] smallest Rayleigh quotient diag(TᵀKT)/diag(TᵀMT) of the stored targets T (the lowest nonzero eigenvalue
        of the model space for p2 labels on the model mesh); +inf when no target has mass."""
        tm, ta = (torch.diagonal(G, dim1=-2, dim2=-1)[:, m:] for G in (G_M, G_A))
        return torch.where(tm > 0, ta / tm.clamp(min=1e-300), torch.full_like(tm, float('inf'))).amin(-1, keepdim=True)

    @staticmethod
    def _normalise_columns(G_M, G_A, m, floor=COLUMN_FLOOR):
        """Scale the V columns of the unprojected Grams [B, m+n, m+n] to unit mass norm (diag VᵀMV = 1) and kill the
        collapsed ones; a null column keeps 0.  span_residual is invariant to column scaling, but its dead-column
        rule (diag ≤ 1e-12 · max, _jacobi_cholesky) is not, and in the unprojected mass Gram a gradient-heavy column
        of huge norm would turn the clean columns "dead": hence the unit scaling.  Because the loss is invariant to
        the scaling, the unit scaling alone does not tame a column that collapses towards zero amplitude: its loss
        gradient on V grows as 1/‖V_i‖ (2.4e6 at 1e-9 relative amplitude, 1e16 at 1e-20), and with gradient clipping
        that one column shrinks every other update.  So a column with d_i ≤ floor · d_ref is dead (scale 0: no span
        or ortho gradient, as the Jacobi rule does in the projected path), which bounds the loss gradient of a live
        column by ~1/√floor = 1e3 times a column at the reference amplitude.  d_ref = the largest mass norm² among the
        curl-bearing columns (Rayleigh quotient ≥ 0.1 λ₁, λ₁ from _label_lambda1; all columns when none is): a
        gradient-heavy column (RQ ≈ 0) does not set the reference, so clean columns next to a 1e7× gradient column
        stay alive.  A live column's gradient is still ∝ 1/amplitude (a scale-invariant loss), so a scaling bound
        s = 1/√max(d, floor·d_max) alone does NOT bound it (probed: ratio 1/amplitude, not 1e3).  Returns
        (G_M, G_A, ratio) with ratio [B] = max/min of the original mass diagonal over the columns (capped at
        1e30, detached)."""
        d = torch.diagonal(G_M, dim1=-2, dim2=-1)[:, :m]
        with torch.no_grad():
            dd, da = d.detach(), torch.diagonal(G_A, dim1=-2, dim2=-1)[:, :m].detach()
            lam = CavityLightning._label_lambda1(G_M.detach(), G_A.detach(), m)
            curl = (dd > 0) & (da >= 0.1 * lam * dd)
            d_ref = torch.where(curl, dd, torch.zeros_like(dd)).amax(-1, keepdim=True)
            d_ref = torch.where(d_ref > 0, d_ref, dd.amax(-1, keepdim=True))
            alive = (dd > 0) & (dd > floor * d_ref)
        s = torch.where(alive, d.clamp(min=1e-300).rsqrt(), torch.zeros_like(d))
        S = torch.cat([s, torch.ones(G_M.shape[0], G_M.shape[1] - m, dtype=d.dtype, device=d.device)], -1)
        scale = S.unsqueeze(-1) * S.unsqueeze(-2)
        dmax = d.detach().amax(-1)
        ratio = dmax / d.detach().amin(-1).clamp(min=1e-30 * dmax).clamp(min=1e-300)
        return G_M * scale, G_A * scale, ratio

    @staticmethod
    def _grad_frac_lower_bound(G_M, G_A, m):
        """Rigorous lower bound of the gradient mass fraction of each V column, from the unscaled unprojected Grams
        [B, m+n, m+n] of [V | T] (no projection, no Kp solve), [B, m]: max(0, 1 − RQ_i/λ₁) with
        RQ_i = (VᵀKV)_ii / (VᵀMV)_ii the Rayleigh quotient of column i (scale invariant) and λ₁ = min_k
        (TᵀKT)_kk / (TᵀMT)_kk the smallest label Rayleigh quotient of the sample (the lowest nonzero eigenvalue).
        With V_i = c + g, c ∈ range(P) (M-orthogonal to the gradients), g = (I − P)V_i: K g = 0 and (c, g)_M = 0, so
        RQ_i = cᵀKc / (cᵀMc + gᵀMg) and cᵀKc ≥ λ₁ cᵀMc for every divergence-free c of the model space; hence
        1 − gᵀMg/‖V_i‖² = cᵀMc/‖V_i‖² ≤ RQ_i/λ₁ and the gradient fraction gᵀMg/‖V_i‖² = 1 − diag(M_div)/diag(M_V) ≥
        1 − RQ_i/λ₁.  Equality when c is a combination of the lowest eigenmodes (RQ_c = λ₁), otherwise a bound: a
        column with high-frequency curl content (RQ_c ≫ λ₁) can hide any amount of gradient mass.  The bound is
        exact in λ₁ for p2 labels on the model mesh (the stored lowest mode is the lowest nonzero eigenpair of the
        model space); for p3 labels projected onto p2 it is the Rayleigh quotient of the projected mode (≥ the p2
        eigenvalue by an amount that was not measured): the bound then holds only up to that factor.  A zero
        column counts as 0.  Needs T to hold the lowest eigenmode."""
        dM, dA = (torch.diagonal(G, dim1=-2, dim2=-1) for G in (G_M, G_A))
        rq = dA[:, :m] / dM[:, :m].clamp(min=1e-300)
        lam = CavityLightning._label_lambda1(G_M, G_A, m)
        frac = (1.0 - rq / lam).clamp(min=0.0, max=1.0)
        return torch.where((dM[:, :m] > 0) & torch.isfinite(lam), frac, torch.zeros_like(frac))

    def _compute_loss_3d_plain(self, batch, prefix='train', basis=None, diag=True):
        """Projection-free loss (projection=False, docs/38), the SAME function for the train and the validation step:
        span_weight · span + ortho_weight · ortho on the Grams of [V | T] WITHOUT the kernel projection
        (hcurl_grams_plain: mass VV = VᵀMV, cross VᵀMT, energy unchanged), the V columns scaled to unit mass norm
        (_normalise_columns).  The stored targets T are
        discrete eigenmodes, M-orthogonal to range(G) (GᵀKT = 0 ⇒ GᵀMT = 0 for λ > 0), so the cross
        blocks equal the projected ones and only the VV mass block differs: gradient content of V costs
        mass-norm span error (r_plain ≥ r_proj) and the energy norm does not see it.  No Kp CG, no
        Rayleigh–Ritz, no frequencies: the Ritz-based metrics (freq errors, field_rel_l2, grad_frac,
        kp_cg_iters, R²) exist for the test split only.  ortho is measured on the curl–curl Gram (gradient-blind:
        the mass correlations of columns with independent gradient parts shrink at constant span, which
        would reward gradient content), mass_diag_ratio is logged.
        Logged under `prefix` ('train' / 'val'): loss, span_loss, ortho_loss, span_rel_l2, span_rel_l2_energy,
        span_mode_k_rel_l2, mass_diag_ratio, grad_frac_lb, grad_frac_lb_max (_grad_frac_lower_bound: a rigorous
        lower bound of the gradient mass fraction of the columns, mean / max over columns) — every key on every
        step (DDP).  prefix None logs nothing (test_step's loss_plain).  basis: the V of an already computed forward
        pass (test_step reuses the projected pass's), else the trunk runs here.  Needs no G / Kp in the batch.
        diag=False (training step on the batches that skip the diagnostics, diag_metrics_every_n_steps > 1): the loss
        and span_loss / ortho_loss / loss are computed and logged as before; span_rel_l2, span_rel_l2_energy,
        span_mode_k_rel_l2, mass_diag_ratio, grad_frac_lb and grad_frac_lb_max are neither computed nor logged.
        No host synchronisation in this function (docs/38 §11; tests/test_throughput.py counts them).
        Returns the loss."""
        from src.models.hcurl import hcurl_grams_plain
        V, T = (self.model.forward_basis(batch)['basis'] if basis is None else basis), batch['Y_field']
        self._mark('trunk')
        B, _, m = V.shape
        G_M, G_A, _ = hcurl_grams_plain(V, T, batch)
        self._mark('grams')
        if prefix is not None and diag:
            with torch.no_grad():
                gf = self._grad_frac_lower_bound(G_M.detach(), G_A.detach(), m)
        G_M, G_A, ratio = self._normalise_columns(G_M, G_A, m)
        fid = batch.get('Fidelity', None)                 # mixed LF / HF labels (docs/34)
        w_fid = None
        if fid is not None:
            w_fid = self._table(self.fidelity_weights, V.device)[fid.reshape(-1).to(V.device)]
        total = self._span_terms(G_M, G_A, m, prefix, w_fid, selfsup=False, ortho='energy', diag=diag).float()
        if prefix is None:
            return total
        self._mark('loss')
        if diag:
            with torch.no_grad():       # gradient-content watch: columns whose VᵀMV norm dwarfs the others' are mostly gradient
                self.log(f'{prefix}/mass_diag_ratio', ratio.mean().float(), on_step=False, on_epoch=True,
                         batch_size=B, sync_dist=True)
                self.log(f'{prefix}/grad_frac_lb', gf.mean().float(), on_step=False, on_epoch=True, batch_size=B,
                         sync_dist=True)
                self.log(f'{prefix}/grad_frac_lb_max', gf.amax().float(), on_step=False, on_epoch=True, batch_size=B,
                         sync_dist=True, reduce_fx='max')
        self.log(f'{prefix}/loss', total, on_step=True, on_epoch=True, prog_bar=True, batch_size=B, sync_dist=True)
        self._mark('log')
        return total

    def _to_ghz(self, f):
        """z-scored frequency → GHz (float64); unchanged when there are no freq_stats."""
        f = f.double()
        if self.freq_stats:
            f = f * float(self.freq_stats['std']) + float(self.freq_stats['mean'])
        return f

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

    def _diag_step(self, batch_idx):
        """True on the training batches that compute the diagnostic metrics (diag_metrics_every_n_steps): batch_idx is
        the same on every DDP rank, so every rank logs the same keys on the same steps."""
        return self.diag_every <= 1 or batch_idx % self.diag_every == 0

    def training_step(self, batch, batch_idx):
        batch = gpu_transfer.wait_ready(batch)
        if self.projection:
            loss, preds, targets = self._compute_loss_3d(batch, "train")
            self.train_r2(preds, targets)
            self.log('train/r2', self.train_r2, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        else:                       # projection-free step (docs/38): no field, no R²; validation still projects
            loss = self._compute_loss_3d_plain(batch, "train", diag=self._diag_step(batch_idx))
        self.log('lr', self._lr_tensor(self.optimizers().param_groups[0]['lr']), prog_bar=True, on_step=True,
                 on_epoch=False)
        self._mark('log')
        return loss

    def validation_step(self, batch, batch_idx):
        batch = gpu_transfer.wait_ready(batch)
        if not self.projection:         # docs/38: the train loss on the validation set; no Ritz fields, no R² / MAE
            return self._compute_loss_3d_plain(batch, "val")
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
        # We use a dict of {mode_idx: value}.  projection=False has no Ritz fields: the per-mode
        # best-approximation (span) errors take their place (docs/38).
        pre = 'mode_' if self.projection else 'span_mode_'
        train_errors = {
            k.split(pre)[1].split('_')[0]: v.item() if torch.is_tensor(v) else v
            for k, v in metrics.items() if k.startswith(f'train/{pre}') and k.endswith('_rel_l2')
        }
        val_errors = {
            k.split(pre)[1].split('_')[0]: v.item() if torch.is_tensor(v) else v
            for k, v in metrics.items() if k.startswith(f'val/{pre}') and k.endswith('_rel_l2')
        }

        mode_indices = sorted(list(set(train_errors.keys()) | set(val_errors.keys())))
        if not mode_indices:
            return

        # Header
        print(f"\n{'━'*64}")
        print(f"  Epoch {epoch:>3d} │ Mode-Specific Relative L2 {'Field' if self.projection else 'Span (best-approximation)'} Error")
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
        if self.projection:
            # Try both direct name and _epoch suffix (Lightning adds _epoch when on_step=True)
            global_val_l2 = metrics.get('val/field_rel_l2_epoch', metrics.get('val/field_rel_l2', 0.0))
            global_val_r2 = metrics.get('val/r2_epoch', metrics.get('val/r2', 0.0))
            print(f"  GLOBAL VAL  │  Rel L2: {global_val_l2:.4f}  │  R²: {global_val_r2:.4f}")
        else:
            val_loss = metrics.get('val/loss_epoch', metrics.get('val/loss', 0.0))
            val_span = metrics.get('val/span_loss', 0.0)
            val_rl2 = metrics.get('val/span_rel_l2', 0.0)
            ratio = metrics.get('val/mass_diag_ratio', 0.0)
            gf = metrics.get('val/grad_frac_lb', 0.0)
            print(f"  GLOBAL VAL  │  span loss (selected): {float(val_span):.4f}  │  loss: {float(val_loss):.4f}  │  span rel L2: "
                  f"{float(val_rl2):.4f}  │  mass diag ratio: {float(ratio):.3g}  │  grad frac >= {float(gf):.3f}")
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
        batch = gpu_transfer.wait_ready(batch)
        shared = {}
        loss, preds, targets = self._compute_loss_3d(batch, "test", shared)     # always projected: the real quality
        if not self.projection:         # the training loss on the test set, next to the projected one (docs/38 §1.2)
            self.log('test/loss_plain', self._compute_loss_3d_plain(batch, None, shared['basis']).float(),
                     on_step=False, on_epoch=True, batch_size=batch['Y_field'].shape[0], sync_dist=True)
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
                    "monitor": monitor_metric(self.projection),
                    "interval": "epoch",
                    # Step only on epochs that actually ran validation, else
                    # Lightning raises "monitor ... not available"
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
