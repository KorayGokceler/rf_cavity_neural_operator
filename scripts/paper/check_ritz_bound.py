"""Ritz-bound stress test (referee item M3, docs/35 check 2).

The model's Rayleigh-Ritz layer (src.models.hcurl.hcurl_ritz: CG kernel projection + whitened Ritz problem
with dropped gradient directions) must return theta_k >= lambda_h,k.  After training the margins are 0.5-3 %,
which says nothing about the inexact CG projection (paper/sections/A04_method.tex).  This script feeds the real
code path synthetic bases V on a stored LF geometry (whose u_model / rq_model are the exact discrete p2
eigenpairs, so lambda_h,k is known to rounding) and reports, for every CG tolerance tau:

    min_k (theta_k / lambda_h,k - 1), the number of violations theta_k < lambda_h,k (1 - 1e-12),
    eta_bar  (a posteriori, A04: E = -sym(Z^T rho) with the EXACT Z from a sparse direct solve of Kp,
              eta_bar = max |eig(W^T D E D W)| over the kept Ritz subspace; W^T D M_div D W = I),
    min_k theta_k (1 + eta_bar) / lambda_h,k - 1  (the guarantee of A04, must be >= 0),
    the number of dropped directions (mu <= drop_tol), CG iterations, and the number of spurious
    modes WITHOUT the projection (Ritz on span(V) with V^T K V c = theta V^T M V c, theta < 0.5 lambda_h,1).

Bases (m = 12 columns, wall rows 0, random columns = white noise per free DOF, unit M-norm):
    a  first K exact eigenvectors + random columns
    b  exact eigenvectors + relative noise eps (M-norm) + random columns
    c  random V
    d  exact eigenvectors + gradient admixture G r of mass fraction gamma + random columns
    g  (extra) exact eigenvectors + m - K pure gradient columns G r

    python scripts/paper/check_ritz_bound.py --data lf_test.h5 --key sample_0024 [--seeds 3] [--json out.json]
"""
import argparse
import json
import os
import sys
import time
import warnings

import torch  # noqa: F401  (before ngsolve: tests/conftest.py)

import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data.field_dataset import FieldDataset, field_collate, field_item  # noqa: E402
from src.models import hcurl  # noqa: E402

VIOL = 1e-12            # theta_k < lambda_k (1 - VIOL) counts as a violation


class Setup:
    """One stored geometry: dense-free operators of the batch (as FieldDataset.load builds them) + exact data."""

    def __init__(self, path, key, K):
        ds = FieldDataset(path, split="test", train_ratio=0.0, val_ratio=0.0, n_modes=K, with_ct=False)
        g_id = [i for i, (_, k) in enumerate(ds.index) if k == key][0]
        ops, rec = ds.load(g_id)
        item = field_item(ops, rec["u_model"][:, :K], rec["freqs"][:K], None, "pillbox_pipes", check=False)
        self.batch = field_collate([item])
        self.K = K
        self.ops = ops
        self.scale = float(ops["scale"])
        self.lam = rec["rq_model"][:K] * self.scale ** 2        # lambda_h,k in the normalised geometry
        self.U = rec["u_model"][:, :K].astype(np.float64)
        self.M, self.Kmat, self.G, self.Kp = ops["M"].tocsr(), ops["K"].tocsr(), ops["G"].tocsr(), ops["Kp"].tocsc()
        self.free = ~np.asarray(ops["bnd"], bool)
        self.n = int(ops["n_dof"])
        self.Kp_lu = spla.splu(self.Kp)
        self.name = key

    def mnorm(self, X):
        return np.sqrt(np.einsum("ik,ik->k", X, self.M @ X))

    def random_cols(self, rng, k):
        X = rng.standard_normal((self.n, k)) * self.free[:, None]
        return X / self.mnorm(X)

    def gradient_cols(self, rng, mass):
        r = rng.standard_normal((self.G.shape[1], mass.size))
        g = self.G @ r
        return g / self.mnorm(g) * np.sqrt(mass)

    def basis(self, case, rng, m, par):
        U = self.U
        pad = lambda cols: np.column_stack([cols, self.random_cols(rng, m - cols.shape[1])])   # noqa: E731
        if case == "a":
            return pad(U)
        if case == "b":
            xi = self.random_cols(rng, U.shape[1])
            return pad(U + par * self.mnorm(U) * xi)
        if case == "g":                                   # exact eigenvectors + pure gradient columns
            return np.column_stack([U, self.gradient_cols(rng, np.ones(m - U.shape[1]))])
        if case == "c":
            return self.random_cols(rng, m)
        if case == "d":
            mass = par / (1.0 - par) * self.mnorm(U) ** 2     # gradient mass fraction par of u + g
            return pad(U + self.gradient_cols(rng, mass))
        raise ValueError(case)


def whitening(proj, mass_ridge, drop_tol):
    """Steps 1-3 of hcurl.projected_eigh: (D, W, mu, keep) for the stored Grams (B = 1)."""
    M_V, M_div = proj["M_V"][0], proj["M_div"][0]
    d = torch.diagonal(M_V).clamp(min=1e-300).rsqrt()
    Dm = lambda A: A * d[:, None] * d[None, :]       # noqa: E731
    eye = torch.eye(M_V.shape[-1], dtype=M_V.dtype)
    C = torch.linalg.cholesky(Dm(M_V) + mass_ridge * eye)
    Ci = torch.linalg.solve_triangular(C, eye, upper=False)
    S = Ci @ Dm(M_div) @ Ci.T
    mu, Q = torch.linalg.eigh(0.5 * (S + S.T))
    keep = mu > drop_tol
    W = (Ci.T @ Q)[:, keep] * mu[keep].rsqrt()[None, :]
    return d, W, mu, keep


def run(S, V, tau, args):
    """One hcurl_ritz call + diagnostics -> dict."""
    Vt = torch.from_numpy(V).unsqueeze(0)
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        theta, _, proj = hcurl.hcurl_ritz(Vt, S.batch, S.K, args.mass_ridge, args.drop_tol, args.eig_broadening,
                                          tau, args.maxiter)
        t_ritz = time.perf_counter() - t0
    unconverged = any("CG stopped at maxiter" in str(w.message) for w in wl)
    theta = theta[0].numpy()
    ratio = theta / S.lam - 1.0
    d, W, mu, keep = whitening(proj, args.mass_ridge, args.drop_tol)
    # exact Z by a sparse direct solve; E = -sym(Z^T rho) with rho = Kp Z~ - B for the CG's Z~
    Bm = S.G.T @ (S.M @ V)
    Zt = proj["Z"][0].numpy()
    Zex = S.Kp_lu.solve(Bm)
    rho = S.Kp @ Zt - Bm
    E = -0.5 * (Zex.T @ rho + rho.T @ Zex)
    Wn, dn = W.numpy(), d.numpy()
    Ew = Wn.T @ (dn[:, None] * E * dn[None, :]) @ Wn
    eta_bar = float(np.abs(np.linalg.eigvalsh(0.5 * (Ew + Ew.T))).max()) if Wn.shape[1] else 0.0
    # Ritz values with the EXACT projection (same drop rule), for reference
    Md = proj["M_V"][0].numpy() - Bm.T @ Zex
    M_div_ex = torch.from_numpy(0.5 * (Md + Md.T)).unsqueeze(0)
    th_ex, _ = hcurl.projected_eigh(proj["M_V"], M_div_ex, proj["A_V"], S.K, args.mass_ridge, args.drop_tol,
                                    args.eig_broadening)
    th_ex = th_ex[0].numpy()
    # without projection: Ritz on span(V)
    A_V, M_Vm = proj["A_V"][0].numpy(), proj["M_V"][0].numpy()
    dd = 1.0 / np.sqrt(np.diag(M_Vm))
    w_np = sla.eigh(dd[:, None] * A_V * dd[None, :], dd[:, None] * M_Vm * dd[None, :] + args.mass_ridge * np.eye(len(dd)),
                    eigvals_only=True)
    return {"tau": tau, "theta_over_lam_min": float(ratio.min()), "ratio": ratio.tolist(),
            "violations": int((theta < S.lam * (1 - VIOL)).sum()),
            "eta_bar": eta_bar, "guarantee_min": float((theta * (1 + eta_bar) / S.lam - 1).min()),
            "guarantee_holds": bool(((theta * (1 + eta_bar) / S.lam - 1) >= -1e-12).all()),
            "dropped": int((~keep).sum()), "mu_min": float(mu.min()), "mu_kept_min": float(mu[keep].min()) if keep.any() else 0.0,
            "cg_iters": int(hcurl.KpSolve.last_iters), "cg_unconverged": bool(unconverged),
            "exact_proj_min": float((th_ex / S.lam - 1).min()),
            "exact_proj_violations": int((th_ex < S.lam * (1 - VIOL)).sum()),
            "spurious_no_projection": int((w_np < 0.5 * S.lam[0]).sum()),
            "spurious_with_projection": int(((theta < 0.5 * S.lam[0])).sum()), "t_ritz_s": t_ritz}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="LF field-label H5 (model_order == label_order)")
    ap.add_argument("--key", default="sample_0024")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--m", type=int, default=12)
    ap.add_argument("--taus", type=float, nargs="+", default=[1e-4, 1e-6, 1e-8, 1e-10])
    ap.add_argument("--eps", type=float, nargs="+", default=[0.0, 1e-8, 1e-6, 1e-4])
    ap.add_argument("--gammas", type=float, nargs="+", default=[1e-6, 1e-4, 1e-2, 0.5, 0.9, 0.99])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--maxiter", type=int, default=3000, help="configs/field_3d.yaml: kp_maxiter")
    ap.add_argument("--mass_ridge", type=float, default=1e-8)
    ap.add_argument("--drop_tol", type=float, default=1e-6)
    ap.add_argument("--eig_broadening", type=float, default=1e-4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    import ngsolve
    ngsolve.SetNumThreads(args.threads)
    S = Setup(args.data, args.key, args.K)
    print(f"{args.key}: n_dof {S.n}, n_pot {S.G.shape[1]}, lambda_h,k (normalised) {np.round(S.lam, 4).tolist()}",
          flush=True)
    cases = [("a", None)] + [("b", e) for e in args.eps] + [("c", None)] + [("d", g) for g in args.gammas] + [("g", None)]
    rows = []
    for case, par in cases:
        for seed in range(args.seeds):
            rng = np.random.default_rng([7, seed])
            V = S.basis(case, rng, args.m, par)
            for tau in args.taus:
                r = run(S, V, tau, args)
                r.update(case=case, par=par, seed=seed)
                rows.append(r)
                print(f"case {case:<3} par {str(par):<7} seed {seed} tau {tau:.0e} | min(theta/lam-1) "
                      f"{r['theta_over_lam_min']:+.3e} viol {r['violations']} | eta_bar {r['eta_bar']:.2e} "
                      f"guar {r['guarantee_min']:+.2e} | dropped {r['dropped']} | it {r['cg_iters']}"
                      f"{' UNCONVERGED' if r['cg_unconverged'] else ''} | exact-proj min {r['exact_proj_min']:+.2e} "
                      f"| spurious w/o proj {r['spurious_no_projection']} with {r['spurious_with_projection']}",
                      flush=True)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"args": vars(args), "lam": S.lam.tolist(), "rows": rows}, fh, indent=1)


if __name__ == "__main__":
    main()
