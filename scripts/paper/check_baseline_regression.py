"""Parametric-regression baselines (referee item E8, docs/35 check 3a).

Gaussian-process regression (scikit-learn; RBF / Matern-5/2 with ARD length scales + white noise) and a small MLP
on the geometry parameters (attrs geom_params of each sample: R, Lc, rp, Lpipe, rho_iris, rho_out) predicting the
K lowest frequencies.  Trained on the frequency labels of a training file (LF 40 / LF 24 / LF 8 / HF 8), evaluated on
the HF test geometries with the statistics of the paper: mean / median / max of |f / f_HF - 1| over
(geometries x K modes), plus the per-geometry mean.  Inputs are standardised; the "log" variants regress log f on
log parameters (f scales like 1/size).  A variant is chosen by 5-fold cross-validation on the TRAINING set only; all
variants are printed so the choice can be audited.

    python scripts/paper/check_baseline_regression.py --test hf_test.h5 --train lf40=lf_train.h5
        --train lf24=lf_train.h5:0-23 --train lf8=lf_train.h5:0-7 --train hf8=hf_train.h5 [--json out.json]

A --train spec is  name=file.h5[:lo-hi]  (sample ids lo..hi inclusive, from the key sample_NNNN).
"""
import argparse
import json
import warnings

import h5py
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

PARAMS = ("R", "Lc", "rp", "Lpipe", "rho_iris", "rho_out")


def load(spec, K):
    """(ids, X [n, len(PARAMS)], F [n, K] GHz) of 'file.h5[:lo-hi]'."""
    path, _, rng = spec.partition(":")
    lo, hi = (int(v) for v in rng.split("-")) if rng else (0, 10 ** 9)
    ids, X, F = [], [], []
    with h5py.File(path, "r") as f:
        for key in sorted(k for k in f if k.startswith("sample_")):
            i = int(key.split("_")[1])
            if lo <= i <= hi:
                gp = json.loads(f[key].attrs["geom_params"])
                ids.append(i)
                X.append([gp[p] for p in PARAMS])
                F.append(np.asarray(f[key]["freqs"])[:K])
    return np.array(ids), np.array(X), np.array(F)


class Regressor:
    """fit(X, F) / predict(X) -> F for one variant (standardised inputs and targets, optional log transform)."""

    def __init__(self, kind, log, seed=0):
        self.kind, self.log, self.seed = kind, log, seed

    def _tx(self, X):
        return np.log(X) if self.log else X

    def fit(self, X, F):
        self.sx, self.sy = StandardScaler().fit(self._tx(X)), StandardScaler().fit(self._tx(F))
        Xs, Ys = self.sx.transform(self._tx(X)), self.sy.transform(self._tx(F))
        d = X.shape[1]
        if self.kind in ("gp_rbf", "gp_matern"):
            base = RBF(np.ones(d), (1e-2, 1e2)) if self.kind == "gp_rbf" else Matern(np.ones(d), (1e-2, 1e2), nu=2.5)
            k = ConstantKernel(1.0, (1e-2, 1e2)) * base + WhiteKernel(1e-4, (1e-9, 1e-1))
            self.m = GaussianProcessRegressor(k, normalize_y=False, n_restarts_optimizer=8, random_state=self.seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                self.m.fit(Xs, Ys)
        elif self.kind == "mlp":
            self.ms = []
            for s in range(5):                                     # ensemble over seeds
                m = MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh", solver="lbfgs", alpha=1e-4, max_iter=5000,
                                 tol=1e-10, random_state=self.seed + s)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    m.fit(Xs, Ys)
                self.ms.append(m)
        else:
            raise ValueError(self.kind)
        return self

    def predict(self, X):
        Xs = self.sx.transform(self._tx(X))
        Ys = self.m.predict(Xs) if self.kind != "mlp" else np.mean([m.predict(Xs) for m in self.ms], 0)
        Y = self.sy.inverse_transform(Ys)
        return np.exp(Y) if self.log else Y


VARIANTS = [("gp_rbf", False), ("gp_matern", False), ("gp_rbf", True), ("gp_matern", True), ("mlp", False),
            ("mlp", True)]


def stats(err):
    a = np.abs(err).ravel()
    return {"mean": float(a.mean()), "median": float(np.median(a)), "max": float(a.max())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", required=True, help="HF test H5 (frequency reference)")
    ap.add_argument("--train", action="append", required=True, help="name=file.h5[:lo-hi]")
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    ids_t, Xt, Ft = load(args.test, args.K)
    out = {"test_ids": ids_t.tolist(), "params": list(PARAMS), "runs": []}
    for spec in args.train:
        name, _, src = spec.partition("=")
        ids, X, F = load(src, args.K)
        lo, hi = X.min(0), X.max(0)
        outside = [[PARAMS[j] for j in range(len(PARAMS)) if not lo[j] <= x[j] <= hi[j]] for x in Xt]
        print(f"\n== train {name}: {len(ids)} samples from {src}; test params outside the training box: "
              f"{dict(zip(ids_t.tolist(), outside))}", flush=True)
        rows = []
        for kind, log in VARIANTS:
            cv = []
            for tr, va in KFold(5, shuffle=True, random_state=0).split(X):
                if len(tr) < 4:
                    break
                cv.append(np.abs(Regressor(kind, log).fit(X[tr], F[tr]).predict(X[va]) / F[va] - 1).ravel())
            cv_mean = float(np.concatenate(cv).mean()) if cv else float("nan")
            P = Regressor(kind, log).fit(X, F).predict(Xt)
            err = P / Ft - 1
            r = {"variant": f"{kind}{'_log' if log else ''}", "cv_mean": cv_mean, **stats(err),
                 "per_geometry_mean": np.abs(err).mean(1).tolist()}
            rows.append(r)
            print(f"  {r['variant']:<14} CV(train) mean {100 * cv_mean:7.3f} % | test mean/median/max "
                  f"{100 * r['mean']:6.3f} / {100 * r['median']:6.3f} / {100 * r['max']:6.3f} % | per geometry "
                  f"{' '.join(f'{100 * v:5.2f}' for v in r['per_geometry_mean'])}", flush=True)
        best = int(np.nanargmin([r["cv_mean"] for r in rows]))
        print(f"  selected by training CV: {rows[best]['variant']}")
        out["runs"].append({"train": name, "n": len(ids), "outside_box": outside, "variants": rows,
                            "selected": rows[best]["variant"]})
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
