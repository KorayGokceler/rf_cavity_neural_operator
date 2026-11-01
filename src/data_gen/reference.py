"""High-order reference check of stored field labels.

For samples of field-label H5 files: rebuild the stored model mesh (identical numbering, field_labels.read_field_group),
solve HCurl p<order> on it (default p4, highorder.solve_modes) and compare the frequencies with the stored labels.
On the same mesh the p4 spectrum is the labels' discretisation error reference: p4 on a box matched the analytic
spectrum to 3e-8, p3 vs p4 differ by ~6e-7 on a pillbox (docs/37).  Frequencies only; no label is touched.

Used by scripts/reference_check.py (JSON schema: see `run`).  Import torch before ngsolve in the calling process.
"""
import glob
import json
import os
import time

import numpy as np


def find_h5(paths):
    """Sorted field-label files: files, globs or directories (a directory is searched RECURSIVELY for *.h5; pass
    explicit file lists to control what is checked)."""
    out = set()
    for p in paths:
        if os.path.isdir(p):
            out.update(glob.glob(os.path.join(p, "**", "*.h5"), recursive=True))
        else:
            out.update(glob.glob(p) or [p])
    return sorted(out)


def select_samples(f, per_family=1, names=None, all_samples=False, seen=None):
    """Sample keys of an open H5 file: names (as given), all, or the first `per_family` per shape_type.
    seen: dict family → count already picked in earlier files (shards `<family>_sNNNNN.h5`: the quota is shared
    over all files; updated in place)."""
    keys = sorted(k for k in f.keys() if k.startswith("sample_"))
    if names:
        return [n for n in names if n in f]               # run() reports names found in no file
    if all_samples:
        return keys
    seen = {} if seen is None else seen
    out = []
    for k in keys:
        fam = str(f[k].attrs.get("shape_type", ""))
        if seen.get(fam, 0) < per_family:
            seen[fam] = seen.get(fam, 0) + 1
            out.append(k)
    return out


def hcurl_ndof(mesh, order):
    from ngsolve import HCurl
    return int(HCurl(mesh, order=order, dirichlet="wall").ndof)


def reference_frequencies(mesh, K, order, max_ndof):
    """(f_GHz [K], ndof, seconds) of HCurl p<order> on mesh; raises RuntimeError above max_ndof."""
    from src.data_gen import highorder as ho
    t0 = time.perf_counter()
    lab = ho.solve_modes(mesh, K + 1, order=order, max_ndof=max_ndof)
    return np.asarray(lab["f_hz"][:K], float) / 1e9, int(lab["fes"].ndof), time.perf_counter() - t0


def check_sample(g, order=4, max_ndof=700000, fallback_order=False, cross_check=False, allow_other_mesh=False):
    """One record of the JSON schema (see `run`) for the H5 group g.  Labels whose frequencies come from another
    (finer) mesh than the stored model mesh (n_tets_label != n_tets, or adaptive: adapt_iters) are skipped unless
    allow_other_mesh (then est_over_true is null); p3 one-solve labels on the model mesh are checked."""
    from src.data_gen import field_labels as fl
    a = g.attrs
    f_label = np.asarray(g["freqs"], float)
    K = len(f_label)
    rec = {"family": str(a.get("shape_type", "")), "n_tets": int(a["n_tets"]), "label_order": int(a["label_order"]),
           "model_order": int(a["model_order"]),
           "est_rel_f": float(a["est_rel_f"]) if "est_rel_f" in a else None,
           "label_ok": int(a["label_ok"]) if "label_ok" in a else None,
           "ref_order": None, "ref_ndof": None, "f_label_GHz": f_label.tolist(), "f_ref_GHz": None, "rel_err": None,
           "max_rel_err": None, "median_rel_err": None, "est_over_true": None, "t_ref_s": None,
           "ref_self_err": None, "skipped": None}
    other = int(a.get("n_tets_label", a["n_tets"])) != int(a["n_tets"]) or "adapt_iters" in a
    if other and not allow_other_mesh:
        rec["skipped"] = "label frequencies from a different (finer) label mesh"
        return rec
    mesh, _ = fl.read_field_group(g)
    use = None
    for o in (order, order - 1) if fallback_order and order > 1 else (order,):
        nd = hcurl_ndof(mesh, o)
        if nd <= max_ndof:
            use = o
            break
        rec["skipped"] = f"HCurl p{o} has {nd} DOF > max_ndof={max_ndof}"
        rec["ref_ndof"] = nd
    if use is None:
        return rec
    rec["skipped"] = None
    f_ref, ndof, t = reference_frequencies(mesh, K, use, max_ndof)
    rel = np.abs(f_label / f_ref - 1.0)
    rec.update(ref_order=use, ref_ndof=ndof, f_ref_GHz=f_ref.tolist(), rel_err=rel.tolist(),
               max_rel_err=float(rel.max()), median_rel_err=float(np.median(rel)), t_ref_s=float(t))
    if rec["est_rel_f"] is not None and rel.mean() > 0 and not other:
        rec["est_over_true"] = float(rec["est_rel_f"] / rel.mean())      # estimate vs the mean relative error
    if other:
        rec["note"] = "label frequencies from a different (finer) label mesh: rel_err mixes mesh and discretisation"
    if cross_check and use > 1 and use - 1 == rec["label_order"] and not other:
        rec["ref_self_err_note"] = (f"order {use - 1} equals the label order: |p{use - 1} - p{use}| would be the "
                                    "label error, not the reference's")
    elif cross_check and use > 1:                                        # the reference's own error bound
        try:
            f_lo, _, t2 = reference_frequencies(mesh, K, use - 1, max_ndof)
            rec["ref_self_err"] = float(np.abs(f_lo / f_ref - 1.0).max())
            rec["t_ref_s"] += float(t2)
        except RuntimeError as e:
            rec["ref_self_err"] = None
            rec["skipped"] = None
            rec["cross_check_error"] = str(e)
    return rec


def _summary(samples):
    ok = [s for s in samples if not s["skipped"]]
    by_fam = {}
    for s in ok:
        by_fam[s["family"]] = max(by_fam.get(s["family"], 0.0), s["max_rel_err"])
    return {"n": len(samples), "n_skipped": len(samples) - len(ok), "max_rel_err_by_family": by_fam}


def _write_json(path, doc):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=1)
    os.replace(tmp, path)


def run(paths, out=None, order=4, max_ndof=700000, per_family=1, names=None, all_samples=False, fallback_order=False,
        cross_check=False, allow_other_mesh=False, log=print):
    """Reference-check the selected samples of the H5 files under `paths`; returns the result document

    {"files": [...], "settings": {...},
     "samples": [{"file","sample","family","n_tets","label_order","model_order","est_rel_f","label_ok","ref_order",
                  "ref_ndof","f_label_GHz":[..],"f_ref_GHz":[..],"rel_err":[..],"max_rel_err","median_rel_err",
                  "est_over_true","t_ref_s","ref_self_err"|null,"skipped"|null}],
     "summary": {"n","n_skipped","max_rel_err_by_family":{family: max over its samples}}}

    rel_err = |f_label / f_ref − 1| per mode; est_over_true = stored est_rel_f / mean(rel_err); ref_self_err =
    max |f_ref(p−1) / f_ref(p) − 1| (cross_check; p = the reference order actually used, null with a
    ref_self_err_note when p−1 is the label order).  per_family counts over ALL files (shards).  "True" error =
    label − reference on the same mesh: it excludes the curve-3 geometry error common to both solves (docs/29: curve
    2 vs 3 differ by 1.4e-4 on a pillbox).  A sample whose space exceeds max_ndof is listed with `skipped`
    set to the reason (and, with fallback_order, is solved at order − 1 when that fits).  With `out` the JSON is
    rewritten after every sample.  Progress lines "ref <i>/<n> <family> …" go to log."""
    import h5py
    files = find_h5(paths)
    todo, seen = [], {}
    for fn in files:
        with h5py.File(fn, "r") as f:
            todo += [(fn, k, str(f[k].attrs.get("shape_type", ""))) for k in
                     select_samples(f, per_family, names, all_samples, seen)]
    if names and (gone := [n for n in names if not any(n == k for _, k, _ in todo)]):
        raise KeyError(f"samples {gone} are in none of {len(files)} files")
    doc = {"files": files, "settings": {"order": order, "max_ndof": max_ndof, "per_family": per_family,
                                        "samples": list(names) if names else None, "all": bool(all_samples),
                                        "fallback_order": bool(fallback_order), "cross_check": bool(cross_check),
                                        "allow_other_mesh": bool(allow_other_mesh)},
           "samples": [], "summary": _summary([])}
    n = len(todo)
    for i, (fn, key, fam) in enumerate(todo, 1):
        log(f"ref {i}/{n} {fam} {os.path.basename(fn)}/{key} ...")
        try:
            with h5py.File(fn, "r") as f:
                rec = check_sample(f[key], order, max_ndof, fallback_order, cross_check, allow_other_mesh)
        except Exception as e:                              # one bad sample must not kill a long run
            rec = {"family": fam, "skipped": f"error: {type(e).__name__}: {e}"}
        rec = {"file": fn, "sample": key, **rec}
        doc["samples"].append(rec)
        doc["summary"] = _summary(doc["samples"])
        if rec["skipped"]:
            log(f"ref {i}/{n} {fam} skipped: {rec['skipped']}")
        else:
            log(f"ref {i}/{n} {fam} p{rec['ref_order']} {rec['ref_ndof']} DOF  max {rec['max_rel_err']:.2e}  "
                f"{rec['t_ref_s']:.0f} s")
        if out:
            _write_json(out, doc)
    return doc


def format_table(doc):
    """Readable table of a result document."""
    lines = ["error = label vs reference on the SAME curve-3 mesh (excludes the geometry error common to both solves, "
             "docs/29: curve 2 vs 3 = 1.4e-4 on a pillbox)"]
    cols = f"{'file/sample':34s} {'family':16s} {'tets':>6s} {'lp':>2s} {'ref':>3s} {'DOF':>7s} {'max':>9s} " \
           f"{'median':>9s} {'est':>9s} {'est/true':>8s} {'self':>9s} {'t[s]':>6s}"
    lines += [cols, "-" * len(cols)]

    def e(x):
        return f"{x:9.2e}" if x is not None else f"{'-':>9s}"

    for s in doc["samples"]:
        name = f"{os.path.basename(s['file'])}/{s['sample']}"[-34:]
        if s["skipped"]:
            lines.append(f"{name:34s} {s['family']:16s} skipped: {s['skipped']}")
            continue
        r = s["est_over_true"]
        lines.append(f"{name:34s} {s['family']:16s} {s['n_tets']:6d} {s['label_order']:2d} p{s['ref_order']:<2d} "
                     f"{s['ref_ndof']:7d} {e(s['max_rel_err'])} {e(s['median_rel_err'])} {e(s['est_rel_f'])} "
                     f"{r if r is None else round(r, 2)!s:>8s} {e(s['ref_self_err'])} {s['t_ref_s']:6.0f}")
    m = doc["summary"]
    lines.append(f"{m['n']} samples, {m['n_skipped']} skipped; max relative error by family: "
                 + ", ".join(f"{k} {v:.2e}" for k, v in sorted(m["max_rel_err_by_family"].items())))
    return "\n".join(lines)
