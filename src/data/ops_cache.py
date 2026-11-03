"""On-disk format of the FieldDataset operator cache (docs/38 §9): one file per geometry holding the
`field_operators` dict (without bary / X / tets) and the stored label record.

Compared with a plain pickle of the scipy matrices (int + float64 per entry):
  - K, M (exactly symmetric, assembled from one triangle) are stored as their upper triangle, the full
    matrix is rebuilt on load entry for entry; the writer checks this bit by bit and stores a matrix in
    full when it is not exactly reproducible (Kp = GᵀMG is not bit-symmetric and is stored in full);
  - integer arrays are int32 (int64 only when a dimension or nnz needs it; asserted), CSR indices and
    indptr are stored as differences, which zlib shrinks ~5–10× (CSR rows are runs of consecutive columns);
  - every array goes through zlib (level 1) only if a trial on its first MiB saves ≥ 10 %: the K values
    (76 % structural zeros) shrink 4×, the incompressible ones stay raw and load without decompression;
  - `narrow=True` (the projection-free train / val variant, which has no G / Kp) stores the values of K, M, C
    and the point arrays P, qx, qw as float32 and casts back to float64 on load.  field_item turns P, qx, qw
    into float32 anyway (storing them narrow changes no item); K, M, C values are rounded to ~6·10⁻⁸ relative.
    narrow=False (the full variant, with G, Kp for the kernel projection and its float64 Kp CG) keeps every
    value in float64, bit for bit.
Files are versioned (FORMAT, the name suffix `.ops2.pkl`): a file of another format is never read as this one.
"""
import os
import pickle
import uuid
import zlib

import numpy as np
import scipy.sparse as sp

FORMAT = 2
SUFFIX = ".ops2.pkl"
NARROW_KEYS = ("P", "qx", "qw")               # dense arrays stored as float32 in the narrow variant
_I32 = np.iinfo(np.int32).max
_TRIAL = 1 << 20                              # bytes compressed to decide whether an array is worth zlib
_GAIN = 0.9                                   # keep zlib only if the trial shrinks the data below this


class OpsCacheError(ValueError):
    """The file is unreadable, truncated or of another format: the caller rebuilds the operators."""


def index_dtypes(n_rows, n_cols, nnz):
    """(indptr dtype, indices dtype) of a CSR matrix: int32 whenever the values fit, asserted, else int64."""
    for v in (n_rows, n_cols, nnz):
        if v < 0 or v > np.iinfo(np.int64).max:
            raise OverflowError(f"sparse dimension {v} out of range")
    # indices: columns < n_cols, and their differences (the stored form) are < n_cols in magnitude
    return (np.dtype(np.int32) if nnz <= _I32 else np.dtype(np.int64),
            np.dtype(np.int32) if max(n_cols, n_rows) <= _I32 else np.dtype(np.int64))


def _enc(a, store=None, delta=False):
    """Packed array: dict(dtype, orig, shape, delta, codec, buf).  store: the dtype written (default: as is)."""
    a = np.asarray(a)
    orig = a.dtype
    st = np.dtype(store) if store is not None else orig
    if delta:
        if a.ndim != 1 or st.kind not in "iu":
            raise TypeError("delta coding is for 1-d integer arrays")
        d = np.diff(a.astype(np.int64), prepend=0)
        if d.size and (d.max() > np.iinfo(st).max or d.min() < np.iinfo(st).min):
            raise OverflowError("index differences do not fit the storage dtype")
        a = d
    if st.kind == "f" and st.itemsize < orig.itemsize and a.size and np.nanmax(np.abs(a)) > np.finfo(st).max:
        raise OverflowError(f"values beyond the largest {st} number cannot be stored narrow")
    b = np.ascontiguousarray(a.astype(st, copy=False))
    raw = b.tobytes()
    codec = "raw"
    if b.nbytes >= 4096:
        n = min(len(raw), _TRIAL)
        if len(zlib.compress(raw[:n], 1)) < _GAIN * n:
            z = zlib.compress(raw, 1)
            if len(z) < _GAIN * len(raw):
                raw, codec = z, "zlib"
    return {"dtype": st.str, "orig": orig.str, "shape": tuple(b.shape), "delta": bool(delta), "codec": codec,
            "buf": raw}


def _dec(p):
    buf = zlib.decompress(p["buf"]) if p["codec"] == "zlib" else p["buf"]
    a = np.frombuffer(buf, dtype=np.dtype(p["dtype"])).reshape(p["shape"])
    if p["delta"]:
        return np.cumsum(a, dtype=np.dtype(p["orig"]))      # partial sums are the original (in-range) values
    return a.astype(np.dtype(p["orig"]))           # a copy: writable, in the original dtype


def _upper(A):
    """Upper triangle (diagonal included) of a canonical CSR matrix as (indptr, indices, data)."""
    n = A.shape[0]
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(A.indptr))
    keep = A.indices >= rows
    ptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(rows[keep], minlength=n), out=ptr[1:])
    return ptr, A.indices[keep], A.data[keep]


def _expand_upper(ptr, ind, val, n):
    """Full canonical CSR (indptr, sorted indices, data) of the symmetric matrix whose upper triangle
    (diagonal included) is given.  Only index arithmetic (scipy's COO → CSR counting sort and its per-row
    sort): the values are moved, never summed, and explicit zeros stay."""
    rows = np.repeat(np.arange(n, dtype=ind.dtype), np.diff(ptr))
    strict = ind > rows
    A = sp.coo_matrix((np.concatenate([val, val[strict]]),
                       (np.concatenate([rows, ind[strict]]), np.concatenate([ind, rows[strict]]))),
                      shape=(n, n)).tocsr()
    return A.indptr, A.indices, A.data


def _bits_equal(a, b):
    """Same shape, dtype and bit pattern (so -0.0 ≠ +0.0 and NaN payloads count)."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    u = {4: np.uint32, 8: np.uint64}.get(a.dtype.itemsize)
    return np.array_equal(a.view(u), b.view(u)) if u and a.dtype.kind == "f" else np.array_equal(a, b)


def _pack_csr(A, narrow, allow_sym):
    A = sp.csr_matrix(A)
    m, n = A.shape
    ipt, ixt = index_dtypes(m, n, A.nnz)
    vst = np.float32 if narrow and A.data.dtype == np.float64 else None
    sym = False
    if allow_sym and m == n and A.has_canonical_format:
        ptr, ind, val = _upper(A)
        fp, fi, fv = _expand_upper(ptr, ind, val, m)
        sym = (_bits_equal(fv, A.data) and np.array_equal(fi, A.indices) and np.array_equal(fp, A.indptr))
        if sym:
            ipt, ixt = index_dtypes(m, n, ind.size)
            ptr_, ind_, val_ = ptr, ind, val
    if not sym:
        ptr_, ind_, val_ = A.indptr, A.indices, A.data
    return {"kind": "csr", "shape": (m, n), "sym": sym,
            "indptr": _enc(ptr_, ipt, delta=True), "indices": _enc(ind_, ixt, delta=True),
            "data": _enc(val_, vst)}


def _unpack_csr(p):
    m, n = p["shape"]
    ptr, ind, val = _dec(p["indptr"]), _dec(p["indices"]), _dec(p["data"])
    if p["sym"]:
        ptr, ind, val = _expand_upper(ptr, ind, val, m)
    ptr = ptr.astype(np.int32 if ptr[-1] <= _I32 else np.int64)
    A = sp.csr_matrix((val, ind.astype(ptr.dtype), ptr), shape=(m, n))
    if p["sym"]:
        A.has_sorted_indices = True        # _pack_csr verified the rebuilt matrix equals a canonical original
    return A


def pack(ops, rec, narrow, symmetric=("K", "M")):
    """Cacheable payload of an operator dict + label record (module docstring)."""
    out = {}
    for k, v in ops.items():
        if sp.issparse(v):
            out[k] = _pack_csr(v, narrow, k in symmetric)
        elif isinstance(v, np.ndarray):
            st = np.float32 if narrow and k in NARROW_KEYS and v.dtype == np.float64 else None
            out[k] = {"kind": "array", "p": _enc(v, st)}
        else:
            out[k] = {"kind": "plain", "v": v}
    r = dict(rec)
    u = r.pop("u_model")
    u32 = u.astype(np.float32)           # labels come from an float32 H5 dataset: float32 is then exact
    exact = u.dtype == np.float64 and np.array_equal(u32.astype(np.float64), u)
    return {"format": FORMAT, "narrow": bool(narrow), "potential": "G" in ops and "Kp" in ops,
            "ops": out, "rec": r, "u_model": _enc(u, np.float32 if exact else None)}


def unpack(payload):
    """(ops, rec) of a payload from pack(); arrays and matrices in their original dtypes."""
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise OpsCacheError(f"operator cache format {payload.get('format') if isinstance(payload, dict) else '?'}, "
                            f"this code reads format {FORMAT}")
    try:
        ops = {}
        for k, p in payload["ops"].items():
            ops[k] = _unpack_csr(p) if p["kind"] == "csr" else _dec(p["p"]) if p["kind"] == "array" else p["v"]
        rec = dict(payload["rec"])
        rec["u_model"] = _dec(payload["u_model"])
    except (zlib.error, KeyError) as e:
        raise OpsCacheError(f"corrupt operator cache payload ({type(e).__name__}: {e})") from e
    return ops, rec


def write(path, payload):
    """Atomic: a temporary file next to `path` (unique per process and call), then os.replace; concurrent
    writers (DDP ranks, DataLoader workers) each write a complete file and the last one wins."""
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        with open(tmp, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read(path):
    """Payload of a cache file; OpsCacheError if it is unreadable or not of this format."""
    try:
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
    except (EOFError, pickle.UnpicklingError, AttributeError, ImportError, IndexError, ValueError) as e:
        raise OpsCacheError(f"{path}: unreadable operator cache ({type(e).__name__}: {e})") from e
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise OpsCacheError(f"{path}: operator cache of another format (this code reads format {FORMAT})")
    return payload
