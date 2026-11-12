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
  - `pack(..., compress=False)` (dataset.cache_compress: false) writes no zlib and no index differences: every array is
    raw and the index arrays are stored as they are; compress='fast' keeps zlib only for the arrays it shrinks to 30 %
    (K values, index differences) and stores the rest raw.  The reader reads all kinds (every array carries its own `codec` and
    `delta` flag), so files written with and without it, and the files of earlier runs, mix freely and nothing is bumped
    (FORMAT stays 2); `uncompress` turns a payload into the raw form (scripts/recode_ops_cache.py).
Files are versioned (FORMAT, the name suffix `.ops2.pkl`): a file of another format is never read as this one.

The matrix-free variant (dataset.operators: matfree, docs/40 §10) has its own files, `{tag}__{key}.mf1.pkl` (`MF_SUFFIX`,
`MF_FORMAT`): the compact element record of src/data/matfree.py + the token features + the labels as raw arrays of one
pickle (no zlib, no index differences: nothing to decode), never read as an operator cache and vice versa.

Reading (docs/38 §11): the upper triangle of K and M is expanded to the full CSR without a sort (`_upper_pattern`: the
transposed strict part and the stored upper part are merged by scipy's O(nnz) CSR sum), and K and M share the
pattern, so it is computed once per file (`_Patterns`); the values are one gather.  Bit for bit the matrix of the former
COO → CSR route.
"""
import os
import pickle
import uuid
import zlib

import numpy as np
import scipy.sparse as sp

FORMAT = 2
SUFFIX = ".ops2.pkl"
MF_FORMAT = 1
MF_SUFFIX = ".mf1.pkl"
NARROW_KEYS = ("P", "qx", "qw")               # dense arrays stored as float32 in the narrow variant
_I32 = np.iinfo(np.int32).max
_TRIAL = 1 << 20                              # bytes compressed to decide whether an array is worth zlib
_GAIN = 0.9                                   # keep zlib only if the trial shrinks the data below this
_GAIN_FAST = 0.3                              # the same for compress='fast'


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


def _gain(compress):
    """zlib is kept for an array only if it shrinks it below this fraction: True 0.9 (the default policy of format 2),
    'fast' 0.3 (only the arrays zlib shrinks a lot: K values, index differences; the poorly compressible float32 blocks
    stay raw and load without decompression), False/None never."""
    if compress is True:
        return _GAIN
    if compress == "fast":
        return _GAIN_FAST
    if compress in (False, None):
        return None
    raise ValueError(f"cache compress must be true, false or 'fast', got {compress!r}")


def _enc(a, store=None, delta=False, compress=True):
    """Packed array: dict(dtype, orig, shape, delta, codec, buf).  store: the dtype written (default: as is).
    compress: see _gain.  An index array is stored as differences only when zlib is kept (they exist to make it shrink);
    otherwise as it is (no cumulative sum when reading)."""
    gain = _gain(compress)
    a = np.asarray(a)
    orig = a.dtype
    st = np.dtype(store) if store is not None else orig
    plain = a
    if delta and gain is not None:
        if a.ndim != 1 or st.kind not in "iu":
            raise TypeError("delta coding is for 1-d integer arrays")
        d = np.diff(a.astype(np.int64), prepend=0)
        if d.size and (d.max() > np.iinfo(st).max or d.min() < np.iinfo(st).min):
            raise OverflowError("index differences do not fit the storage dtype")
        a = d
    else:
        delta = False
    if st.kind == "f" and st.itemsize < orig.itemsize and a.size and np.nanmax(np.abs(a)) > np.finfo(st).max:
        raise OverflowError(f"values beyond the largest {st} number cannot be stored narrow")
    b = np.ascontiguousarray(a.astype(st, copy=False))
    raw = b.tobytes()
    codec = "raw"
    if gain is not None and b.nbytes >= 4096:
        n = min(len(raw), _TRIAL)
        if len(zlib.compress(raw[:n], 1)) < gain * n:
            z = zlib.compress(raw, 1)
            if len(z) < gain * len(raw):
                raw, codec = z, "zlib"
    if delta and codec == "raw":                 # zlib was not worth it: keep the indices as they are
        return _enc(plain, store, compress=False)
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


def _upper_pattern(ptr, ind, n):
    """(indptr, indices, src) of the full canonical CSR of the symmetric matrix whose upper triangle (diagonal included,
    sorted columns) has the CSR arrays (ptr, ind): the full values are `val[src]`.  Row i of the full matrix is the
    transposed strict upper part (the entries (j, i), j < i, ascending j) followed by the stored upper row i (columns
    ≥ i): scipy's O(nnz) CSR → CSC counting sort transposes the strict part and its CSR sum merges the two, with the
    positions in the upper array (+ 1, so that no entry is zero) as the values: no sort, no COO."""
    rows = np.repeat(np.arange(n, dtype=ind.dtype), np.diff(ptr))
    pos = np.flatnonzero(ind > rows)                                     # strict upper entries, row-major
    sptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(rows[pos], minlength=n), out=sptr[1:])
    lower = sp.csr_matrix((pos + 1, ind[pos], sptr), shape=(n, n)).tocsc()      # CSC of S = CSR of Sᵀ
    lower = sp.csr_matrix((lower.data, lower.indices, lower.indptr), shape=(n, n))
    full = sp.csr_matrix((np.arange(1, ind.size + 1), ind, ptr), shape=(n, n)) + lower
    return full.indptr, full.indices, full.data - 1


class _Patterns:
    """The last upper-triangle pattern expanded and its result: K and M of one file share it (assembled on the same
    sparsity pattern), so the second matrix costs only the value gather.  Keyed by the stored arrays themselves."""

    def __init__(self):
        self.key = self.out = None

    def expand(self, ptr, ind, n):
        k = self.key
        if k is None or k[0] != n or not (np.array_equal(k[1], ptr) and np.array_equal(k[2], ind)):
            self.key, self.out = (n, ptr, ind), _upper_pattern(ptr, ind, n)
        return self.out


def _expand_upper(ptr, ind, val, n, patterns=None):
    """Full canonical CSR (indptr, sorted indices, data) of the symmetric matrix whose upper triangle (diagonal
    included) is given.  Only index arithmetic and one gather: the values are moved, never summed, and explicit zeros
    stay."""
    fptr, find, src = (patterns or _Patterns()).expand(ptr, ind, n)
    return fptr, find, val[src]


def _bits_equal(a, b):
    """Same shape, dtype and bit pattern (so -0.0 ≠ +0.0 and NaN payloads count)."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    u = {4: np.uint32, 8: np.uint64}.get(a.dtype.itemsize)
    return np.array_equal(a.view(u), b.view(u)) if u and a.dtype.kind == "f" else np.array_equal(a, b)


def _pack_csr(A, narrow, allow_sym, compress=True):
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
            "indptr": _enc(ptr_, ipt, delta=True, compress=compress),
            "indices": _enc(ind_, ixt, delta=True, compress=compress), "data": _enc(val_, vst, compress=compress)}


def _unpack_csr(p, patterns=None):
    m, n = p["shape"]
    ptr, ind, val = _dec(p["indptr"]), _dec(p["indices"]), _dec(p["data"])
    if p["sym"]:
        ptr, ind, val = _expand_upper(ptr, ind, val, m, patterns)
    ptr = ptr.astype(np.int32 if ptr[-1] <= _I32 else np.int64, copy=False)
    A = sp.csr_matrix((val, ind.astype(ptr.dtype, copy=False), ptr), shape=(m, n))
    if p["sym"]:
        A.has_sorted_indices = True        # _pack_csr verified the rebuilt matrix equals a canonical original
    return A


def pack(ops, rec, narrow, symmetric=("K", "M"), compress=True):
    """Cacheable payload of an operator dict + label record (module docstring).  compress=False: no zlib, no index
    differences (bigger file, no decompression and no cumulative sums on reading)."""
    out = {}
    for k, v in ops.items():
        if sp.issparse(v):
            out[k] = _pack_csr(v, narrow, k in symmetric, compress)
        elif isinstance(v, np.ndarray):
            st = np.float32 if narrow and k in NARROW_KEYS and v.dtype == np.float64 else None
            out[k] = {"kind": "array", "p": _enc(v, st, compress=compress)}
        else:
            out[k] = {"kind": "plain", "v": v}
    r = dict(rec)
    u = r.pop("u_model")
    u32 = u.astype(np.float32)           # labels come from an float32 H5 dataset: float32 is then exact
    exact = u.dtype == np.float64 and np.array_equal(u32.astype(np.float64), u)
    return {"format": FORMAT, "narrow": bool(narrow), "potential": "G" in ops and "Kp" in ops,
            "ops": out, "rec": r, "u_model": _enc(u, np.float32 if exact else None, compress=compress)}


def unpack(payload):
    """(ops, rec) of a payload from pack(); arrays and matrices in their original dtypes."""
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise OpsCacheError(f"operator cache format {payload.get('format') if isinstance(payload, dict) else '?'}, "
                            f"this code reads format {FORMAT}")
    try:
        ops, patterns = {}, _Patterns()
        for k, p in payload["ops"].items():
            ops[k] = _unpack_csr(p, patterns) if p["kind"] == "csr" else _dec(p["p"]) if p["kind"] == "array" else p["v"]
        rec = dict(payload["rec"])
        rec["u_model"] = _dec(payload["u_model"])
    except (zlib.error, KeyError) as e:
        raise OpsCacheError(f"corrupt operator cache payload ({type(e).__name__}: {e})") from e
    return ops, rec


def _uncompress_array(p):
    """The array record `p` in raw form: decompressed, index differences summed (same values on reading)."""
    if p["codec"] == "raw" and not p["delta"]:
        return p
    buf = zlib.decompress(p["buf"]) if p["codec"] == "zlib" else p["buf"]
    st = np.dtype(p["dtype"])
    a = np.frombuffer(buf, dtype=st).reshape(p["shape"])
    if p["delta"]:
        a = np.cumsum(a, dtype=np.int64).astype(st)      # the partial sums are the original values, which fit st
    return dict(p, codec="raw", delta=False, buf=np.ascontiguousarray(a).tobytes())


def uncompress(payload):
    """A payload of pack() with every array raw (no zlib, no index differences): what pack(compress=False) writes, made
    from an existing payload without the operators (scripts/recode_ops_cache.py).  unpack() of both is identical."""
    ops = {}
    for k, p in payload["ops"].items():
        if p["kind"] == "csr":
            ops[k] = dict(p, **{n: _uncompress_array(p[n]) for n in ("indptr", "indices", "data")})
        elif p["kind"] == "array":
            ops[k] = dict(p, p=_uncompress_array(p["p"]))
        else:
            ops[k] = p
    return dict(payload, ops=ops, u_model=_uncompress_array(payload["u_model"]))


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


# ─────────────────────────── matrix-free variant (docs/40 §10) ───────────────────────────

def pack_mf(bundle, rec):
    """Payload of the matrix-free item data: bundle = {"mf": record of matfree.build_record, "qfeat": float32 [Nt·nq, F],
    "qw": float32 [Nt·nq]}, rec = the label record of read_field_group (u_model, freqs, qoi, ...).  Every array is stored raw
    (float32 labels when that is exact, as pack does)."""
    import src.data.matfree as matfree
    r = dict(rec)
    u = r.pop("u_model")
    exact = u.dtype == np.float64 and np.array_equal(u.astype(np.float32).astype(np.float64), u)
    return {"kind": "matfree", "mf_format": MF_FORMAT, "matfree_format": matfree.FORMAT,
            "ngsolve": matfree._ngsolve_version(), "mf": dict(bundle["mf"]),
            "qfeat": np.ascontiguousarray(bundle["qfeat"], dtype=np.float32),
            "qw": np.ascontiguousarray(bundle["qw"], dtype=np.float32), "rec": r,
            "u_model": np.ascontiguousarray(u.astype(np.float32) if exact else u)}


def unpack_mf(payload):
    """(bundle, rec) of a payload from pack_mf; OpsCacheError if it is of another kind / format / NGSolve version (the
    DOF numbering and the hierarchical basis belong to the build that wrote the record)."""
    import src.data.matfree as matfree
    if (not isinstance(payload, dict) or payload.get("kind") != "matfree" or payload.get("mf_format") != MF_FORMAT
            or payload.get("matfree_format") != matfree.FORMAT):
        raise OpsCacheError(f"matrix-free cache of another kind / format (this code reads mf_format {MF_FORMAT}, "
                            f"matfree.FORMAT {matfree.FORMAT})")
    if payload.get("ngsolve") != matfree._ngsolve_version():
        raise OpsCacheError(f"matrix-free cache written with ngsolve {payload.get('ngsolve')}, this is "
                            f"{matfree._ngsolve_version()}")
    try:
        rec = dict(payload["rec"])
        rec["u_model"] = payload["u_model"].astype(np.float64)
        return {"mf": payload["mf"], "qfeat": payload["qfeat"], "qw": payload["qw"]}, rec
    except (KeyError, AttributeError) as e:
        raise OpsCacheError(f"corrupt matrix-free cache payload ({type(e).__name__}: {e})") from e


def read_mf(path):
    """Payload of a matrix-free cache file (raw pickle); OpsCacheError if unreadable or not of this kind."""
    try:
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
    except (EOFError, pickle.UnpicklingError, AttributeError, ImportError, IndexError, ValueError) as e:
        raise OpsCacheError(f"{path}: unreadable matrix-free cache ({type(e).__name__}: {e})") from e
    if not isinstance(payload, dict) or payload.get("kind") != "matfree":
        raise OpsCacheError(f"{path}: not a matrix-free cache file")
    return payload
