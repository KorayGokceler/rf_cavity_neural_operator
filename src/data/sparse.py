"""Sparse helpers of the data pipeline and the service: scipy CSR from stored tuples (to_csr) and the
block-diagonal torch operators of a padded batch (block_diag, layout of src.models.hcurl)."""
import numpy as np
import scipy.sparse as sp
import torch


def to_csr(obj, shape=None):
    """scipy CSR from a scipy matrix, an (indptr, indices, data[, shape]) tuple
    or a {'indptr', 'indices', 'data'[, 'shape']} dict."""
    if sp.issparse(obj):
        return obj.tocsr()
    if isinstance(obj, dict):
        obj = (obj['indptr'], obj['indices'], obj['data']) + ((obj['shape'],) if 'shape' in obj else ())
    indptr, indices, data = (np.asarray(a) for a in obj[:3])
    if len(obj) > 3:
        shape = tuple(int(s) for s in obj[3])
    if shape is None:
        shape = (len(indptr) - 1, int(indices.max()) + 1 if len(indices) else 0)
    return sp.csr_matrix((data.astype(np.float64), indices, indptr), shape=shape)


def block_diag(mats, rows, cols, layout='csr'):
    """Block-diagonal torch CSR (float64) of scipy matrices on a padded
    (rows × cols per block) layout.  CSR: ~3–8× faster sparse products than
    COO (review docs/22, m5) and no coalesce step.  layout='coo': a coalesced
    torch COO tensor (row-major) of the same matrix.

    The CSR blocks are concatenated directly (canonical: sorted columns, duplicates summed — what
    scipy's block_diag gives through COO, entry for entry); 0.80 → 0.34 s (steady state) for one 140k-DOF p2
    geometry (docs/31 §3.6)."""
    if layout not in ('csr', 'coo'):
        raise ValueError(f"layout must be 'csr' or 'coo', got {layout!r}")
    blocks = []
    for A in mats:
        A = sp.csr_matrix(A, dtype=np.float64)
        if A.shape[0] > rows or A.shape[1] > cols:
            raise ValueError(f"block_diag: block {A.shape} exceeds the padded {rows} × {cols}")
        if not A.has_canonical_format:
            A = A.copy()
            A.sum_duplicates()
        blocks.append(A)
    if layout == 'coo':
        for A in blocks:
            A.resize((rows, cols))
        S = sp.block_diag(blocks, format='csr')
        S.sort_indices()
        C = S.tocoo()
        idx = torch.from_numpy(np.vstack([C.row, C.col]).astype(np.int64))
        return torch.sparse_coo_tensor(idx, torch.from_numpy(C.data.astype(np.float64)),
                                       size=S.shape).coalesce()
    # one allocation per output array (fresh memory dominates the cost); int64 offsets
    nb, tot = len(blocks), sum(A.nnz for A in blocks)
    ptr = np.zeros(nb * rows + 1, np.int64)
    ind = np.empty(tot, np.int64)
    val = blocks[0].data.copy() if nb == 1 else np.empty(tot, np.float64)   # never the dataset's arrays
    o = 0
    for b, A in enumerate(blocks):
        r0, r = b * rows + 1, A.shape[0]
        np.add(A.indptr[1:], o, out=ptr[r0:r0 + r], dtype=np.int64)
        ptr[r0 + r:r0 + rows] = o + A.nnz
        np.add(A.indices, b * cols, out=ind[o:o + A.nnz], dtype=np.int64)
        if nb > 1:
            val[o:o + A.nnz] = A.data
        o += A.nnz
    return torch.sparse_csr_tensor(torch.from_numpy(ptr), torch.from_numpy(ind), torch.from_numpy(val),
                                   size=(nb * rows, nb * cols), check_invariants=False)


def length_mask(lengths, n):
    """bool [B, n]: entry j of row b is valid (j < lengths[b])."""
    return torch.arange(n)[None, :] < torch.as_tensor(lengths)[:, None]
