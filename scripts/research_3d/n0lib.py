"""docs/18 research code. Lowest-order Nédélec (Whitney edge) Maxwell eigen-solver on tetrahedra with
scikit-fem + SciPy, and gmsh helpers.  Research/feasibility code (docs/18).

Problem:  curl curl E = k² E in Ω,  n × E = 0 on ∂Ω (PEC),  E ∈ H0(curl).
Discrete: K u = λ M u,  K = (curl φ_i, curl φ_j),  M = (φ_i, φ_j).
Kernel:   ker K = G · (interior P1 potentials)  (discrete gradients, exact
          sequence P1 --grad--> N0), dim = #interior vertices  → λ = 0 modes.
"""
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import jn_zeros, jnp_zeros
from skfem import Basis, BilinearForm, ElementTetN0, MeshTet
from skfem.helpers import curl, dot, grad

# ─────────────────────────── meshes ────────────────────────────

def box_mesh(a, b, d, n):
    """Structured tet mesh of [0,a]×[0,b]×[0,d] with ~n cells along the longest side."""
    h = max(a, b, d) / n
    ax = [np.linspace(0, L, max(2, int(round(L / h))) + 1) for L in (a, b, d)]
    return MeshTet.init_tensor(*ax)


def gmsh_mesh(build, h, order_opt=True):
    """build(occ) creates OCC volume(s); returns MeshTet (linear tets) meshed at size h."""
    import gmsh
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("m")
    build(gmsh.model.occ)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)
    gmsh.option.setNumber("Mesh.MeshSizeMin", 0.2 * h)
    if order_opt:
        gmsh.option.setNumber("Mesh.Optimize", 1)
    gmsh.model.mesh.generate(3)
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    et, _, conn = gmsh.model.mesh.getElements(3)
    i4 = [k for k, t in enumerate(et) if int(t) == 4][0]
    tet = np.asarray(conn[i4], dtype=np.int64).reshape(-1, 4)
    xyz = np.asarray(xyz).reshape(-1, 3)
    tags = np.asarray(tags, dtype=np.int64)
    t2r = np.full(tags.max() + 1, -1, dtype=np.int64); t2r[tags] = np.arange(len(tags))
    tet = t2r[tet]
    used = np.unique(tet); remap = np.full(len(tags), -1); remap[used] = np.arange(len(used))
    gmsh.finalize()
    return MeshTet(np.ascontiguousarray(xyz[used].T), np.ascontiguousarray(remap[tet].T))


# ─────────────────────────── assembly ──────────────────────────

@BilinearForm
def _curlcurl(u, v, w):
    return dot(curl(u), curl(v))


@BilinearForm
def _mass(u, v, w):
    return dot(u, v)


@BilinearForm
def _lap(u, v, w):
    return dot(grad(u), grad(v))


def assemble_n0(mesh):
    """Returns dict with K, M (interior edge DOFs), G (interior edges × interior
    vertices), Kp = Gᵀ M G (= P1 Dirichlet stiffness), and index maps."""
    bN = Basis(mesh, ElementTetN0())
    K = _curlcurl.assemble(bN).tocsr()
    M = _mass.assemble(bN).tocsr()
    D = bN.get_dofs(mesh.boundary_facets()).all()           # tangential (edge) DOFs on ∂Ω
    I = np.setdiff1d(np.arange(bN.N), D)
    e = mesh.edges
    ne = e.shape[1]
    dof = bN.edge_dofs[0]
    G = sp.csr_matrix((np.r_[-np.ones(ne), np.ones(ne)], (np.r_[dof, dof], np.r_[e[0], e[1]])),
                      shape=(bN.N, mesh.p.shape[1]))
    Iv = np.setdiff1d(np.arange(mesh.p.shape[1]), mesh.boundary_nodes())
    Gi = G[I][:, Iv].tocsr()
    Ki, Mi = K[I][:, I].tocsr(), M[I][:, I].tocsr()
    return dict(basis=bN, K=Ki, M=Mi, G=Gi, I=I, Iv=Iv, Kfull=K, Mfull=M, Gfull=G)


# ─────────────────────────── solvers ───────────────────────────

def factor(A, backend="auto"):
    """Sparse direct factorisation → (solve, nnz or -1).  Uses MKL PARDISO via
    pypardiso when importable (multithreaded, nested dissection), else SuperLU."""
    try:
        if backend == "superlu":
            raise ImportError
        import pypardiso
        s = pypardiso.PyPardisoSolver()
        s.set_matrix_type(11)          # real nonsymmetric (robust default)
        A = sp.csr_matrix(A)
        s.factorize(A)
        return (lambda b: s.solve(A, b)), -1
    except ImportError:
        lu = spla.splu(sp.csc_matrix(A), permc_spec='MMD_AT_PLUS_A')
        return lu.solve, lu.L.nnz + lu.U.nnz


def solve_projected(A, k, sigma=None, tol=0.0):
    """k lowest NON-zero eigenpairs via shift-invert with σ < 0 (K − σM SPD)
    and M-orthogonal projection onto discrete divergence-free fields in every
    Arnoldi step:  P = I − G Kp⁻¹ Gᵀ M,  Kp = Gᵀ M G (P1 Laplacian).
    Gradient fields become θ = 0 (λ = ∞) of the projected operator."""
    K, M, G = A['K'], A['M'], A['G']
    t0 = time.perf_counter()
    if sigma is None:  # scale-aware small negative shift
        sigma = -1e-2 * (K.diagonal().mean() / M.diagonal().mean())
    lu_solve, fill = factor(K - sigma * M)
    Kp = (G.T @ M @ G).tocsr()
    lup_solve, fillp = factor(Kp, backend='superlu')   # small SPD potential problem
    MG = (M @ G).tocsr()

    def P(x):
        return x - G @ lup_solve(MG.T @ x)

    def opinv(x):
        # ARPACK mode 3 applies OPinv to M·y (a dual vector); S = (K−σM)⁻¹ maps
        # gradients to gradients, so P·S·M = S·M·P and projecting after the solve suffices.
        return P(lu_solve(x))
    OP = spla.LinearOperator(K.shape, matvec=opinv, dtype=float)
    t1 = time.perf_counter()
    v0 = P(np.random.default_rng(0).standard_normal(K.shape[0]))
    vals, vecs = spla.eigsh(K, k=k, M=M, sigma=sigma, OPinv=OP, which="LM", v0=v0, tol=tol, ncv=max(2 * k + 1, 30))
    t2 = time.perf_counter()
    o = np.argsort(vals)
    return vals[o], vecs[:, o], dict(t_factor=t1 - t0, t_eig=t2 - t1, lu_nnz=fill,
                                     lu_nnz_p=fillp, sigma=sigma)


def solve_naive(A, k, sigma):
    """Plain shift-invert around σ ≥ 0 — returns what ARPACK finds (zeros included)."""
    K, M = A['K'], A['M']
    t0 = time.perf_counter()
    vals, vecs = spla.eigsh(K, k=k, M=M, sigma=sigma, which='LM')
    return np.sort(vals), vecs, time.perf_counter() - t0


def solve_filtered(A, k, sigma, extra=0, zero_tol=1e-6):
    """Shift-invert at σ > 0 (≈ between the kernel and the target band), then
    discard |λ| < zero_tol·σ.  Only safe if σ > λ_k / 2 (zeros farther than targets)."""
    K, M = A['K'], A['M']
    t0 = time.perf_counter()
    vals, vecs = spla.eigsh(K, k=k + extra, M=M, sigma=sigma, which='LM')
    keep = vals > zero_tol * sigma
    o = np.argsort(vals[keep])
    return vals[keep][o][:k], vecs[:, keep][:, o][:, :k], time.perf_counter() - t0, int((~keep).sum())


def solve_penalty(A, k, s):
    """Grad-div-type regularisation: K_s = K + s·(MG) D⁻¹ (MG)ᵀ with D the lumped
    P1 mass.  Physical λ unchanged (they are M-orthogonal to gradients); the
    gradient modes move to s·μ_j (μ_j: lumped-P1 Dirichlet Laplacian values).
    K_s is SPD → σ = 0 shift-invert.  Wrong if s·μ_1 < λ_k."""
    K, M, G = A['K'], A['M'], A['G']
    t0 = time.perf_counter()
    MG = (M @ G).tocsr()
    # lumped P1 mass on interior vertices from the basis volume shares
    mesh = A['basis'].mesh
    vol = np.abs(np.linalg.det(np.stack([mesh.p[:, mesh.t[i]] - mesh.p[:, mesh.t[0]] for i in (1, 2, 3)], -1).transpose(1, 0, 2))) / 6.0
    lump = np.zeros(mesh.p.shape[1])
    for i in range(4):
        np.add.at(lump, mesh.t[i], vol / 4.0)
    Dinv = sp.diags(1.0 / lump[A['Iv']])
    Ks = (K + s * (MG @ Dinv @ MG.T)).tocsc()
    vals, vecs = spla.eigsh(Ks, k=k, M=M, sigma=0.0, which='LM')
    return np.sort(vals), vecs, time.perf_counter() - t0, Ks.nnz


# ─────────────────────────── analytic spectra ──────────────────

def box_spectrum(a, b, d, nmax=6, kmax=None):
    """PEC box: k² = (mπ/a)² + (nπ/b)² + (pπ/d)²; multiplicity 2 if m,n,p ≥ 1
    (TE + TM), 1 if exactly one index is 0, none if two are 0."""
    out = []
    for m_ in range(nmax):
        for n_ in range(nmax):
            for p_ in range(nmax):
                z = (m_ == 0) + (n_ == 0) + (p_ == 0)
                if z >= 2:
                    continue
                k2 = (m_ * np.pi / a) ** 2 + (n_ * np.pi / b) ** 2 + (p_ * np.pi / d) ** 2
                out += [k2] * (2 if z == 0 else 1)
    return np.sort(out)


def pillbox_spectrum(R, L, nmax=5):
    """PEC pillbox (radius R, length L):  TM_mnp: (j_mn/R)² + (pπ/L)², p ≥ 0;
    TE_mnp: (j'_mn/R)² + (pπ/L)², p ≥ 1; m ≥ 1 twofold degenerate.
    Returns sorted list of (k², label)."""
    out = []
    for m_ in range(nmax):
        jz, jpz = jn_zeros(m_, nmax), jnp_zeros(m_, nmax) if m_ > 0 else jn_zeros(1, nmax)  # J0' = −J1
        mult = 1 if m_ == 0 else 2
        for n_ in range(nmax):
            for p_ in range(nmax):
                out += [((jz[n_] / R) ** 2 + (p_ * np.pi / L) ** 2, f"TM{m_}{n_+1}{p_}")] * mult
                if p_ >= 1:
                    out += [((jpz[n_] / R) ** 2 + (p_ * np.pi / L) ** 2, f"TE{m_}{n_+1}{p_}")] * mult
    out.sort(key=lambda t: t[0])
    return out
