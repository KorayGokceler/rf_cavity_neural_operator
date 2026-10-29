# AFEM theory for Maxwell (and elliptic) eigenvalue problems: SOLVE → ESTIMATE → MARK → REFINE

Scope: how a provably convergent and rate-optimal adaptive loop for the Maxwell eigenproblem
`curl(μ⁻¹ curl u) = ω² ε u`, `u × n = 0` is built; which ingredients the proofs need; what is proven
versus what is only practice. Elliptic results are included where they set the template. Sources were
read at abstract level unless marked "(full text read)". Boffi–Gastaldi 2019 and Carstensen–Gedicke
2012 were read in full.

---

## 1. The reference algorithm (what the proven Maxwell loop actually is)

### Takeaway
The only proven-optimal 3D Maxwell eigenvalue AFEM (Boffi–Gastaldi 2019) is: Nédélec edge elements on a
polyhedral, simply connected domain; a residual estimator; Dörfler marking with a *small enough* θ;
"smallest admissible refinement" by Binev–Dahmen–DeVore / Stevenson bisection, so the meshes are nested
and shape-regular; and an initial mesh that is *fine enough*. The proof works through the equivalent mixed
formulation (Boffi–Fernandes–Gastaldi–Perugia 1999). It is written for a simple eigenvalue and says the
cluster case follows from Boffi–Gallistl–Gardini–Gastaldi 2017.

### Cited Findings
- Boffi & Gastaldi, "Adaptive finite element method for the Maxwell eigenvalue problem", SIAM J. Numer. Anal. 57(1) (2019) 478–494, DOI 10.1137/18M1179389: "we prove the optimal convergence of a standard adaptive scheme based on edge finite elements… The proof uses the known equivalence… with a mixed eigenvalue problem." (full text read) — [Boffi–Gastaldi 2019 PDF](https://iris.unibs.it/retrieve/93b0a02a-19e5-4e94-b044-61128c442618/18m1179389.pdf)
- Local estimator (eq. 3.1): η̃²_K = h²_K‖εu_h − curl(μ⁻¹curl u_h)/ω²_h‖²_K + h²_K‖div(εu_h)‖²_K + ½Σ_{F⊂K interior} h_F(‖[μ⁻¹curl u_h/ω²_h × n]‖²_F + ‖[εu_h·n]‖²_F). It contains a **divergence residual and a normal-jump term**, which measure the error in the gradient (kernel) part. The mixed-form estimator satisfies η̃²_K = η²_K/λ_h. — [Boffi–Gastaldi 2019](https://iris.unibs.it/retrieve/93b0a02a-19e5-4e94-b044-61128c442618/18m1179389.pdf)
- MARK: "choosing a minimal subset M_ℓ of T_ℓ such that θ η̃²_ℓ(T_ℓ) ≤ η̃²_ℓ(M_ℓ)", 0 < θ ≤ 1. REFINE: "the smallest admissible refinement of T_ℓ satisfying M_ℓ ∩ T_{ℓ+1} = ∅ according to the rules defined in [4, 42]". Here [4] is Binev–Dahmen–DeVore, Numer. Math. 97 (2004) 219–268, DOI 10.1007/s00211-003-0492-7, and [42] is Stevenson, Math. Comp. 77 (2008) 227–241, DOI 10.1090/S0025-5718-07-01959-X. So **every marked element must be refined** (it may not survive into T_{ℓ+1}). — [Boffi–Gastaldi 2019](https://iris.unibs.it/retrieve/93b0a02a-19e5-4e94-b044-61128c442618/18m1179389.pdf)
- Theorem 3.2: "Provided the meshsize of the initial mesh T₀ and the bulk parameter θ are small enough, if the eigenspace satisfies |W̃|_{A_s} < ∞", then δ(W̃, W̃_ℓ) ≤ C(card(T_ℓ) − card(T₀))^{−s}|W̃|_{A_s}, and |ω − ω_ℓ| ≤ C δ(W̃, W̃_ℓ)² (double-order rate for the eigenvalue). — [Boffi–Gastaldi 2019](https://iris.unibs.it/retrieve/93b0a02a-19e5-4e94-b044-61128c442618/18m1179389.pdf)
- The three ingredients of the proof are "discrete reliability, quasi-orthogonality, and the contraction property". The contraction is ξ²_{ℓ+1} ≤ γ ξ²_ℓ for ξ²_ℓ = η(T_ℓ)² + β(‖σ_ℓ − σ_{ℓ+1}‖² + ‖p_ℓ − p_{ℓ+1}‖²), with γ ∈ (0,1) "if the initial mesh T₀ is sufficiently fine". The authors call discrete reliability "the most challenging part of the proof". — [Boffi–Gastaldi 2019](https://iris.unibs.it/retrieve/93b0a02a-19e5-4e94-b044-61128c442618/18m1179389.pdf)
- Reliability and efficiency hold only "for h small enough", with higher-order terms ρ(h) → 0. Efficiency for the primal estimator is **not local**: it involves h_K‖ω²u − ω²_h u_h‖, which is a global quantity. Local efficiency holds in the mixed form. The estimator was proven in Boffi–Gastaldi–Rodríguez–Šebestová, IMA J. Numer. Anal. 37 (2017) 1710–1732 (DOI 10.1093/imanum/drw066) and J. Sci. Comput. (2018/2019), DOI 10.1007/s10915-018-0808-5. — [Boffi–Gastaldi 2019](https://iris.unibs.it/retrieve/93b0a02a-19e5-4e94-b044-61128c442618/18m1179389.pdf)
- Domain and topology assumptions: Ω is a polyhedron, "simply connected" for simplicity. For non-trivial topology, the harmonic fields H (curl h = 0, div εh = 0) are zero-frequency eigenfunctions "outside the scope of this paper". The authors point to Demlow, Numer. Math. 136 (2017) 941–971 (DOI 10.1007/s00211-016-0862-6) for AFEM of harmonic forms, and to the Kikuchi mixed formulation. — [Boffi–Gastaldi 2019](https://iris.unibs.it/retrieve/93b0a02a-19e5-4e94-b044-61128c442618/18m1179389.pdf)
- "the mixed formulations… are not used for the definition of the method (nor for its implementation) but are crucial ingredients for its analysis." The kernel ∇N_h is not removed algorithmically. The count of nonzero discrete eigenvalues equals dim Q_h because dim Σ_h = dim ∇N_h + N(h) (discrete Helmholtz decomposition). — [Boffi–Gastaldi 2019](https://iris.unibs.it/retrieve/93b0a02a-19e5-4e94-b044-61128c442618/18m1179389.pdf)
- 2D Maxwell AFEM follows from the mixed-Laplacian cluster result through the H(curl) ≅ H(div) isomorphism. The 3D extension "is not trivial", and "suitably designed interpolation operators" (Scott–Zhang type, plus results of Schöberl, Math. Comp. 77 (2008) 633–649, DOI 10.1090/S0025-5718-07-02030-3) are needed. — [Boffi–Gastaldi 2019](https://iris.unibs.it/retrieve/93b0a02a-19e5-4e94-b044-61128c442618/18m1179389.pdf)
- For the source problem (indefinite time-harmonic Maxwell), Zhong–Chen–Shu–Wittum–Xu, Math. Comp. 81 (2012) 623–642, proved contraction and optimality for Nédélec elements of arbitrary order with "minimal element refinement without the interior node property", but "only when the initial mesh is sufficiently fine". Their key tools are quasi-orthogonality and a localized estimator. — [AMS page](https://www.ams.org/mcom/2012-81-278/S0025-5718-2011-02544-5/)

### Inferences
- A loop that matches the theory is: (i) solve for *all* modes in the targeted window (cluster); (ii) a residual estimator that **includes the div(εu_h) and [εu_h·n] terms**; (iii) Dörfler marking on the cluster-summed indicator; (iv) conforming bisection with closure, so every marked element is bisected at least once; (v) a start from a mesh that already resolves the target modes, so the asymptotic regime holds.
- The mixed formulation and the discrete Helmholtz decomposition are tools of the analysis. A practical code may still need an explicit kernel treatment in the *eigensolver* (gradient projection, shift-invert away from 0) for efficiency, but this is not part of the convergence theory.

### Gaps
- I found no published optimality proof for 3D Maxwell eigen-AFEM on **curved** (non-polyhedral) domains, on **multiply connected** domains for the non-zero spectrum combined with an adaptive harmonic-field treatment, or with **higher-order curved (isoparametric) H(curl)** elements. Boffi–Gastaldi assume polyhedral Ω.
- The full statement for clusters in 3D Maxwell is claimed to follow "by arguments similar to [BGGG]". I found no separate 3D Maxwell cluster paper.

---

## 2. Marking: Dörfler/bulk, maximum, equidistribution — which is needed for which result

### Takeaway
*Plain convergence* (error → 0, no rate) holds for essentially any "reasonable" marking (Dörfler, maximum,
equidistribution), provided every marked element is actually refined. *Contraction and rate optimality* are
proven only for **Dörfler marking with θ small enough and a (quasi-)minimal marked set**. Practice uses
θ ≈ 0.3–0.6: Carstensen–Gedicke observed optimal rates from θ ≤ 0.5 and a "stair" effect for θ ≤ 0.2.

### Cited Findings
- Morin–Siebert–Veeser, "A basic convergence result for conforming adaptive finite elements", M3AS 18(5) (2008) 707–737. Their framework assumes conforming and **nested** spaces, a reliable and locally/discretely efficient estimator, and "marked elements subdivided at least once". They give "a sufficient and essentially necessary condition on the marking step", satisfied by Dörfler, maximum *and* equidistribution strategies. — [MSV abstract](https://amcaonline.org.ar/ojs/index.php/cmm/article/view/2534); [preprint](https://opus.bibliothek.uni-augsburg.de/opus4/frontdoor/index/index/docId/512)
- Garau–Morin–Zuppa, "Convergence of adaptive finite element methods for eigenvalue problems", M3AS 19 (2009) 721–747 (DOI 10.1142/S0218202509003590; arXiv:0803.0365). They prove convergence for Lagrange elements of any degree, simple *and multiple* eigenvalues, "starting from any initial triangulation", for "all reasonable marking strategies", with minimal refinement of marked elements. This is convergence without a rate and *without* a fine-initial-mesh condition. — [arXiv:0803.0365](https://arxiv.org/abs/0803.0365); DOI from [BG2019 refs](https://iris.unibs.it/retrieve/93b0a02a-19e5-4e94-b044-61128c442618/18m1179389.pdf)
- Dörfler, "A convergent adaptive algorithm for Poisson's equation", SIAM J. Numer. Anal. 33 (1996) 1106–1124, DOI 10.1137/0733054. This paper introduced bulk marking. — [BG2019 refs](https://iris.unibs.it/retrieve/93b0a02a-19e5-4e94-b044-61128c442618/18m1179389.pdf)
- Optimality template: Stevenson, Found. Comput. Math. 7 (2007) 245–269 (DOI 10.1007/s10208-005-0183-0), and Cascón–Kreuzer–Nochetto–Siebert, SIAM J. Numer. Anal. 46 (2008) 2524–2550 (DOI 10.1137/07069047X). In this template, Dörfler marking with a minimal set plus contraction gives optimality in the nonlinear approximation classes A_s. — [BG2019 refs](https://iris.unibs.it/retrieve/93b0a02a-19e5-4e94-b044-61128c442618/18m1179389.pdf)
- For clusters (BGGG 2017), the proof needs an explicit small-θ bound 0 < θ ≤ 1/(C₁(B/A)²(2N² + 4N³)), where [A, B] contains the cluster and N is its size. This is a severe worst-case constant that grows with the cluster size. — [ar5iv 1504.06418](https://ar5iv.arxiv.org/html/1504.06418)
- Bonito–Demlow (arXiv:1508.06265) showed for higher-order elements that "the bulk marking parameter can be chosen independently of the target cluster's properties". All such results still "assume a fineness condition on the initial mesh". — [arXiv:1508.06265](https://arxiv.org/abs/1508.06265)
- Carstensen–Gedicke 2012 (full text read) measured, for the slit domain with P1, a "strong dependency of the size of the eigenvalue error on θ". θ = 1 (uniform refinement) gives suboptimal rates. Smaller θ gives optimal rates, and the error decreases down to θ = 0.4. "for values θ ≤ 0.2 the algorithm marks too few elements such that the algorithm accepts the value of the previous level as approximation for the next one from time to time… convergence plots look like a stair." Their default is θ = 0.5. — [Carstensen–Gedicke preprint](https://ins.uni-bonn.de/media/public/publication-media/CG_2012.pdf?pk=1445)
- The marked set only needs "almost minimal cardinality" (greedy or sorting) for the optimality proof. — [Carstensen–Gedicke](https://ins.uni-bonn.de/media/public/publication-media/CG_2012.pdf?pk=1445)

### Inferences
- **Marked but not refined:** all of these proofs (MSV, GMZ, CKNS, BG) assume that every marked element is subdivided at least once. The estimator-reduction step, η²_{ℓ+1} ≤ (1+δ)(η²_ℓ − λ η²_ℓ(M_ℓ)) + …, relies on h_T shrinking by a fixed factor on M_ℓ. If marked elements are skipped (a remesher that ignores part of the size field, a refinement that is cancelled, or failed bisections that are rolled back), the contraction constant is lost. The loop can then stall: the mesh does not change, so the solution and estimator repeat. The "stair" effect Carstensen–Gedicke report for small θ is the mild version of this.
- Maximum and equidistribution marking are safe for *convergence* but carry no proven *rate*. Dörfler with θ ∈ [0.3, 0.6] is the usual compromise.

### Gaps
- I found no paper that quantifies how much of the marked set may be left unrefined before convergence fails.

---

## 3. Refinement: bisection (NVB, Bänsch / Arnold–Mukherjee–Pouly, Stevenson) vs red-green vs remeshing

### Takeaway
The theory needs (a) nested conforming spaces, (b) uniform shape regularity, and (c) a closure (completion)
with bounded overhead, #T_ℓ − #T₀ ≤ C Σ_k #M_k (BDD 2004 in 2D, Stevenson 2008 in n-D, under an
initial-mesh labelling condition). Newest-vertex / Maubach / AMP bisection meets all three. Red-green
refinement is non-nested in general and needs special analysis. Metric-based anisotropic remeshing is
non-nested, and only one convergence proof is known (Carson et al. 2022, for a source problem).

### Cited Findings
- Stevenson 2008: "To retain conformity, a bisection of a simplex has to be complemented by bisections of some of its neighbours…". The paper extends the 2D BDD completion estimate to n-simplices and "generaliz[es] the optimality result for adaptive finite element methods to general space dimensions". Its approach gives "uniformly shape-regular simplices, and nested finite element spaces". — [AMS page](https://www.ams.org/mcom/2008-77-261/S0025-5718-07-01959-X/); [dare.uva.nl](https://dare.uva.nl/id/dc1b6ee3-a3a2-4019-b73c-59eb121bde18)
- The completion bound needs properties of the initial mesh, i.e. a matching labelling of refinement edges, for the recursion to terminate and the complexity bound to hold. This is summarized secondarily (Alkämper–Gaspoz–Klöfkorn), not verified in Stevenson's original. — [search summary of Stevenson 2008 follow-ups](https://dare.uva.nl/id/dc1b6ee3-a3a2-4019-b73c-59eb121bde18)
- Arnold–Mukherjee–Pouly, "Locally adapted tetrahedral meshes using bisection", SIAM J. Sci. Comput. 22(2) (2000) 431–448, DOI 10.1137/S1064827597323373. Repeated bisection produces only finitely many similarity classes of tetrahedra, which gives shape regularity, and the extra refinement needed for conformity is bounded. The algorithm is essentially equivalent to Bänsch and Liu–Joe. — [SIAM](https://epubs.siam.org/doi/10.1137/S1064827597323373); [PDF](https://www-users.cse.umn.edu/~arnold//papers/bistet.pdf)
- For 2D NVB the closure adds a bounded number of refinements, with a bound depending only on T₀. Later work removes most initial-mesh assumptions in 2D. — [arXiv:1306.5100](https://arxiv.org/pdf/1306.5100); [arXiv:1210.0367](https://arxiv.org/pdf/1210.0367)
- Red-green refinement is non-nested. Optimal convergence for P1 conforming and non-conforming elements with Dörfler marking on red-green meshes was proven via "a new quasi-orthogonality" under an initial-mesh "Condition A", plus a modified refinement under a weaker "Condition B". — [Numer. Math. DOI 10.1007/s00211-012-0495-3](https://link.springer.com/doi/10.1007/s00211-012-0495-3)
- Carson–Allmaras–Galbraith–Darmofal, "Convergence of anisotropic mesh adaptation via metric optimization", SIAM J. Numer. Anal. 60(3) (2022), DOI 10.1137/20M1338721. Standard AFEM proofs assume nested meshes. MAFEM optimizes a Riemannian metric (continuous mesh model) and produces "a sequence of nonnested meshes". They prove convergence "using an optimization statement to relate the error on the sequence of meshes", and the optimal asymptotic rate for a fixed polynomial order. The test problem is a singularly perturbed advection–diffusion *source* problem. — [MIT DSpace](https://dspace.mit.edu/handle/1721.1/165659)
- The Bonito–Canuto–Nochetto–Veeser Acta Numerica survey, "Adaptive finite element methods", Acta Numer. 33 (2024) 163–485, DOI 10.1017/S0962492924000011 (arXiv:2402.07273), is the current reference for linear convergence and rate optimality of AFEM. It includes a chapter on bisection. The older Nochetto–Siebert–Veeser 2009 chapter is in *Multiscale, Nonlinear and Adaptive Approximation*, Springer, pp. 409–542. — [arXiv:2402.07273](https://arxiv.org/abs/2402.07273v2); [NSV chapter record](https://ideas.repec.org/h/spr/sprchp/978-3-642-03413-8_12.html)

### Curved boundaries
- Dörfler–Rumpf, Math. Comp. 67 (1998) 1361–1382: AFEM on domains with piecewise smooth boundary. "Boundary vertices sit on the true boundary." The estimator also controls the boundary-approximation error (energy and L² norms). Efficiency is proven "in the case of a saturated boundary approximation". The adaptation is Delaunay-based. — [AMS](https://www.ams.org/mcom/1998-67-224/S0025-5718-98-00993-4/)
- Bonito–Cascón–Mekchay–Morin–Nochetto, "High-order AFEM for the Laplace–Beltrami operator: convergence rates", Found. Comput. Math. 16(6) (2016) 1473–1539, DOI 10.1007/s10208-016-9335-7 (arXiv:1511.05019). With parametric geometry, the surface must be "well resolved in W¹_∞ relative to the PDE's resolution in H¹". This gives only a **conditional contraction**, and rates are "governed by the weaker of the decay rates of the surface error… and the PDE error". — [arXiv:1511.05019](https://arxiv.org/abs/1511.05019); [record](https://produccioncientifica.usal.es/documentos/5e69576d2999522cccdd1243)

### Inferences
- When a boundary edge is bisected and its midpoint is moved onto the curved surface, the discrete domain Ω_{h,ℓ+1} ≠ Ω_{h,ℓ}. The old space is then not a subspace of the new one (V_ℓ ⊄ V_{ℓ+1}), and Galerkin orthogonality / quasi-orthogonality pick up a geometric consistency term. The available theory (Dörfler–Rumpf, BCMMN) handles this by adding a **geometry estimator** and requiring the geometry to be resolved at least as well as the field. For H(curl) eigenproblems on curved 3D cavities, I found no proof of this kind.
- After projecting midpoints, shape regularity is not guaranteed automatically: tets near the boundary can invert or degenerate. Codes then need repair or smoothing. Any repair that is not itself a bisection step leaves the nested-bisection framework.
- For metric-based remeshing, the only convergence result (Carson et al. 2022) is for source problems and relies on an optimization argument rather than on nestedness. No such result exists for eigenproblems or H(curl).

### Gaps
- I did not verify Stevenson's exact n-D initial-labelling condition from the original paper.
- I found no convergence theory for Maxwell eigenvalues under non-nested remeshing or curved bisection.

---

## 4. Convergence and optimality for eigenvalue problems: chronology and conditions

### Takeaway
Elliptic: Garau–Morin–Zuppa (plain convergence from any T₀) and Dai–Xu–Zhou (optimal complexity, fine T₀),
then Dai–He–Zhou and Gallistl (multiple eigenvalues and clusters), Bonito–Demlow (higher order, θ independent
of the cluster), and Carstensen–Gedicke (inexact solve). Mixed and Maxwell: BGGG 2017 (mixed Laplace
clusters, RT/BDM, 2D/3D), then Boffi–Gastaldi 2019 (3D edge elements). Every *rate* result needs a
**sufficiently fine initial mesh** and a **small θ**. Discrete compactness is the convergence prerequisite
on the Maxwell side.

### Cited Findings
- Dai–Xu–Zhou, "Convergence and optimal complexity of adaptive finite element eigenvalue computations", Numer. Math. 110(3) (2008) 313–355, DOI 10.1007/s00211-008-0169-3. The analysis is based on "a certain relationship between the finite element eigenvalue approximation and the associated finite element boundary value approximation". — [PSU record](https://pure.psu.edu/en/publications/convergence-and-optimal-complexity-of-adaptive-finite-element-eig/); [OpenAlex](https://api.openalex.org/works/doi:10.1007%2FS00211-008-0169-3)
- Gallistl, "An optimal adaptive FEM for eigenvalue clusters", Numer. Math. 130 (2015) 467–496, DOI 10.1007/s00211-014-0671-8. Constants are explicit in the initial mesh size, the eigenvalues and the cluster width. — [Matheon preprint](https://opus4.kobv.de/opus4-matheon/frontdoor/index/index/docId/1287); DOI from [BG2019 refs](https://iris.unibs.it/retrieve/93b0a02a-19e5-4e94-b044-61128c442618/18m1179389.pdf)
- Boffi–Gallistl–Gardini–Gastaldi, "Optimal convergence of adaptive FEM for eigenvalue clusters in mixed form", Math. Comp. 86 (2017) 2213–2237, DOI 10.1090/mcom/3212 (arXiv:1504.06418). It covers RT/BDM of any fixed degree in 2D and 3D and is "cluster robust". Quasi-orthogonality comes from a superconvergence result, ‖P_h p − p_h‖ ≤ ρ_sc(h)(…). — [AMS](https://www.ams.org/mcom/2017-86-307/S0025-5718-2017-03212-9/); [arXiv](https://arxiv.org/pdf/1504.06418); [WONAPDE abstract](https://www.ci2ma.udec.cl/wonapde2016/plenary_speakers/abstract/Boffi.pdf)
- BGGG assumptions: initial mesh-size small (e.g. ε ≤ √(1+1/(2N)) − 1, C_qo ρ(h₀) ≤ 1/2), θ small, refinement "by using the refinement rules of [BDD, Stevenson]", and discrete eigenvalues outside the cluster separated from it (K = sup λ_j/|λ_j − λ_{h,k}| < ∞). The theorem bounds the cluster gap δ(W, W_ℓ)(card T_ℓ − card T₀)^s ≲ |W|_{A_s}, with eigenvalue error ≲ δ². — [ar5iv 1504.06418](https://ar5iv.arxiv.org/html/1504.06418)
- Bonito–Demlow, "Convergence and optimality of higher-order adaptive finite element methods for eigenvalue clusters" (arXiv:1508.06265; SIAM J. Numer. Anal. 54 (2016), per my recollection, unverified). It closes the gap that earlier arguments did not prove equivalence of practical and theoretical estimators for clusters with higher-order elements or variable coefficients. — [arXiv:1508.06265](https://arxiv.org/abs/1508.06265)
- Discrete compactness: Boffi–Costabel–Dauge–Demkowicz show that discrete compactness plus approximability gives convergence of discrete eigenpairs. They prove it for hp-adaptive rectangular edge elements including 1-irregular (hanging-node) meshes with varying order. Caorsi–Fernandes–Raffetto (SIAM J. Numer. Anal. 38 (2000) 580–607, DOI 10.1137/S0036142999357506) showed it is "not only a sufficient but also a necessary condition for the convergence of Maxwell eigenvalues without… spurious mode[s]". — [BCDD 2005 PDF](https://perso.univ-rennes1.fr/monique.dauge/publis/BoCoDaDe05.pdf); [BG2019 refs](https://iris.unibs.it/retrieve/93b0a02a-19e5-4e94-b044-61128c442618/18m1179389.pdf)
- Boffi, "Fortin operator and discrete compactness for edge elements", Numer. Math. 87 (2000) 229–246, DOI 10.1007/s002110000182, and Boffi, Acta Numer. 19 (2010) 1–120, DOI 10.1017/S0962492910000012. These are the standard references for the discrete-compactness machinery. Boffi–Gastaldi rely on the regularity H^s, s > 1/2, of the mixed source problem on Lipschitz polyhedra and on the error bound ‖σ_g − σ_{g,h}‖ + ‖p_g − p_{g,h}‖ ≤ C h^s‖g‖ (eq. 4.6). — [BG2019](https://iris.unibs.it/retrieve/93b0a02a-19e5-4e94-b044-61128c442618/18m1179389.pdf)

### Inferences
- "Fine initial mesh" in practice means that, on T₀, the discrete spectrum already separates the target cluster from its neighbours, i.e. the mode ordering and mode count in the window are correct. This makes ρ(h₀) small enough for the higher-order terms. If T₀ is too coarse, mode tracking across levels can switch modes, and the proofs do not apply.
- On graded, shape-regular simplicial meshes, Nédélec spaces with bounded commuting quasi-interpolants (Schöberl; Falk–Winther) give discrete compactness uniformly along the AFEM sequence. This is what makes the edge-element AFEM spurious-free. Hanging-node meshes need separate proofs (BCDD for rectangles).

### Gaps
- I did not verify the Bonito–Demlow journal reference or the Dai–He–Zhou IMA J. Numer. Anal. 2015 multiple-eigenvalue paper beyond secondary mentions.

---

## 5. Clustered and degenerate eigenvalues

### Takeaway
Single-mode estimators fail inside clusters and degenerate eigenspaces. Discrete modes rotate within the
(near-)degenerate space from level to level, and estimating only one of them misses error in the others,
which produces degenerate or suboptimal convergence. The cure that has been proven is to estimate and mark
the **whole cluster**: the indicator Σ_{j∈J} η²_{ℓ,j}(T), measured in the gap δ(W, W_ℓ) between spaces.

### Cited Findings
- "degeneracy of the convergence may be observed when the error estimator is computed by taking into account only a subset of the discrete eigenmodes approximating the eigensolutions we are interested in (multiple or belonging to a cluster) [Solin–Giani 2012, Gallistl 2015, BDGG 2017]". — [Boffi–Gastaldi 2019](https://iris.unibs.it/retrieve/93b0a02a-19e5-4e94-b044-61128c442618/18m1179389.pdf)
- "adaptive schemes driven by the error indicator associated to an individual eigenvalue may produce unsatisfactory results". BGGG mark with θΣ_j η²_{ℓ,j}(T_ℓ) ≤ Σ_j η²_{ℓ,j}(M_ℓ). — [ar5iv 1504.06418](https://ar5iv.arxiv.org/html/1504.06418)
- Practical AFEM "immediately leads to eigenvalue clusters", which "require simultaneous marking". The earlier Dai–He–Zhou analysis (arXiv:1210.1846) marks by the element-wise sum over the multiple eigenvalues. — [Gallistl Matheon](https://opus4.kobv.de/opus4-matheon/frontdoor/index/index/docId/1287)
- Solin & Giani, "An iterative adaptive finite element method for elliptic eigenvalue problems", J. Comput. Appl. Math. 236 (2012) 4582–4599, DOI 10.1016/j.cam.2012.05.002. Cited by Boffi–Gastaldi as observing degeneracy with single-mode estimation. — [BG2019 refs](https://iris.unibs.it/retrieve/93b0a02a-19e5-4e94-b044-61128c442618/18m1179389.pdf)
- Boffi–Durán–Gardini–Gastaldi, "A posteriori error analysis for nonconforming approximation of multiple eigenvalues", Math. Methods Appl. Sci. 40 (2017) 350–369, DOI 10.1002/mma.3452 (arXiv:1404.5560). — [BG2019 refs](https://iris.unibs.it/retrieve/93b0a02a-19e5-4e94-b044-61128c442618/18m1179389.pdf)
- Cancès–Dusson–Maday–Stamm–Vohralík, "Guaranteed a posteriori bounds for eigenvalues and eigenvectors: multiplicities and clusters", Math. Comp. 89 (2020) 2563–2611 (arXiv:2008.04140). The error measures are the *sum of eigenvalues* in the cluster and the *density matrix*, the orthogonal projector onto the eigenspace. Both are invariant to rotations within the cluster. — [arXiv:2008.04140](https://arxiv.org/pdf/2008.04140); [HAL](https://hal.archives-ouvertes.fr/hal-02127954)
- Giani–Grubišić–Ovall benchmark studies use invariant-subspace reliability estimates "to justify the observed robustness of eigenvalue error estimates in the presence of repeated or clustered eigenvalues". — [search summary, Durham repository](https://durham-repository.worktribe.com/output/1497342)

### Inferences
- For RF cavities, where dipole pairs are degenerate in a symmetric cavity and near-degenerate after small perturbations, the marking quantity should be the sum over all modes of the window, and convergence should be judged on cluster-invariant quantities: the sum or mean of θ_k in the cluster, or subspace angles. Per-mode quantities whose individual eigenvectors rotate between levels are a poor measure.

### Gaps
- I found no Maxwell-specific paper quantifying single-mode failure on 3D cavities.

---

## 6. Stopping criteria, effectivity, guaranteed bounds, algebraic-error balance

### Takeaway
Proven rules stop on an a posteriori estimator (η ≤ TOL), ideally a *guaranteed* bound. Stopping on
successive-level differences is not backed by theory and fails when the mesh stalls. The algebraic
eigensolver should be terminated relative to the discretization estimator
(|λ_ℓ − λ̃_ℓ| + |||u_ℓ − ũ_ℓ|||² ≤ ω η²_ℓ with small ω, Carstensen–Gedicke) rather than to a fixed tight
tolerance.

### Cited Findings
- Carstensen–Gedicke, "An adaptive finite element eigenvalue solver of asymptotic quasi-optimal computational complexity", SIAM J. Numer. Anal. 50(3) (2012) 1029–1057, DOI 10.1137/090769430 (full text read). AFEMES has "a proper termination criterion for the algebraic eigenvalue solver and does not need any coarsening". The inexact iterates must satisfy |||u_ℓ − ũ_ℓ|||² + |λ_ℓ − λ̃_ℓ| ≤ ω η²_ℓ(λ̃_ℓ, ũ_ℓ) "for sufficiently small ω > 0". The tolerance is then halved inside a while-loop until ε_ℓ ≤ √ω η_ℓ, and the next level is started from the prolonged iterate. The resulting runtime is linear, t_ℓ ≈ N_ℓ, if the solver (e.g. multigrid-preconditioned PINVIT/LOBPCG) contracts independently of N. Experiments use θ = 0.5 and ω = 10⁻³. ω = 1 was "not small enough". — [CG 2012](https://ins.uni-bonn.de/media/public/publication-media/CG_2012.pdf?pk=1445)
- The same paper notes that the earlier convergence and optimality works (DXZ08, GMZ09, …) "do assume unrealistically the exact knowledge of algebraic eigenpairs". — [CG 2012](https://ins.uni-bonn.de/media/public/publication-media/CG_2012.pdf?pk=1445)
- Miedlar's thesis, *Inexact Adaptive Finite Element Methods for Elliptic PDE Eigenvalue Problems* (TU Berlin; related work with Mehrmann, e.g. Mehrmann–Miedlar, Numer. Linear Algebra Appl. 2011, per Carstensen–Gedicke's "[MM11]"), gives a balancing algorithm in which the eigensolver stops once the discrete (algebraic) part of the estimator no longer dominates the continuous part ("equilibrating strategy"). It also gives a three-way adaptive homotopy for non-self-adjoint problems that balances discretization, iteration and homotopy errors. — [depositonce TU Berlin](https://depositonce.tu-berlin.de/items/4f67c06f-9966-43d9-98cc-116cba5bdca8); [HU edoc](https://edoc.hu-berlin.de/bitstreams/7235eacd-fc73-4929-a3d7-7cb5f2a57bc9/download)
- Arioli–Liesen–Miedlar–Strakoš survey the interplay of discretization and algebraic error: "the algebraic error should be incorporated in the context of the AFEM and its a posteriori error estimators". — [Charles Univ. seminar record](https://seminar-nm.karlin.mff.cuni.cz/science/result/2613)
- Cancès–Dusson–Maday–Stamm–Vohralík, "Guaranteed and robust a posteriori bounds for Laplace eigenvalues and eigenvectors: conforming approximations", SIAM J. Numer. Anal. (2017). "Inexact algebraic solvers are taken into account; the estimates are valid on each iteration and can serve for the design of adaptive stopping criteria." The unified framework, Numer. Math. 140(4) (2018) 1033–1079, DOI 10.1007/s00211-018-0984-0, gives guaranteed upper and lower bounds on a simple eigenvalue under "the sole condition that the approximate i-th eigenvalue lies between the exact (i−1)-th and (i+1)-th eigenvalue" with sufficiently large relative gaps. — [Springer](https://link.springer.com/doi/10.1007/s00211-018-0984-0); [IP Paris](https://researchportal.ip-paris.fr/en/publications/guaranteed-and-robust-a-posteriori-bounds-for-laplace-eigenvalues-2/)
- Maxwell guaranteed lower bounds: Gallistl–Olkhovskiy, "Computational lower bounds of the Maxwell eigenvalues" (arXiv:2110.02605; the author page lists 2023), for lowest-order Nédélec in 2D/3D. It extends Liu–Oishi using a hypercircle argument plus a perturbation argument with Falk–Winther commuting quasi-interpolation and computable stability constants. Conforming Galerkin gives upper bounds, and "guaranteed lower bounds has remained open" before this work. — [arXiv:2110.02605](https://arxiv.org/pdf/2110.02605); [author PDF](https://users.fmi.uni-jena.de/~gallistl/publikationen/pdf/2023-go-maxw.pdf)
- An hp-adaptive strategy with inexact solvers and a computable guaranteed bound on the error reduction factor (Daniel–Ern–Smears–Vohralík line) shows that the stopping and adaptivity decisions can both be driven by guaranteed quantities (elliptic source problem). — [HAL hal-01931448](https://hal.archives-ouvertes.fr/hal-01931448)
- Effectivity: Boffi–Gastaldi reliability and efficiency hold with h-dependent higher-order terms and only "for h small enough". For the primal Maxwell estimator, efficiency is not local. So effectivity indices η/error approach a constant only asymptotically. — [BG2019](https://iris.unibs.it/retrieve/93b0a02a-19e5-4e94-b044-61128c442618/18m1179389.pdf)
- Carstensen–Gedicke's experiments stop at "a tolerance of 10⁻⁹ in the eigenvalue error… or the number of degrees of freedom exceeds 10⁶", i.e. an error tolerance plus a DOF budget. — [CG 2012](https://ins.uni-bonn.de/media/public/publication-media/CG_2012.pdf?pk=1445)

### Inferences
- **Risks of difference-based stopping** (|λ_ℓ − λ_{ℓ−1}| < tol): (i) if REFINE changes nothing, because nothing was marked or refinement failed or was rolled back, the difference is exactly 0 and the loop "converges" falsely. Carstensen–Gedicke's small-θ "stair" (the previous value accepted for the next level) is a documented instance. (ii) Adaptive refinement in a region that barely affects λ can stall the difference while the error elsewhere, e.g. at a re-entrant edge, is still large. (iii) The difference is roughly the error reduction per step, |λ_ℓ − λ| (1 − ρ), so it understates the error by 1/(1 − ρ) when the contraction ρ is close to 1. (iv) Inside a degenerate cluster, per-mode differences jump because of mode swapping. A safe rule combines η_ℓ ≤ TOL (or a guaranteed bound), a check that the mesh actually grew (#T_{ℓ+1} > #T_ℓ and every marked element refined), and cluster-invariant quantities.
- For eigenvalues computed with a conforming Galerkin/Ritz method, λ_h ≥ λ (upper bound, min-max). A difference-based rule can be complemented by a lower bound (Gallistl–Olkhovskiy for Maxwell) to bracket λ.

### Gaps
- I did not retrieve the exact Mehrmann–Miedlar 2011 citation (journal and pages) directly. It is cited as "[MM11]" in Carstensen–Gedicke, which describes it as a combined AFEM and linear-algebra algorithm "assuming a saturation assumption".
- I found no paper that formally analyses false convergence of difference-based stopping. The points above are inferences.

---

## 7. hp-adaptivity for (Maxwell) eigenvalue problems

### Takeaway
Exponential convergence of hp-FEM for Maxwell eigenvalues is proven in 2D polygons (Costabel–Dauge–Schwab,
weighted regularization). For edge elements, discrete compactness on hp meshes must also be shown (BCDD,
rectangles with hanging nodes). Proven hp-AFEM *algorithms* (Canuto–Nochetto–Stevenson–Verani) exist for
source problems in 1D/2D and use a coarsening step (Binev's hp-tree). Practical hp eigen-adaptivity
(Giani–Hall DG; Harmon–Notaroš; Jiang–Zhang–Zheng tetrahedral) uses smoothness estimation to choose h or p
and reports exponential rates.

### Cited Findings
- Costabel–Dauge–Schwab (2004/2005): exponential convergence of hp-FEM for weighted regularization of Maxwell in plane polygons, and "as a direct consequence… exponential convergence of hp-FEM approximations to Maxwell eigenvalues and eigenvectors". With edge elements, approximation "has to be combined with the proof of the discrete compactness property, which is not obvious". — [CDS PDF](https://costabel.perso.math.cnrs.fr/publis/CoDaSchwab04.pdf)
- Boffi–Costabel–Dauge–Demkowicz: discrete compactness for hp edge elements on 1-irregular rectangular meshes with variable order. This also gives a rigorous convergence proof for the p-version. — [BCDD PDF](https://perso.univ-rennes1.fr/monique.dauge/publis/BoCoDaDe05.pdf); Demkowicz's [seminar slides](https://www.math.mcgill.ca/humphries/applied/seminar/2006/leszek2.pdf)
- Boffi, CMAME 196 (2007) 3672–3681, DOI 10.1016/j.cma.2006.10.024: discrete compactness and hp mixed finite elements. — [BG2019 refs](https://iris.unibs.it/retrieve/93b0a02a-19e5-4e94-b044-61128c442618/18m1179389.pdf)
- Bürg, Appl. Numer. Math. 72 (2013) 188–204, DOI 10.1016/j.apnum.2013.04.008: convergence of an automatic hp-adaptive strategy for Maxwell's equations (source problem). — [BG2019 refs](https://iris.unibs.it/retrieve/93b0a02a-19e5-4e94-b044-61128c442618/18m1179389.pdf)
- Canuto–Nochetto–Stevenson–Verani, "Convergence and optimality of hp-AFEM" (MOX report 14/2015; n = 1, 2). The method alternates a near-best hp-approximation (Binev's algorithm, which coarsens) with refinement to a comparable accuracy, and proves "convergence and instance optimality". p-robustness is a practical concern. — [MOX](https://mox.polimi.it/reports-and-books/publication-results/?id=539)
- Giani–Hall, M3AS (2012): residual estimator for hp-adaptive DG for Laplace eigenvalues (2D/3D), reliable and efficient. hp-adaptation driven by it "can achieve exponential convergence, even for non-smooth eigenfunctions". — [Nottingham repository](https://nottingham-repository.worktribe.com/output/1010717/an-a-posteriori-error-estimator-for-hp-adaptive-discontinuous-galerkin-methods-for-elliptic-eigenvalue-problems)
- For multiple eigenvectors per element, one hp strategy picks "an eigenvector which contributes the most to the total element's error" and bases the h/p decision on that vector's smoothness. — [arXiv:1612.02305](https://arxiv.org/pdf/1612.02305)
- Harmon–Notaroš: goal-oriented isotropic and anisotropic hp-refinement for 2D Maxwell eigenvalues, with high accuracy even for singular eigenfunctions. Jiang–Zhang–Zheng (2013): hp-adaptivity on tetrahedra for point and edge singularities and an engineering benchmark. — [Harmon–Notaroš TechRxiv](https://atmos.colostate.edu/~notaros/Papers/TechRxiv_adaptive_aniso_hp_paper.pdf)
- 2026: Liang–Xu–Zhang, "Adaptive multilevel methods for the Maxwell eigenvalue problem" (arXiv:2603.29718, 31 Mar 2026). It proposes a preconditioned Helmholtz–Jacobi–Davidson eigensolver with a local multilevel preconditioner on adaptive meshes, with a convergence factor "independent of mesh size and mesh level if the coarse mesh is sufficiently fine". It also highlights that the large curl kernel makes the per-level eigensolve the bottleneck. — [arXiv:2603.29718](https://arxiv.org/abs/2603.29718)

### Inferences
- In 3D Maxwell, corner and edge singularities, e.g. at re-entrant edges of iris or coupler geometries, need geometric h-grading toward the edges combined with p increase away from them. Isotropic h-AFEM only reaches algebraic rates. There is no rigorous 3D hp-AFEM optimality proof for Maxwell eigenproblems; the results are 2D and/or source-problem results.

### Gaps
- I did not verify the bibliographic details of Demkowicz's 3D hp Maxwell codes (*Computing with hp-Adaptive Finite Elements*, Vol. 2) or Schwab's book in this session.
- I did not verify the Numer. Math. journal publication of Canuto–Nochetto–Stevenson–Verani.
