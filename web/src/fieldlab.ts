// Client-side field engine of the workspace: every mode of a solution arrives once (GET
// /api/predictions/{id}/fields): the HCurl p2 field at the P2 nodes of the curved mesh, the 8 linear sub-tets per
// tet on those nodes, and |E|, |H| on the curved wall; cut planes, the 3D volume grid, modes, phase and E/H are
// computed here, so no interaction waits for the network.  The cells are linear, so everything below treats
// `nodes` / `tets` as a plain tetrahedral mesh with one vector per node.
//
//   slicePlane   tets ∩ {x_axis = pos} → triangles; each point on a mesh edge (a, b) at t: F = (1−t)F_a + tF_b
//   rasterize    regular grid over the bbox; each tet visits the grid points of its box (barycentric test)
//   sampleSlice / sampleGrid / gather   one mode's vectors at those points (linear, as the display mesh)
//   sliceArrows / thinGrid               the arrow positions (a square grid in the cut / every n-th grid line)

export interface FieldBundle {
  K: number; nv: number; ns: number;   // modes, display nodes, wall points
  nodes: Float32Array;                 // [Nv,3] mm: the P2 nodes of the curved mesh
  tets: Uint32Array;                   // [Nc,4]: 8 linear sub-tets per element
  E: Float32Array; H: Float32Array;    // [K,Nv,3] physical (V/m, A/m), U = 1 J
  surfE: Float32Array; surfH: Float32Array;   // [K,Ns] |E|, |H| at the geometry's curved wall points
  Emax: number[]; Hmax: number[];      // per mode, max over the nodes and the wall
  f: number[];
  axis: { z: number[]; Ez: (number | null)[][] };
}

export interface Slice {
  axis: 0 | 1 | 2; pos: number;        // the plane x_axis = pos [mm]
  points: Float32Array;                // [P,3] mm
  triangles: Uint32Array;              // [T,3]
  a: Uint32Array; b: Uint32Array; t: Float32Array;   // point p = (1 − t)·x_a + t·x_b
}

export interface Grid {
  dims: [number, number, number]; origin: [number, number, number]; spacing: number;
  idx: Int32Array;                     // [M] flat grid index (x fastest) of the points inside the mesh
  tv: Uint32Array; bw: Float32Array;   // [M,4] vertices of the containing tet and the barycentric weights
}

let HALF: Float32Array | null = null;  // float16 → float32 table (65536 entries)
function halfTable(): Float32Array {
  if (HALF) return HALF;
  HALF = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
    HALF[h] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return HALF;
}

/** float16 [K,per] (each mode divided by scale[k]) → float32, physical units. */
export function decodeHalf(u16: Uint16Array, K: number, per: number, scale: number[]): Float32Array {
  const tab = halfTable();
  const out = new Float32Array(u16.length);
  for (let k = 0; k < K; k++) {
    const s = scale[k];
    for (let i = k * per; i < (k + 1) * per; i++) out[i] = tab[u16[i]] * s;
  }
  return out;
}

/** float16 [K,Nv,3] vectors → float32. */
export function decodeModes(u16: Uint16Array, K: number, nv: number, scale: number[]): Float32Array {
  return decodeHalf(u16, K, 3 * nv, scale);
}

/** Mode k of E or H: a view [Nv,3]. */
export function modeField(b: FieldBundle, field: 'E' | 'H', k: number): Float32Array {
  const F = field === 'E' ? b.E : b.H;
  return F.subarray(3 * b.nv * k, 3 * b.nv * (k + 1));
}

/** Mode k of |E| or |H| on the wall points: a view [Ns]. */
export function modeSurf(b: FieldBundle, field: 'E' | 'H', k: number): Float32Array {
  const F = field === 'E' ? b.surfE : b.surfH;
  return F.subarray(b.ns * k, b.ns * (k + 1));
}

export function slicePlane(b: FieldBundle, axis: 0 | 1 | 2, pos: number): Slice {
  const { nodes, tets, nv } = b;
  const map = new Map<number, number>();
  const pa: number[] = [], pb: number[] = [], pt: number[] = [], pts: number[] = [], tri: number[] = [];
  const at = (u: number, w: number) => {                 // the plane point on edge (u, w), shared by its tets
    const lo = Math.min(u, w), hi = Math.max(u, w);
    const key = lo * nv + hi;
    let i = map.get(key);
    if (i !== undefined) return i;
    const dl = nodes[3 * lo + axis] - pos, dh = nodes[3 * hi + axis] - pos;
    const t = dl === dh ? 0 : dl / (dl - dh);
    i = pa.length;
    pa.push(lo); pb.push(hi); pt.push(t);
    for (let c = 0; c < 3; c++) pts.push(nodes[3 * lo + c] + t * (nodes[3 * hi + c] - nodes[3 * lo + c]));
    map.set(key, i);
    return i;
  };
  const v = [0, 0, 0, 0];
  const neg: number[] = [], posv: number[] = [];
  for (let e = 0; e < tets.length / 4; e++) {
    neg.length = 0; posv.length = 0;
    for (let i = 0; i < 4; i++) {
      v[i] = tets[4 * e + i];
      (nodes[3 * v[i] + axis] - pos < 0 ? neg : posv).push(v[i]);
    }
    if (neg.length === 0 || posv.length === 0) continue;
    if (neg.length === 2) {                             // a quad: n0p0 → n0p1 → n1p1 → n1p0
      const q = [at(neg[0], posv[0]), at(neg[0], posv[1]), at(neg[1], posv[1]), at(neg[1], posv[0])];
      tri.push(q[0], q[1], q[2], q[0], q[2], q[3]);
    } else {                                             // a triangle around the lone vertex
      const [one, rest] = neg.length === 1 ? [neg[0], posv] : [posv[0], neg];
      tri.push(at(one, rest[0]), at(one, rest[1]), at(one, rest[2]));
    }
  }
  return { axis, pos, points: new Float32Array(pts), triangles: new Uint32Array(tri), a: new Uint32Array(pa),
           b: new Uint32Array(pb), t: new Float32Array(pt) };
}

export function sampleSlice(s: Slice, F: Float32Array): Float32Array {
  const n = s.t.length;
  const out = new Float32Array(3 * n);
  for (let p = 0; p < n; p++) {
    const a = 3 * s.a[p], b = 3 * s.b[p], t = s.t[p];
    for (let c = 0; c < 3; c++) out[3 * p + c] = (1 - t) * F[a + c] + t * F[b + c];
  }
  return out;
}

export function rasterize(b: FieldBundle, nPoints = 80000): Grid {
  const { nodes, tets } = b;
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < nodes.length; i += 3) for (let c = 0; c < 3; c++) {
    lo[c] = Math.min(lo[c], nodes[i + c]); hi[c] = Math.max(hi[c], nodes[i + c]);
  }
  const ext = [0, 1, 2].map((c) => Math.max(hi[c] - lo[c], 1e-9));
  const d = Math.cbrt((ext[0] * ext[1] * ext[2]) / nPoints);
  const dims = ext.map((x) => Math.max(2, Math.ceil(x / d) + 1)) as [number, number, number];
  const [nx, ny] = dims;
  const N = dims[0] * dims[1] * dims[2];
  const owner = new Int32Array(N).fill(-1);
  const wts = new Float32Array(4 * N);
  const P = new Float64Array(12);
  for (let e = 0; e < tets.length / 4; e++) {
    const mn = [Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < 4; i++) {
      const v = tets[4 * e + i];
      for (let c = 0; c < 3; c++) {
        const x = nodes[3 * v + c];
        P[3 * i + c] = x; mn[c] = Math.min(mn[c], x); mx[c] = Math.max(mx[c], x);
      }
    }
    const i0 = [0, 1, 2].map((c) => Math.max(0, Math.ceil((mn[c] - lo[c]) / d - 1e-9)));
    const i1 = [0, 1, 2].map((c) => Math.min(dims[c] - 1, Math.floor((mx[c] - lo[c]) / d + 1e-9)));
    if (i0[0] > i1[0] || i0[1] > i1[1] || i0[2] > i1[2]) continue;
    // inverse of J = [p1 − p0, p2 − p0, p3 − p0] (columns)
    const a = P[3] - P[0], bb = P[6] - P[0], c3 = P[9] - P[0];
    const dd = P[4] - P[1], ee = P[7] - P[1], ff = P[10] - P[1];
    const g = P[5] - P[2], h = P[8] - P[2], k = P[11] - P[2];
    const det = a * (ee * k - ff * h) - bb * (dd * k - ff * g) + c3 * (dd * h - ee * g);
    if (Math.abs(det) < 1e-30) continue;
    const r = 1 / det;
    const m = [(ee * k - ff * h) * r, (c3 * h - bb * k) * r, (bb * ff - c3 * ee) * r,
               (ff * g - dd * k) * r, (a * k - c3 * g) * r, (c3 * dd - a * ff) * r,
               (dd * h - ee * g) * r, (bb * g - a * h) * r, (a * ee - bb * dd) * r];
    for (let kz = i0[2]; kz <= i1[2]; kz++) for (let jy = i0[1]; jy <= i1[1]; jy++) for (let ix = i0[0]; ix <= i1[0]; ix++) {
      const x = lo[0] + d * ix - P[0], y = lo[1] + d * jy - P[1], z = lo[2] + d * kz - P[2];
      const l1 = m[0] * x + m[1] * y + m[2] * z, l2 = m[3] * x + m[4] * y + m[5] * z, l3 = m[6] * x + m[7] * y + m[8] * z;
      const l0 = 1 - l1 - l2 - l3;
      if (l0 < -1e-9 || l1 < -1e-9 || l2 < -1e-9 || l3 < -1e-9) continue;
      const gi = ix + nx * (jy + ny * kz);
      owner[gi] = e;
      wts[4 * gi] = l0; wts[4 * gi + 1] = l1; wts[4 * gi + 2] = l2; wts[4 * gi + 3] = l3;
    }
  }
  let M = 0;
  for (let i = 0; i < N; i++) if (owner[i] >= 0) M++;
  const idx = new Int32Array(M), tv = new Uint32Array(4 * M), bw = new Float32Array(4 * M);
  let j = 0;
  for (let i = 0; i < N; i++) {
    const e = owner[i];
    if (e < 0) continue;
    idx[j] = i;
    for (let q = 0; q < 4; q++) { tv[4 * j + q] = tets[4 * e + q]; bw[4 * j + q] = wts[4 * i + q]; }
    j++;
  }
  return { dims, origin: [lo[0], lo[1], lo[2]], spacing: d, idx, tv, bw };
}

export function sampleGrid(g: Grid, F: Float32Array): Float32Array {
  const M = g.idx.length;
  const out = new Float32Array(3 * M);
  for (let p = 0; p < M; p++) for (let q = 0; q < 4; q++) {
    const v = 3 * g.tv[4 * p + q], w = g.bw[4 * p + q];
    out[3 * p] += w * F[v]; out[3 * p + 1] += w * F[v + 1]; out[3 * p + 2] += w * F[v + 2];
  }
  return out;
}

/** F at the vertices ids [n] → [n,3]. */
export function gather(F: Float32Array, ids: Uint32Array): Float32Array {
  const out = new Float32Array(3 * ids.length);
  for (let i = 0; i < ids.length; i++) for (let c = 0; c < 3; c++) out[3 * i + c] = F[3 * ids[i] + c];
  return out;
}

/** |F| per point of [P,3]. */
export function magnitude(F: Float32Array): Float32Array {
  const out = new Float32Array(F.length / 3);
  for (let i = 0; i < out.length; i++) out[i] = Math.hypot(F[3 * i], F[3 * i + 1], F[3 * i + 2]);
  return out;
}

export function scaled(x: Float32Array, a: number): Float32Array {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = a * x[i];
  return out;
}

/** b with the mesh arrays of an earlier bundle when the mesh is the same (the FE, model and difference
 *  solutions of one geometry): cuts and the 3D grid, cached per mesh, then carry over between them. */
export function shareMesh(b: FieldBundle, others: FieldBundle[]): FieldBundle {
  const same = (x: ArrayLike<number>, y: ArrayLike<number>) => {
    if (x.length !== y.length) return false;
    for (let i = 0; i < x.length; i++) if (x[i] !== y[i]) return false;
    return true;
  };
  for (const o of others) {
    if (o.nv === b.nv && same(o.tets, b.tets) && same(o.nodes, b.nodes)) {
      return { ...b, nodes: o.nodes, tets: o.tets };
    }
  }
  return b;
}

/** Arrow positions on a cut: a square grid of about n points (spacing step [mm]) over it, each inside a
 *  slice triangle: points [m,3] mm, the triangle's slice points ids [m,3] and barycentric weights w [m,3]. */
export function sliceArrows(s: Slice, n: number): { points: Float32Array; ids: Uint32Array; w: Float32Array; step: number } {
  const P = s.points, T = s.triangles, np = P.length / 3;
  if (!T.length) return { points: new Float32Array(0), ids: new Uint32Array(0), w: new Float32Array(0), step: 1 };
  const [u, v] = [0, 1, 2].filter((c) => c !== s.axis);
  let u0 = Infinity, u1 = -Infinity, v0 = Infinity, v1 = -Infinity;
  for (let i = 0; i < np; i++) {
    u0 = Math.min(u0, P[3 * i + u]); u1 = Math.max(u1, P[3 * i + u]);
    v0 = Math.min(v0, P[3 * i + v]); v1 = Math.max(v1, P[3 * i + v]);
  }
  let area = 0;                                          // of the cut (not its box): about n arrows inside
  for (let t = 0; t < T.length / 3; t++) {
    const a = 3 * T[3 * t], b = 3 * T[3 * t + 1], c = 3 * T[3 * t + 2];
    area += Math.abs((P[b + u] - P[a + u]) * (P[c + v] - P[a + v]) - (P[c + u] - P[a + u]) * (P[b + v] - P[a + v])) / 2;
  }
  const h = Math.sqrt(area / Math.max(1, Math.min(np, n))) || 1;
  const nu = Math.floor((u1 - u0) / h) + 1, nv = Math.floor((v1 - v0) / h) + 1;
  const ou = u0 + ((u1 - u0) - (nu - 1) * h) / 2, ov = v0 + ((v1 - v0) - (nv - 1) * h) / 2;   // centred
  const owner = new Int32Array(nu * nv).fill(-1);
  const wts = new Float32Array(3 * nu * nv);
  for (let t = 0; t < T.length / 3; t++) {
    const a = 3 * T[3 * t], b = 3 * T[3 * t + 1], c = 3 * T[3 * t + 2];
    const xa = P[a + u], ya = P[a + v], xb = P[b + u] - xa, yb = P[b + v] - ya, xc = P[c + u] - xa, yc = P[c + v] - ya;
    const det = xb * yc - xc * yb;
    if (Math.abs(det) < 1e-12) continue;
    const i0 = Math.max(0, Math.ceil((Math.min(0, xb, xc) + xa - ou) / h - 1e-9));
    const i1 = Math.min(nu - 1, Math.floor((Math.max(0, xb, xc) + xa - ou) / h + 1e-9));
    const j0 = Math.max(0, Math.ceil((Math.min(0, yb, yc) + ya - ov) / h - 1e-9));
    const j1 = Math.min(nv - 1, Math.floor((Math.max(0, yb, yc) + ya - ov) / h + 1e-9));
    for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) {
      const g = i + nu * j;
      if (owner[g] >= 0) continue;
      const x = ou + h * i - xa, y = ov + h * j - ya;
      const l1 = (x * yc - xc * y) / det, l2 = (xb * y - x * yb) / det, l0 = 1 - l1 - l2;
      if (l0 < -1e-9 || l1 < -1e-9 || l2 < -1e-9) continue;
      owner[g] = t; wts[3 * g] = l0; wts[3 * g + 1] = l1; wts[3 * g + 2] = l2;
    }
  }
  let m = 0;
  for (let g = 0; g < owner.length; g++) if (owner[g] >= 0) m++;
  const points = new Float32Array(3 * m), ids = new Uint32Array(3 * m), w = new Float32Array(3 * m);
  let q = 0;
  for (let g = 0; g < owner.length; g++) {
    const t = owner[g];
    if (t < 0) continue;
    points[3 * q + s.axis] = s.pos;
    points[3 * q + u] = ou + h * (g % nu);
    points[3 * q + v] = ov + h * Math.floor(g / nu);
    for (let k = 0; k < 3; k++) { ids[3 * q + k] = T[3 * t + k]; w[3 * q + k] = wts[3 * g + k]; }
    q++;
  }
  return { points, ids, w, step: h };
}

/** Values V [P,3] at the arrow points of sliceArrows: Σ_k w_k V[ids_k]. */
export function sampleArrows(V: Float32Array, ids: Uint32Array, w: Float32Array): Float32Array {
  const out = new Float32Array(ids.length);
  for (let q = 0; q < ids.length; q += 3) for (let k = 0; k < 3; k++) {
    const i = 3 * ids[q + k], wk = w[q + k];
    out[q] += wk * V[i]; out[q + 1] += wk * V[i + 1]; out[q + 2] += wk * V[i + 2];
  }
  return out;
}

/** The inside grid points (positions in g.idx) on every step-th grid line, step chosen for about n of them. */
export function thinGrid(g: Grid, n: number): { sel: Uint32Array; step: number } {
  const st = Math.max(1, Math.round(Math.cbrt(g.idx.length / n)));
  const [nx, ny] = g.dims;
  const out: number[] = [];
  for (let m = 0; m < g.idx.length; m++) {
    const gi = g.idx[m], i = gi % nx, j = Math.floor(gi / nx) % ny, k = Math.floor(gi / (nx * ny));
    if (i % st === 0 && j % st === 0 && k % st === 0) out.push(m);
  }
  return { sel: new Uint32Array(out), step: st };
}

/** Positions [n,3] mm of the inside grid points sel. */
export function gridPoints(g: Grid, sel: Uint32Array): Float32Array {
  const [nx, ny] = g.dims;
  const out = new Float32Array(3 * sel.length);
  for (let q = 0; q < sel.length; q++) {
    const gi = g.idx[sel[q]];
    out[3 * q] = g.origin[0] + g.spacing * (gi % nx);
    out[3 * q + 1] = g.origin[1] + g.spacing * (Math.floor(gi / nx) % ny);
    out[3 * q + 2] = g.origin[2] + g.spacing * Math.floor(gi / (nx * ny));
  }
  return out;
}

/** Per inside point values [M] → the whole grid (x fastest), 0 outside. */
export function fillGrid(g: Grid, v: Float32Array): Float32Array {
  const out = new Float32Array(g.dims[0] * g.dims[1] * g.dims[2]);
  for (let m = 0; m < g.idx.length; m++) out[g.idx[m]] = v[m];
  return out;
}
