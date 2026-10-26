// Shared types and per-mesh caches of the workspace (cuts and the 3D grid are computed once per mesh).
import { rasterize, slicePlane, type FieldBundle, type Grid, type Slice } from './fieldlab';

export const AXES = ['x', 'y', 'z'] as const;
export type Axis = (typeof AXES)[number];
export type Source = 'fe' | 'model' | 'diff';
export type Display = 'field' | 'feature';
export type FeatureKind = 'position' | 'wall' | 'torsion' | 'mesh' | 'raw';
export const FEATURE_KINDS: FeatureKind[] = ['position', 'wall', 'torsion', 'mesh', 'raw'];
export type View3D = 'cut' | 'volume';              // field on a cut plane | |field| in the whole volume
export const MAX_ARROWS = 700;                      // in the volume
export const CUT_ARROWS = 300;                      // on a cut

// cuts and the 3D grid per mesh (the FE, model and difference solutions of a geometry share one: shareMesh)
const cutCache = new WeakMap<Uint32Array, Map<string, Slice>>();
const gridCache = new WeakMap<Uint32Array, Grid>();

export function cutOf(b: FieldBundle, axis: 0 | 1 | 2, pos: number): Slice {
  let m = cutCache.get(b.tets);
  if (!m) { m = new Map(); cutCache.set(b.tets, m); }
  const key = `${axis}:${pos}`;
  let s = m.get(key);
  if (!s) {
    if (m.size >= 8) m.clear();                    // a slider sweep: keep the recent ones only
    s = slicePlane(b, axis, pos);
    m.set(key, s);
  }
  return s;
}

export function gridOf(b: FieldBundle): Grid {
  let g = gridCache.get(b.tets);
  if (!g) { g = rasterize(b); gridCache.set(b.tets, g); }
  return g;
}

export interface ArrowBase { points: Float32Array; vectors: Float32Array; vmax: number; length: number }
export function arrowBase(points: Float32Array, vectors: Float32Array, length: number): ArrowBase {
  let vmax = 0;
  for (let i = 0; i < vectors.length; i += 3) vmax = Math.max(vmax, Math.hypot(vectors[i], vectors[i + 1], vectors[i + 2]));
  return { points, vectors, vmax, length };
}
