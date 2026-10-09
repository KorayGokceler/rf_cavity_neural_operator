// Typed client of src/service/api.py. Large arrays arrive as base64 little-endian float32 / uint32.

export interface ModelInfo {
  field: 'E' | 'H'; n_modes: number; n_params: number; device: string; untrained: boolean; checkpoint: string | null;
}
export interface Info {
  version: string; model: ModelInfo; families: { train: string[]; ood: string[] };
  limits: { max_upload_mb: number; max_tets: number; cad: string[]; mesh: string[] };
  conventions: Record<string, string | number>;
}
export interface Check {
  ok: boolean; problems: string[]; n_nodes: number; n_tets: number; volume_cm3?: number;
  bbox_mm?: [number[], number[]]; betti1?: number; n_bnd_components?: number;
}
export interface Surface { points: Float32Array; triangles: Uint32Array }
export interface Solution {
  id: string; source: 'model' | 'fe' | 'diff'; field: 'E' | 'H'; n_edges: number; modes: ModeRow[];
  axis_length_mm: number; f_next?: number | null; labels?: Record<string, number>[] | null;
}
export interface Geometry {
  id: string; source: Record<string, unknown>; check: Check; family?: string;
  params?: Record<string, number>; surface: Surface;
  truth?: Solution | null; field?: 'E' | 'H'; predictable?: boolean;
}
export interface QoiCmp { fe: number | null; model: number | null; rel_err: number | null }
export interface CmpRow {
  mode: number; f_fe: number; f_model: number; f_rel_err: number; rel_l2: number; degenerate: boolean;
  accelerating: boolean; eta: number | null; qoi: Record<string, QoiCmp>;
}
export interface Features {
  names: string[]; min: number[]; max: number[]; n_vertices: number; n_edges: number; scale_mm: number;
  torsion_max: number;
}
export interface DatasetRef { id: string; name: string; kind: 'h5' | 'pkl' }
export interface DatasetRow {
  id: number; family: string; n_nodes: number; n_edges: number; n_tets: number; betti1: number;
  deformed: boolean | null; f_GHz: number[]; field: string;
}
export interface DatasetStats {
  total: number; fields: string[]; n_modes: number;
  families: Record<string, { count: number; f0: number[]; n_edges: number[] }>;
}
export interface ModeRow {
  mode: number; f_GHz: number; degenerate: boolean; eta: number | null; accelerating: boolean;
  Q0: number | null; G_ohm: number | null; R_over_Q_ohm: number | null; R_sh_ohm: number | null;
  T_transit: number | null; Epk_Eacc: number | null; Bpk_Eacc_mT_per_MVm: number | null;
}
export interface Prediction {
  id: string; geometry_id: string; source: 'model'; field: 'E' | 'H'; n_edges: number; modes: ModeRow[];
  axis_length_mm: number; time_s: Record<string, number>; warnings: string[]; model: ModelInfo;
  comparison?: { rows: CmpRow[]; diff_id: string; truth_id: string };
}
export interface ModeData {
  mode: number; f_GHz: number;
  surface: { E: Float32Array; H: Float32Array; E_max: number; H_max: number };
  axis: { z_mm: number[]; Ez: (number | null)[] };
}
export interface Plane { points: Float32Array; triangles: Uint32Array; E: Float32Array; H: Float32Array }

function f32(b64: string): Float32Array {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const u8 = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) u8[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}
function u32(b64: string): Uint32Array {
  return new Uint32Array(f32(b64).buffer);
}

async function call<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { const j = await r.json(); if (j.detail) msg = typeof j.detail === 'string' ? j.detail : JSON.stringify(j.detail); } catch { /* not JSON */ }
    throw new Error(msg);
  }
  return r.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
});

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function geometry(j: any): Geometry {
  return { ...j, surface: { points: f32(j.surface.points_b64), triangles: u32(j.surface.triangles_b64) } };
}

export const api = {
  info: () => call<Info>('/api/info'),
  upload: async (file: File, unit: string, meshSize: number, nCells: number) => {
    const fd = new FormData();
    fd.append('file', file);
    fd.append('unit', unit);
    fd.append('mesh_size', String(meshSize));
    fd.append('n_cells', String(nCells));
    return geometry(await call('/api/geometries', { method: 'POST', body: fd }));
  },
  sample: async (family: string, id: number | null, meshSize: number) =>
    geometry(await call('/api/geometries/sample', json({ family, id, mesh_size: meshSize }))),
  predict: (geometryId: string, compareTo: string | null = null) =>
    call<Prediction>('/api/predictions', json({ geometry_id: geometryId, compare_to: compareTo })),
  features: (gid: string) => call<Features>(`/api/geometries/${gid}/features`),
  featurePlane: async (gid: string, name: string, axis: string, pos: number | null, res = 121) => {
    const q = new URLSearchParams({ name, axis, res: String(res) });
    if (pos !== null) q.set('pos', String(pos));
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const j: any = await call(`/api/geometries/${gid}/feature_plane?${q}`);
    return { points: f32(j.points_b64), triangles: u32(j.triangles_b64), values: f32(j.values_b64) };
  },
  featureSurface: async (gid: string, name: string) => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const j: any = await call(`/api/geometries/${gid}/feature_surface?${new URLSearchParams({ name })}`);
    return f32(j.values_b64);
  },
  datasets: () => call<DatasetRef[]>('/api/datasets'),
  datasetItems: (did: string, p: { family?: string; offset?: number; limit?: number; sort?: string; desc?: boolean }) => {
    const q = new URLSearchParams();
    Object.entries(p).forEach(([k, v]) => v !== undefined && v !== '' && q.set(k, String(v)));
    return call<{ total: number; offset: number; items: DatasetRow[] }>(`/api/datasets/${did}/items?${q}`);
  },
  datasetStats: (did: string) => call<DatasetStats>(`/api/datasets/${did}/stats`),
  datasetOpen: async (did: string, sid: number) =>
    geometry(await call(`/api/datasets/${did}/items/${sid}/open`, { method: 'POST' })),
  mode: async (pid: string, k: number): Promise<ModeData> => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const j: any = await call(`/api/predictions/${pid}/modes/${k}`);
    return { ...j, surface: { ...j.surface, E: f32(j.surface.E_b64), H: f32(j.surface.H_b64) } };
  },
  plane: async (pid: string, k: number, axis: string, pos: number | null, res = 121): Promise<Plane> => {
    const q = new URLSearchParams({ mode: String(k), axis, res: String(res) });
    if (pos !== null) q.set('pos', String(pos));
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const j: any = await call(`/api/predictions/${pid}/plane?${q}`);
    return { points: f32(j.points_b64), triangles: u32(j.triangles_b64), E: f32(j.E_b64), H: f32(j.H_b64) };
  },
  exportUrl: (pid: string) => `/api/predictions/${pid}/export`,
};
