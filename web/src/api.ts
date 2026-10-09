// Typed client of src/service/api.py. Large arrays arrive as base64 little-endian float32 / uint32.

export interface ModelInfo {
  n_modes: number; n_params: number; device: string; untrained: boolean; checkpoint: string | null;
  load_error?: string | null; run?: string | null;
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
  id: string; source: 'model' | 'fe' | 'diff'; n_edges: number; modes: ModeRow[];
  axis_length_mm: number; f_next?: number | null; labels?: Record<string, number>[] | null;
}
export interface Geometry {
  id: string; source: Record<string, unknown>; check: Check; family?: string;
  params?: Record<string, number>; surface: Surface;
  truth?: Solution | null;
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
export interface ScalarMesh { points: Float32Array; triangles: Uint32Array; scalars: Float32Array }
export interface Segments { role: 'grid' | 'axis' | 'marker' | 'iso' | 'arrow'; points: Float32Array }
export interface FeatureView {
  kind: string; plane: ScalarMesh | null; cells: ScalarMesh | null; segments: Segments[];
  range: [number, number]; signed: boolean; neutral: boolean; legend: string;
}
export interface JobsConfig {
  enabled: boolean; gen_root: string | null; cpu_count: number; max_total: number;
  families: { train: string[]; ood: string[] }; blocks: Record<string, number | null>; tag_example: string;
}
export interface GenParams {
  families: string[]; n_total: number; mesh_size: number; n_modes: number; sampling: 'sobol' | 'random';
  deform_prob: number; deform_max: number; seed: number; shard_size: number; workers: number | null;
  convert: boolean; lean: boolean; tag: string | null;
}
export interface Job {
  id: string; status: 'queued' | 'running' | 'done' | 'failed' | 'cancelled';
  params: GenParams & { family: string; tag: string; block: number };
  stage: string; progress: { done: number; total: number; ok: number; failed: number };
  created: number; started: number | null; finished: number | null; error: string | null;
  outputs: string[]; log: string[];
}
export interface Preset { embed_dim: number; n_heads: number; n_layers: number; n_basis: number; params_m: number }
export interface TrainConfig {
  enabled: boolean; runs_root: string | null; gpu: string | null; cpu_count: number;
  presets: Record<string, Preset>; busy: boolean; active: string | null; model: ModelInfo;
}
export interface TrainParams {
  name: string; datasets: string[]; preset: string; epochs: number; batch_size: number; lr: number;
  n_modes: number; patience: number; qoi_weight: number; cache_operators: boolean | null;
}
export type RunStatus = 'preparing' | 'running' | 'finished' | 'stopped' | 'failed' | 'checkpoint' | 'empty';
export interface RunSummary {
  name: string; status: RunStatus; epoch: number; max_epochs: number | null; batch: number | null;
  n_batches: number | null; best_score: number | null; monitor: string | null; val_field_rel_l2: number | null;
  val_freq_rel_err: number | null; n_params: number | null; updated: number; n_ckpt: number; has_last: boolean;
  resumable: boolean; preset: string | null; from_ui: boolean; error: string | null;
}
export interface RunDetail extends RunSummary {
  history: Record<string, number>[]; test: Record<string, number> | null;
  best: { path: string; score: number | null; monitor: string } | null;
  params: (Omit<TrainParams, 'name' | 'datasets'> & { sources: string[] }) | null;
  stage: string | null; log: string[]; data_cached: boolean | null;
}
export interface DatasetRef { id: string; name: string; kind: 'h5' | 'pkl' }
export interface DatasetRow {
  id: number; family: string; n_nodes: number; n_edges: number; n_tets: number; betti1: number;
  deformed: boolean | null; f_GHz: number[];
}
export interface DatasetStats {
  total: number; n_modes: number;
  families: Record<string, { count: number; f0: number[]; n_edges: number[] }>;
}
export interface ModeRow {
  mode: number; f_GHz: number; degenerate: boolean; eta: number | null; accelerating: boolean;
  Q0: number | null; G_ohm: number | null; R_over_Q_ohm: number | null; R_sh_ohm: number | null;
  T_transit: number | null; Epk_Eacc: number | null; Bpk_Eacc_mT_per_MVm: number | null;
}
export interface Prediction {
  id: string; geometry_id: string; source: 'model'; n_edges: number; modes: ModeRow[];
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
  featureView: async (gid: string, kind: string, channel: string | null, axis: string, pos: number | null,
                      res = 121): Promise<FeatureView> => {
    const q = new URLSearchParams({ kind, axis, res: String(res) });
    if (channel) q.set('channel', channel);
    if (pos !== null) q.set('pos', String(pos));
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const j: any = await call(`/api/geometries/${gid}/feature_view?${q}`);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const mesh = (m: any) => (m ? { points: f32(m.points_b64), triangles: u32(m.triangles_b64), scalars: f32(m.scalars_b64) } : null);
    return { ...j, plane: mesh(j.plane), cells: mesh(j.cells),
             // eslint-disable-next-line @typescript-eslint/no-explicit-any
             segments: j.segments.map((s: any) => ({ role: s.role, points: f32(s.points_b64) })) };
  },
  jobsConfig: () => call<JobsConfig>('/api/jobs/config'),
  generate: (p: GenParams) => call<Job[]>('/api/jobs/generate', json(p)),
  jobs: () => call<Job[]>('/api/jobs'),
  cancelJob: (id: string) => call<Job>(`/api/jobs/${id}/cancel`, { method: 'POST' }),
  trainConfig: () => call<TrainConfig>('/api/train/config'),
  trainRuns: () => call<RunSummary[]>('/api/train/runs'),
  trainRun: (name: string) => call<RunDetail>(`/api/train/runs/${encodeURIComponent(name)}`),
  trainStart: (p: TrainParams) => call<RunDetail>('/api/train/start', json(p)),
  trainResume: (name: string, epochs: number | null) =>
    call<RunDetail>(`/api/train/runs/${encodeURIComponent(name)}/resume`, json({ epochs })),
  trainStop: () => call<{ stopped: string | null }>('/api/train/stop', { method: 'POST' }),
  loadModel: (run: string) => call<ModelInfo>('/api/model/load', json({ run })),
  rescan: () => call<DatasetRef[]>('/api/datasets/rescan', { method: 'POST' }),
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
