// Typed client of src/service/api.py. Large arrays arrive as base64 little-endian float32 / uint32.
import { decodeModes, type FieldBundle } from './fieldlab';

export interface ModelInfo {
  n_modes: number; n_params: number; device: string; untrained: boolean; checkpoint: string | null;
  load_error?: string | null; run?: string | null; model_type?: string;
}
export interface Info {
  version: string; model: ModelInfo; families: { train: string[]; ood: string[] };
  limits: { max_upload_mb: number; max_tets: number; max_predict_tets?: number; cad: string[]; mesh: string[] };
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
  qoi_source?: QoiSource;
}
// where the figures of merit come from: the predicted p2 fields, the stored p3 labels, the FE display fields
export type QoiSource = 'p2_model' | 'p3_labels' | 'fe';
export interface Geometry {
  id: string; source: Record<string, unknown>; check: Check; family?: string;
  params?: Record<string, number>; surface: Surface;
  truth?: Solution | null;
  can_predict: boolean; predict_blocker: string | null;    // the field3d model needs a curved mesh of a CAD solid
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
  enabled: boolean; gen_root: string | null; cpu_count: number; mem_gb: number; max_total: number;
  families: { train: string[]; ood: string[] }; blocks: Record<string, number | null>; tag_example: string;
  field: { defaults: FieldGenParams; plan: FieldPlan; core_s: number; adapt_factor: number; no_cad: string[] };
}
// field labels (docs/29–30): the generator's --labels field options
export interface FieldGenParams {
  threads: number | null; label_order: number; model_order: number; min_fillet: number; maxh_factor: number;
  field_max_elements: number; adapt: boolean; tol_f: number; tol_q: number; model_max_elements: number;
  max_ndof: number | null; round_edges: boolean; heal: boolean; sample_timeout: number;
}
export interface GenParams extends FieldGenParams {
  families: string[]; labels: 'field'; n_total: number; mesh_size: number; n_modes: number;
  sampling: 'sobol' | 'random'; deform_prob: number; deform_max: number; seed: number; shard_size: number;
  workers: number | null; tag: string | null;
}
// workers × NGSolve threads for field labels on the server's machine (src/service/jobs.field_plan)
export interface FieldPlan {
  workers: number; threads: number; mem_per_worker_gb: number; mem_total_gb: number; cpus: number; mem_gb: number;
  warnings: ('oversubscribed' | 'memory')[];
}
export interface Job {
  id: string; status: 'queued' | 'running' | 'done' | 'failed' | 'cancelled';
  params: GenParams & { family: string; tag: string; block: number };
  stage: string; progress: { done: number; total: number; ok: number; failed: number };
  created: number; started: number | null; finished: number | null; error: string | null;
  outputs: string[]; log: string[];
}
// TRUBA page (src/service/truba.py): the form, the one-line command built from it
export interface TrubaForm {
  action: 'all' | 'dataset' | 'train' | 'status' | 'setup'; labels: 'field'; families: string[];
  n_per_family: number | ''; mesh_size: number; n_modes: number; label_order: number; model_order: number;
  min_fillet: number | ''; threads: number; max_elements: number; maxh_factor: number; adapt: boolean;
  tol_f: number; tol_q: number; model_max_elements: number; max_ndof: number; round_edges: boolean; heal: boolean;
  deform_prob: number;
  deform_max: number; sampling: 'sobol' | 'random'; seed: number; tag: string; partition: string; cpus: number;
  time: string; account: string; max_parallel: number; test: boolean; gpu_partition: string; gpus: number;
  gpu_cpus: number; gpu_time: string; exp: string; model: string; epochs: number; batch: number | '';
  lr: number; num_workers: number | ''; cache_dir: string; train_extra: string[]; dry_run: boolean;
  repo: string; branch: string; repo_dir: string;
}
export interface TrubaConfig {
  repo: string; branch: string; families: { train: string[]; ood: string[] }; models: string[];
  actions: string[]; labels: string[]; form: TrubaForm;
}
export interface TrubaResult {
  command: string; run_args: string; tag: string; families: string[]; group: string; notes: string[];
  estimate: { geometries: number; core_hours: number; disk_gb: number; workers_per_job: number };
}
export interface Preset { embed_dim: number; n_heads: number; n_layers: number; n_basis: number; params_m: number }
export interface TrainConfig {
  enabled: boolean; runs_root: string | null; gpu: string | null; cpu_count: number;
  presets: Record<string, Preset>; busy: boolean; active: string | null; model: ModelInfo;
  model_type: string; config: string; dataset_kinds: string[]; defaults: Partial<Omit<TrainParams, 'name' | 'datasets'>>;
}
export interface TrainParams {
  name: string; datasets: string[]; preset: string; epochs: number; batch_size: number; lr: number;
  n_modes: number; patience: number; num_workers: number;
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
  stage: string | null; log: string[];
}
// kind 'field': field-label H5 (predictable, trainable); 'h5': N0-label H5 (opened for viewing only)
export interface DatasetRef { id: string; name: string; kind: 'h5' | 'field' }
// field-label datasets: label quality per geometry (src/service/datasets.FieldH5Dataset)
export interface FieldRowInfo {
  n_tets_label: number | null; n_dof_label: number | null; n_tets_start: number | null;
  adapt_iters: number | null; adapt_stop: 'converged' | 'budget' | 'max_iter' | null;
  est_rel_f: number | null; est_rel_wall: number | null; adapt_df: number | null; adapt_dq: number | null;
  rounded_edges: number | null; sharp_edges: number; proj_err_max: number | null; t_labels_s: number | null;
}
export interface DatasetRow {
  id: number; family: string; n_nodes: number; n_edges: number; n_tets: number; betti1: number;
  deformed: boolean | null; f_GHz: number[]; field?: FieldRowInfo;
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
  axis_length_mm: number; warnings: string[]; model: ModelInfo;
  time_s: Record<string, number>;            // mesh, operators, model, qoi, display, total [s]
  qoi_source: QoiSource; n_dof: number; n_tets: number;     // p2 DOFs and curved tets of the model's mesh
  comparison?: { rows: CmpRow[]; diff_id: string; truth_id: string };
}
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

function u8(b64: string): Uint8Array {
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
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
  jobsPlan: (p: { max_ndof?: number | null; threads?: number | null; workers?: number | null }) => {
    const q = new URLSearchParams();
    Object.entries(p).forEach(([k, v]) => v !== undefined && v !== null && q.set(k, String(v)));
    return call<FieldPlan>(`/api/jobs/plan?${q}`);
  },
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
  // every mode of a solution + the volume mesh, once per solution (the workspace computes cuts, the 3D grid,
  // modes and the phase from it)
  fields: async (pid: string): Promise<FieldBundle> => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const j: any = await call(`/api/predictions/${pid}/fields`);
    const K = j.n_modes, nv = j.n_vertices;
    return { K, nv, nodes: f32(j.nodes_b64), tets: u32(j.tets_b64), vid: u32(j.vid_b64),
             E: decodeModes(new Uint16Array(u8(j.E_b64).buffer), K, nv, j.E_max),
             H: decodeModes(new Uint16Array(u8(j.H_b64).buffer), K, nv, j.H_max),
             Emax: j.E_max, Hmax: j.H_max, f: j.f_GHz, axis: { z: j.axis.z_mm, Ez: j.axis.Ez } };
  },
  exportUrl: (pid: string) => `/api/predictions/${pid}/export`,
  trubaConfig: () => call<TrubaConfig>('/api/truba/config'),
  trubaCommand: (f: TrubaForm) => call<TrubaResult>('/api/truba/command', json(f)),
};
