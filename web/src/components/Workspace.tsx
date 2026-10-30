// The workspace page: geometry panel (left), the 3D viewer as the main canvas with a floating display toolbar,
// the inspector (right: modes, selected-mode values, comparison), the axis plot dock and the status bar.
// The field engine runs in the browser (fieldlab.ts): no interaction waits for the network.
// One or two panes (split view) show a (source, mode) each from the data already downloaded.
import { useCallback, useDeferredValue, useEffect, useImperativeHandle, useMemo, useReducer, useRef, useState, type Ref } from 'react';
import { api, type FeatureView, type Features, type Geometry, type Info, type Prediction } from '../api';
import { shareMesh, type FieldBundle, type Slice } from '../fieldlab';
import { cssGradient, divergingStops, sequentialStops, type Theme } from '../colors';
import { t, type Key, type Lang } from '../i18n';
import { fmtTimings, int, sci } from '../format';
import { usePersisted } from '../store';
import { useOverlay } from '../overlay';
import { composePng, modesCsv, safeName, saveBlob, type PanePng } from '../exportFiles';
import { AXES, type Axis, type Display, type FeatureKind, type Source, type View3D } from '../workspace';
import GeometryPanel, { type GeomForm } from './GeometryPanel';
import GeometryInfo from './GeometryInfo';
import ModeTable from './ModeTable';
import ModeDetails from './ModeDetails';
import AxisPlot from './AxisPlot';
import ColorBar from './ColorBar';
import DisplayToolbar from './DisplayToolbar';
import Icon, { type IconName } from './Icon';
import { Collapsible, IconButton, Menu, Seg, useMedia } from './ui';
import Viewer3D, { PRESETS, phaseFactor, volumeStops, type Ray, type SurfaceMode, type ViewerHandle } from './Viewer3D';
import { useFieldView } from './useFieldView';
import { useToast } from './Toasts';

export interface WorkspaceHandle {
  openDataset: (did: string, sid: number) => Promise<boolean>;
  closeGeometry: () => boolean;             // false when nothing is open
  key: (e: KeyboardEvent) => void;          // workspace keyboard shortcuts (the app filters typing and modifiers)
}

/** A geometry that can be opened again with one click (uploads cannot: the file is not kept). */
interface Recent {
  key: string; label: string; meta: string; at: number;
  dataset?: { did: string; sid: number };
  sample?: { family: string; id: number; meshSize: number };
}
const RECENT_MAX = 8;
const TOOLBAR_INSET = 76;                    // px of the canvas the floating toolbar covers (two rows)

interface Props {
  ref?: Ref<WorkspaceHandle>;
  lang: Lang; theme: Theme; info: Info | null; hidden: boolean; modelKey: string;
  onWarnings: (w: string[]) => void;
}

/** One line for a toast / status bar: "elliptical #12" or the file name. */
function describe(g: Geometry): string {
  const s = g.source as Record<string, unknown>;
  if (s.kind === 'upload') return String(s.name ?? 'upload');
  return `${s.family ?? g.family ?? '?'}${s.id !== undefined ? ` #${s.id}` : ''}`;
}

type Sheet = 'geom' | 'controls' | 'modes' | null;

/** The colour scale: automatic per mode, locked to the value at lock time, or a typed maximum; linear or log. */
interface Scale {
  lock: boolean;
  locked: { E: number | null; H: number | null };
  manual: { E: string; H: string };           // typed maxima, '' = automatic
  log: boolean;
  decades: 2 | 3;
}
const SCALE0: Scale = { lock: false, locked: { E: null, H: null }, manual: { E: '', H: '' }, log: false, decades: 3 };

/** The probe: a point on the cut plane, kept as its in-plane coordinates (it follows the cut position). */
interface Probe { axis: number; u: number; v: number }

/** |F| at (u, v) of a cut: barycentric interpolation inside the triangle that contains the point (null: outside). */
function probeValue(cut: Slice, amp: Float32Array, axis: number, u: number, v: number): number | null {
  const [cu, cv] = [0, 1, 2].filter((c) => c !== axis);
  const P = cut.points, T = cut.triangles;
  const e = 1e-9;
  for (let k = 0; k < T.length; k += 3) {
    const a = T[k], b = T[k + 1], c = T[k + 2];
    const ax = P[3 * a + cu], ay = P[3 * a + cv];
    const bx = P[3 * b + cu] - ax, by = P[3 * b + cv] - ay, cx = P[3 * c + cu] - ax, cy = P[3 * c + cv] - ay;
    const det = bx * cy - cx * by;
    if (Math.abs(det) < 1e-18) continue;
    const dx = u - ax, dy = v - ay;
    const l1 = (dx * cy - cx * dy) / det, l2 = (bx * dy - dx * by) / det, l0 = 1 - l1 - l2;
    if (l0 < -e || l1 < -e || l2 < -e) continue;
    return l0 * amp[a] + l1 * amp[b] + l2 * amp[c];
  }
  return null;
}

export default function Workspace({ ref, lang, theme, info, hidden, modelKey, onWarnings }: Props) {
  const toast = useToast();
  const compact = useMedia('(max-width: 860px)');
  const [geom, setGeom] = useState<Geometry | null>(null);
  const [pred, setPred] = useState<Prediction | null>(null);
  const [source, setSource] = useState<Source>('model');
  const [mode, setMode] = useState(0);
  const [features, setFeatures] = useState<Features | null>(null);
  const [display, setDisplay] = useState<Display>('field');
  const [fKind, setFKind] = useState<FeatureKind>('wall');
  const [feature, setFeature] = useState('torsion');                 // channel of the 'raw' view
  const [fView, setFView] = useState<FeatureView | null>(null);
  const [fSurf, setFSurf] = useState<Float32Array | null>(null);
  const [busy, setBusy] = useState(false);
  const [predicting, setPredicting] = useState<number | null>(null);   // start time of a running prediction

  const [field, setField] = useState<'E' | 'H'>('E');
  const [view3d, setView3d] = useState<View3D>('cut');
  const [arrowsOn, setArrowsOn] = useState(false);
  const [axis, setAxis] = useState<Axis>('y');
  const [pos, setPos] = useState<number | null>(null);
  const [phase, setPhase] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [surfaceMode, setSurfaceMode] = useState<SurfaceMode>('ghost');
  const fields = useRef(new Map<string, FieldBundle | Promise<void>>());   // solution id → all its modes
  const [, fieldsLoaded] = useReducer((x: number) => x + 1, 0);

  // colour scale, probe, split view
  const [sc, setSc] = useState<Scale>(SCALE0);
  const [probeOn, setProbeOn] = useState(false);
  const [probe, setProbe] = useState<Probe | null>(null);
  const [split, setSplit] = useState(false);
  const [link, setLink] = useState(true);
  const [source2, setSource2] = useState<Source>('model');
  const [mode2, setMode2] = useState(0);

  const [leftOpen, setLeftOpen] = usePersisted('ws.left', true);
  const [rightOpen, setRightOpen] = usePersisted('ws.right', true);
  const [dockOpen, setDockOpen] = usePersisted('ws.dock', true);
  const [sheet, setSheet] = useState<Sheet>(null);
  useOverlay(sheet !== null, () => setSheet(null));
  const [form, setForm] = useState<GeomForm>({ family: 'elliptical', sid: null, unit: 'mm', meshSize: 0.1, nCells: 1 });
  const viewer = useRef<ViewerHandle>(null);
  const viewer2 = useRef<ViewerHandle>(null);
  const [recent, setRecent] = usePersisted<Recent[]>('recent', []);
  const [dropDepth, setDropDepth] = useState(0);

  const run = useCallback(async <T,>(fn: () => Promise<T>, fail: string): Promise<T | undefined> => {
    setBusy(true);
    try { return await fn(); } catch (e) { toast('error', `${fail}: ${(e as Error).message}`); return undefined; } finally { setBusy(false); }
  }, [toast]);

  const truth = geom?.truth ?? null;
  const geomId = useRef<string | null>(null);              // the open geometry: late answers for another one are dropped
  const resetPerGeometry = () => {
    setPos(null); setProbe(null);
    setSc((s) => ({ ...SCALE0, log: s.log, decades: s.decades }));
    fields.current.clear();
    setFeatures(null); setFView(null); setFSurf(null);
  };
  const remember = (r: Recent) => setRecent((l) => [r, ...l.filter((x) => x.key !== r.key)].slice(0, RECENT_MAX));
  const newGeometry = (g: Geometry | undefined, mem?: Recent): boolean => {
    if (!g) return false;
    if (mem) remember(mem);
    geomId.current = g.id;
    setGeom(g); setPred(null);
    resetPerGeometry();
    const acc = g.truth?.modes.find((m) => m.accelerating);
    setMode(acc ? acc.mode : 0);
    setSource(g.truth ? 'fe' : 'model');
    api.features(g.id).then((f) => { if (geomId.current === g.id) setFeatures(f); }).catch(() => setFeatures(null));
    toast('ok', `${t(lang, 'toastGeomOk')}: ${describe(g)} · ${int(g.check.n_tets)} ${t(lang, 'tets')}`);
    return true;
  };

  // unload the geometry: viewer, field data, prediction and the per-geometry state, back to the empty workspace.
  // Frontend only; the server's store expires its items (RFCAV_TTL_S)
  const closeGeometry = () => {
    geomId.current = null;
    setGeom(null); setPred(null); setPlaying(false); setSplit(false); setProbeOn(false);
    resetPerGeometry();
    setMode(0); setSource('model');
  };

  // the field3d model on the curved mesh: seconds to ~30 s on a CPU; the panel shows a spinner and the elapsed time
  const predict = () => geom && geom.can_predict && run(async () => {
    setPredicting(performance.now());
    try {
      const p = await api.predict(geom.id, truth?.id ?? null);
      if (geomId.current !== geom.id) return;               // closed (or replaced) meanwhile
      setPred(p);
      setSource('model');
      if (!truth) {
        const acc = p.modes.find((m) => m.accelerating);
        setMode(acc ? acc.mode : 0);
      }
      toast('ok', `${t(lang, 'toastPredOk')} · ${fmtTimings(lang, p.time_s)}`);
    } finally { setPredicting(null); }
  }, t(lang, 'toastPredFail'));

  const sampleRecent = (family: string, id: number, meshSize: number): Recent => ({
    key: `s:${family}:${id}:${meshSize}`, at: Date.now(), label: `${family} #${id}`,
    meta: `${t(lang, 'meshSize').split(' ')[0]} ${meshSize}`, sample: { family, id, meshSize },
  });
  const buildSample = (family: string, id: number | null, meshSize: number) =>
    run(() => api.sample(family, id, meshSize), t(lang, 'toastGeomFail')).then((g) => {
      const sid = g ? Number((g.source as Record<string, unknown>).id) : NaN;      // the drawn id when none was given
      return newGeometry(g, Number.isFinite(sid) ? sampleRecent(family, sid, meshSize) : undefined);
    });
  const upload = (file: File) =>
    run(() => api.upload(file, form.unit, form.meshSize, form.nCells), t(lang, 'toastGeomFail')).then((g) => newGeometry(g));
  const datasetRecent = (g: Geometry, did: string, sid: number): Recent => {
    const s = g.source as Record<string, unknown>;
    return { key: `d:${did}:${sid}`, at: Date.now(), label: `${s.family ?? g.family ?? '?'} #${sid}`,
             meta: String(s.dataset ?? ''), dataset: { did, sid } };
  };
  const openDataset = (did: string, sid: number) =>
    run(() => api.datasetOpen(did, sid), t(lang, 'toastGeomFail')).then((g) => newGeometry(g, g ? datasetRecent(g, did, sid) : undefined));
  const reopen = (r: Recent) => {
    const job = r.dataset ? api.datasetOpen(r.dataset.did, r.dataset.sid) : api.sample(r.sample!.family, r.sample!.id, r.sample!.meshSize);
    run(() => job, t(lang, 'toastGeomFail')).then((g) => {
      if (g) newGeometry(g, r);
      else setRecent((l) => l.filter((x) => x.key !== r.key));          // gone (another dataset root, a restart)
    });
  };

  // a CAD / mesh file dropped on the workspace goes the way of the upload form (same unit, mesh size, cells)
  const accepted = info ? [...info.limits.cad, ...info.limits.mesh] : [];
  const hasFiles = (e: React.DragEvent) => Array.from(e.dataTransfer?.types ?? []).includes('Files');
  const onDragEnter = (e: React.DragEvent) => { if (hasFiles(e)) { e.preventDefault(); setDropDepth((d) => d + 1); } };
  const onDragOver = (e: React.DragEvent) => { if (hasFiles(e)) { e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; } };
  const onDragLeave = (e: React.DragEvent) => { if (hasFiles(e)) setDropDepth((d) => Math.max(0, d - 1)); };
  const onDrop = (e: React.DragEvent) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    setDropDepth(0);
    const f = e.dataTransfer.files[0];
    if (!f || busy) return;
    const ext = `.${f.name.split('.').pop()?.toLowerCase() ?? ''}`;
    if (accepted.length && !accepted.includes(ext)) { toast('error', `${t(lang, 'dropBadType')}: ${ext} (${accepted.join(' ')})`); return; }
    upload(f);
  };

  const closeIfOpen = () => { if (!geom) return false; closeGeometry(); return true; };

  useImperativeHandle(ref, () => ({ openDataset, closeGeometry: closeIfOpen, key: onKey }));

  const solId = (s: Source) => (s === 'fe' ? truth?.id : s === 'diff' ? pred?.comparison?.diff_id : pred?.id);
  const srcId = solId(source);
  const sources: Source[] = [...(truth ? ['fe' as Source] : []), ...(pred ? ['model' as Source] : []),
                             ...(pred?.comparison ? ['diff' as Source] : [])];

  // every mode of a solution, fetched once (GET …/fields); the shown one first, then the other sources of
  // the geometry, so switching the source does not wait either
  const bundleOf = (id: string | null | undefined): FieldBundle | null => {
    const b = id ? fields.current.get(id) : undefined;
    return b && !(b instanceof Promise) ? b : null;
  };
  const want = useCallback((id: string | null | undefined) => {
    const cache = fields.current;
    if (!id || cache.has(id)) return;
    const job: Promise<void> = api.fields(id).then((b) => {
      if (cache.get(id) !== job) return;                  // a new geometry meanwhile
      cache.set(id, shareMesh(b, [...cache.values()].filter((x): x is FieldBundle => !(x instanceof Promise))));
      fieldsLoaded();
    }).catch((e) => { cache.delete(id); toast('error', e.message); });
    cache.set(id, job);
  }, [toast]);
  const diffId = pred?.comparison?.diff_id;
  const bundle = bundleOf(srcId);
  const refBundle = source === 'diff' ? bundleOf(truth?.id) : bundle;    // the colour scale
  useEffect(() => {                                       // drop the solutions no longer shown
    const keep = new Set([truth?.id, pred?.id, diffId]);
    for (const id of [...fields.current.keys()]) if (!keep.has(id)) fields.current.delete(id);
  }, [truth?.id, pred?.id, diffId]);
  useEffect(() => {
    if (display !== 'field') return;
    want(srcId);
    if (source === 'diff') want(truth?.id);
    if (bundle) [truth?.id, pred?.id, diffId].forEach(want);
  }, [display, srcId, source, truth?.id, pred?.id, diffId, bundle, want]);
  // model input features
  useEffect(() => {
    if (!geom || display !== 'feature') return;
    let live = true;
    const h = setTimeout(() => {
      api.featureView(geom.id, fKind, fKind === 'raw' ? feature : null, axis, pos)
        .then((v) => live && setFView(v)).catch((e) => toast('error', e.message));
    }, 120);
    if (fKind === 'raw') api.featureSurface(geom.id, feature).then((v) => live && setFSurf(v)).catch(() => undefined);
    else setFSurf(null);
    return () => { live = false; clearTimeout(h); };
  }, [geom, fKind, feature, axis, pos, display, toast]);

  // phase animation 0 → 180° (loop)
  const raf = useRef<number | null>(null);
  useEffect(() => {
    if (!playing) return;
    let last = performance.now();
    const step = (now: number) => {
      if (now - last > 50) { setPhase((p) => (p + 5) % 185); last = now; }
      raf.current = requestAnimationFrame(step);
    };
    raf.current = requestAnimationFrame(step);
    return () => { if (raf.current) cancelAnimationFrame(raf.current); };
  }, [playing]);

  const bbox = geom?.check.bbox_mm;
  const ai = AXES.indexOf(axis);
  const range = bbox ? [bbox[0][ai], bbox[1][ai]] : [0, 1];
  const posValue = pos ?? (range[0] + range[1]) / 2;
  const cutPos = useDeferredValue(posValue);          // the slider moves at once, the cut follows

  // split view: the second pane's (source, mode)
  const splitOn = split && display === 'field' && !!geom;
  const srcId2 = solId(source2);
  const bundle2 = splitOn ? bundleOf(srcId2) : null;
  const refBundle2 = source2 === 'diff' ? bundleOf(truth?.id) : bundle2;
  useEffect(() => {
    if (!splitOn) return;
    want(srcId2);
    if (source2 === 'diff') want(truth?.id);
  }, [splitOn, srcId2, source2, truth?.id, want]);
  useEffect(() => {                                   // keep the second pane on an available source
    if (splitOn && !sources.includes(source2) && sources.length) setSource2(sources[sources.length - 1]);
  }, [splitOn, sources, source2]);

  // colour scale: a typed maximum, else the locked one, else the mode's own
  const typed = (f: 'E' | 'H') => { const v = Number(sc.manual[f].replace(',', '.')); return sc.manual[f] !== '' && Number.isFinite(v) && v > 0 ? v : null; };
  const override = (f: 'E' | 'H') => typed(f) ?? (sc.lock ? sc.locked[f] : null);
  const logD = sc.log ? sc.decades : 0;

  // the shown field of each pane: a cut / the grid per mesh, the vectors per (solution, mode, E/H), and per
  // phase step only the arrows and the gain of the colour maps
  const A = useFieldView({ display, bundle, refBundle, mode, field, view3d, ai: ai as 0 | 1 | 2, cutPos, arrowsOn, phase, source,
                           maxOverride: override(field), logDecades: logD, fView, fSurf });
  const B = useFieldView({ display: 'field', bundle: bundle2, refBundle: refBundle2, mode: mode2, field, view3d, ai: ai as 0 | 1 | 2,
                           cutPos, arrowsOn, phase, source: source2, maxOverride: override(field), logDecades: logD });
  const view = A.view, view2 = B.view;
  useEffect(() => {                                   // lock: remember the current maximum of this quantity
    if (sc.lock && sc.locked[field] === null && display === 'field' && A.onField) setSc((s) => ({ ...s, locked: { ...s.locked, [field]: A.lim } }));
  }, [sc.lock, sc.locked, field, display, A.onField, A.lim]);

  const gradient = legendGradient(view.signed, !!view.volume, theme);
  const shown = source === 'fe' ? truth : pred;      // model + diff rows come from the model solution
  const shown2 = source2 === 'fe' ? truth : pred;
  const cmp = pred?.comparison?.rows ?? null;
  const row = shown?.modes[mode];
  const nModes = shown?.modes.length ?? 0;
  useEffect(() => { if (nModes && mode2 >= (shown2?.modes.length ?? 0)) setMode2(0); }, [nModes, mode2, shown2]);

  // another model was loaded (Training page): the shown prediction is stale
  const firstKey = useRef(modelKey);
  useEffect(() => {
    if (firstKey.current === modelKey) return;
    firstKey.current = modelKey;
    setPred(null);
    setSource(truth ? 'fe' : 'model');
  }, [modelKey, truth]);

  // warnings of the prediction go to the top bar
  useEffect(() => { onWarnings(pred?.warnings.filter((w) => !w.startsWith('untrained')) ?? []); }, [pred, onWarnings]);

  // ── probe: a click on the cut plane ─────────────────────────────────────────
  const canProbe = display === 'field' && view3d === 'cut' && !!A.cut;
  const probeOnNow = probeOn && canProbe;
  const onPick = (ray: Ray) => {
    const cut = A.cut;
    if (!cut || Math.abs(ray.dir[ai]) < 1e-12) return;
    const tt = (cut.pos - ray.origin[ai]) / ray.dir[ai];
    if (tt <= 0) return;
    const [cu, cv] = [0, 1, 2].filter((c) => c !== ai);
    setProbe({ axis: ai, u: ray.origin[cu] + tt * ray.dir[cu], v: ray.origin[cv] + tt * ray.dir[cv] });
  };
  const probeAt = probeOnNow && probe && probe.axis === ai && A.cut ? probe : null;
  const worldPt = (cut: Slice): [number, number, number] => {
    const p: [number, number, number] = [0, 0, 0];
    const [cu, cv] = [0, 1, 2].filter((c) => c !== ai);
    p[ai] = cut.pos; p[cu] = probeAt!.u; p[cv] = probeAt!.v;
    return p;
  };
  const marker = probeAt && A.cut ? worldPt(A.cut) : null;
  const probeText = (V: typeof A, who: string): string | null => {
    if (!probeAt || !V.cut || !V.planeAmp) return null;
    const amp = probeValue(V.cut, V.planeAmp, ai, probeAt.u, probeAt.v);
    const p = worldPt(V.cut);
    const q = `|${source === 'diff' && who === 'A' ? 'Δ' : ''}${field}|`;
    const pos3 = `(${p.map((x) => x.toFixed(1)).join(', ')}) mm`;
    return amp === null ? `${t(lang, 'probeOutside')} · ${pos3}` : `${q} = ${sci(amp * Math.abs(V.a))} ${field === 'E' ? 'V/m' : 'A/m'} · ${pos3}`;
  };
  const probeA = probeText(A, 'A');
  const probeB = splitOn ? probeText(B, 'B') : null;

  const presetBtn = (id: string) => { viewer.current?.preset(id); viewer2.current?.preset(id); };
  const homeBtn = () => { viewer.current?.home(); viewer2.current?.home(); };

  // linked cameras: either pane's camera drives the other
  useEffect(() => {
    if (!splitOn || !link || !geom) return;
    const a = viewer.current, b = viewer2.current;
    const ca = a?.camera(), cb = b?.camera();
    if (!a || !b || !ca || !cb) return;
    let busy_ = false;
    const follow = (from: any, to: ViewerHandle) => from.onModified(() => {      // eslint-disable-line @typescript-eslint/no-explicit-any
      if (busy_) return;
      busy_ = true; try { to.syncFrom(from); } finally { busy_ = false; }
    });
    b.syncFrom(ca);
    const s1 = follow(ca, b), s2 = follow(cb, a);
    return () => { s1.unsubscribe(); s2.unsubscribe(); };
  }, [splitOn, link, geom]);

  // ── exports ────────────────────────────────────────────────────────────────
  const stopsOf = (V: typeof view): [number, string][] => {
    const st = V.signed ? divergingStops(theme) : V.volume ? volumeStops(theme) : sequentialStops(theme);
    return st.map(([p, c]) => [V.signed ? (p + 1) / 2 : p, c]);
  };
  const paneTitle = (V: typeof view, src: Source, m: number, b: FieldBundle | null) => (display === 'feature' ? V.label
    : `${V.label} · ${t(lang, 'statusMode')} ${m}${b ? ` · ${b.f[m]?.toFixed(4)} GHz` : ''} · ${t(lang, src === 'fe' ? 'fe' : src === 'model' ? 'model' : 'diff')}`);
  const scaleTag = (V: typeof view) => (V.log ? `log10 · ${V.log} ${t(lang, 'decades')}` : t(lang, 'scaleLinear'));
  const exportPng = async () => {
    const handles = [viewer.current, ...(splitOn ? [viewer2.current] : [])];
    try {
      const imgs = await Promise.all(handles.map((h) => h!.capture()));
      const infos = [{ V: view, src: source, m: mode, b: bundle, note: probeA }, ...(splitOn ? [{ V: view2, src: source2, m: mode2, b: bundle2, note: probeB }] : [])];
      const panes: PanePng[] = infos.map((x, i) => ({
        image: imgs[i], title: `${splitOn ? (i ? 'B' : 'A') + ' · ' : ''}${paneTitle(x.V, x.src, x.m, x.b)}`, stops: stopsOf(x.V),
        range: x.V.phys, unit: x.V.unit, scaleTag: scaleTag(x.V), log: x.V.log, neutral: x.V.neutral, note: x.note ?? undefined,
      }));
      const cs = getComputedStyle(document.documentElement);
      const cssW = (document.querySelector('.pane .viewer') as HTMLElement | null)?.clientWidth ?? 1;
      const legendBits = [t(lang, 'title'), geom ? describe(geom) : '', display === 'field' ? `${field}  φ = ${phase}°` : '',
                          display === 'field' && view3d === 'cut' ? `${t(lang, 'cutShort')} ⟂ ${axis} = ${cutPos.toFixed(1)} mm` : '',
                          display === 'field' && view3d === 'volume' ? t(lang, 'showVolume') : '', 'U = 1 J'];
      const blob = await composePng(panes, legendBits.filter(Boolean).join('  ·  '), compact && splitOn, {
        bg: cs.getPropertyValue('--surface-1').trim() || '#ffffff', fg: cs.getPropertyValue('--text-primary').trim() || '#000000',
        muted: cs.getPropertyValue('--muted').trim() || '#666666', line: cs.getPropertyValue('--border-strong').trim() || '#999999' }, cssW);
      saveBlob(blob, `${safeName(`rfcav_${geom ? describe(geom) : 'view'}_${display === 'field' ? field + mode : fKind}_${source}`)}.png`);
      toast('ok', t(lang, 'toastPngOk'));
    } catch (e) { toast('error', `${t(lang, 'toastExportFail')}: ${(e as Error).message}`); }
  };
  const exportCsv = () => {
    if (!shown) return;
    saveBlob(new Blob([modesCsv(shown.modes, cmp && source !== 'fe' ? cmp : null)], { type: 'text/csv;charset=utf-8' }),
             `${safeName(`rfcav_${geom ? describe(geom) : 'modes'}_${source}`)}_modes.csv`);
    toast('ok', t(lang, 'toastCsvOk'));
  };

  // ── keyboard shortcuts (App filters typing in inputs and modified keys, and calls this on the workspace page) ──
  const toggleSplit = () => {
    if (display !== 'field') return;
    if (!split) {                                     // the second pane: another source of the mode, else the next mode
      const other = sources.find((s) => s !== source);
      setSource2(other ?? source);
      setMode2(other ? mode : Math.min(mode + 1, Math.max(0, nModes - 1)));
    }
    setSplit(!split);
  };
  const toggleLock = () => setSc((s) => ({ ...s, lock: !s.lock, locked: { E: null, H: null } }));
  const onKey = (e: KeyboardEvent) => {
    if (!geom) return;
    const k = e.key.length === 1 ? e.key.toLowerCase() : e.key;
    const field_ = display === 'field';
    const step = (range[1] - range[0]) / 40;
    let used = true;
    if (/^[1-7]$/.test(k)) presetBtn(PRESETS[Number(k) - 1].id);
    else if (k === 'h') homeBtn();
    else if (k === 'ArrowLeft' && field_ && nModes) setMode((m) => Math.max(0, m - 1));
    else if (k === 'ArrowRight' && field_ && nModes) setMode((m) => Math.min(nModes - 1, m + 1));
    else if (k === 'e' && field_) setField('E');
    else if (k === 'm' && field_) setField('H');
    else if (k === 's' && field_ && sources.length > 1) setSource(sources[(sources.indexOf(source) + 1) % sources.length]);
    else if (k === 'v' && field_) setView3d(view3d === 'cut' ? 'volume' : 'cut');
    else if (k === 'a' && field_) setArrowsOn(!arrowsOn);
    else if (k === 'x' || k === 'y' || k === 'z') { setAxis(k); setPos(null); setProbe(null); }
    else if ((k === '[' || k === ']') && (!field_ || view3d === 'cut')) {
      setPos(Math.min(range[1], Math.max(range[0], posValue + (k === ']' ? step : -step))));
    }
    else if (k === ' ' && field_ && shown) setPlaying(!playing);
    else if (k === 'p' && field_) setProbeOn(!probeOn);
    else if (k === 'd' && field_) toggleSplit();
    else if (k === 'l' && field_) toggleLock();
    else if (k === 'g' && field_) setSc((s) => ({ ...s, log: !s.log }));
    else used = false;
    if (used) e.preventDefault();
  };

  const scaleActive = sc.lock || sc.log || typed(field) !== null;
  const scaleMenu = (
    <Menu icon="palette" label={t(lang, 'scaleMenu')} testid="scale-menu" active={scaleActive} disabled={display !== 'field'}>
      <div className="menu-form" data-testid="scale-form">
        <label className="check"><input type="checkbox" checked={sc.lock} data-testid="scale-lock" onChange={toggleLock} />
          <span>{t(lang, 'scaleLock')}</span></label>
        <label className="field">
          <span className="field-label">{t(lang, 'scaleMax')} ({field === 'E' ? 'V/m' : 'A/m'})</span>
          <input type="text" inputMode="decimal" value={sc.manual[field]} placeholder={`${t(lang, 'scaleAuto')}: ${sci(A.lim)}`} data-testid="scale-max"
                 aria-invalid={sc.manual[field] !== '' && typed(field) === null}
                 onChange={(e) => setSc((s) => ({ ...s, manual: { ...s.manual, [field]: e.target.value } }))} />
        </label>
        <label className="check"><input type="checkbox" checked={sc.log} data-testid="scale-log"
                                        onChange={(e) => setSc((s) => ({ ...s, log: e.target.checked }))} /><span>{t(lang, 'scaleLog')}</span></label>
        <Seg value={String(sc.decades) as '2' | '3'} label={t(lang, 'scaleDecades')} className="full"
             onChange={(v) => setSc((s) => ({ ...s, decades: Number(v) as 2 | 3 }))}
             options={[{ value: '2', label: `2 ${t(lang, 'decades')}`, disabled: !sc.log }, { value: '3', label: `3 ${t(lang, 'decades')}`, disabled: !sc.log }]} />
        <button type="button" className="btn small" onClick={() => setSc((s) => ({ ...SCALE0, log: s.log, decades: s.decades }))}>{t(lang, 'scaleReset')}</button>
      </div>
    </Menu>
  );
  const tools = (<>
    <span className="tb-sep" aria-hidden="true" />
    {scaleMenu}
    <IconButton icon="probe" label={`${t(lang, canProbe || display !== 'field' ? 'probeTip' : 'probeNeedsCut')} (P)`} active={probeOnNow}
                disabled={!canProbe && !probeOn} onClick={() => setProbeOn(!probeOn)} testid="probe" />
    <IconButton icon="split" label={`${t(lang, 'splitTip')} (D)`} active={splitOn} disabled={display !== 'field'} onClick={toggleSplit} testid="split" />
    {splitOn && <IconButton icon={link ? 'link' : 'unlink'} label={t(lang, link ? 'linkOn' : 'linkOff')} active={link} onClick={() => setLink(!link)} testid="split-link" />}
    <Menu icon="download" label={t(lang, 'exportMenu')} testid="export-menu" align="end">
      {(close) => (
        <div className="menu-list" role="menu">
          <button type="button" role="menuitem" data-testid="export-png" onClick={() => { close(); exportPng(); }}><Icon name="image" size={14} />{t(lang, 'exportPng')}</button>
          <button type="button" role="menuitem" data-testid="export-csv" disabled={!shown} onClick={() => { close(); exportCsv(); }}><Icon name="table" size={14} />{t(lang, 'exportCsv')}</button>
        </div>
      )}
    </Menu>
  </>);

  const toolbar = (
    <DisplayToolbar lang={lang} display={display} setDisplay={setDisplay} sources={sources} source={source}
                    setSource={setSource} field={field} setField={setField} view3d={view3d} setView3d={setView3d}
                    arrowsOn={arrowsOn} setArrowsOn={setArrowsOn} axis={axis}
                    setAxis={(x) => { setAxis(x); setPos(null); setProbe(null); }} posValue={posValue} posRange={range} setPos={setPos}
                    phase={phase} setPhase={setPhase} playing={playing} setPlaying={setPlaying} canPlay={!!shown}
                    surfaceMode={surfaceMode} setSurfaceMode={setSurfaceMode} fKind={fKind} setFKind={setFKind}
                    feature={feature} setFeature={setFeature} featureNames={features?.names ?? []} extra={tools} />
  );

  // the side panels do not depend on the phase / cut position: rebuilt only when their own inputs change
  /* eslint-disable react-hooks/exhaustive-deps */
  const geomPanel = useMemo(() => (
    <GeometryPanel lang={lang} info={info} geometry={geom} busy={busy} predicting={predicting} form={form} setForm={setForm}
                   onUpload={upload} onSample={buildSample} onPredict={predict} onClose={closeGeometry}
                   after={geom ? <GeometryInfo lang={lang} geometry={geom} features={features} /> : null}>
      {recent.length > 0 && (
        <Collapsible title={t(lang, 'recentTitle')} className="panel-section recent"
                     actions={<IconButton icon="x" label={t(lang, 'recentClear')} className="flat" onClick={() => setRecent([])} testid="recent-clear" />}>
          <ul className="recent-list" data-testid="recent">
            {recent.map((r) => (
              <li key={r.key}>
                <button type="button" className="recent-item" disabled={busy} title={`${t(lang, 'recentOpen')}: ${r.label} ${r.meta}`}
                        onClick={() => reopen(r)} data-testid="recent-item">
                  <Icon name={r.dataset ? 'database' : 'cube'} size={14} />
                  <span className="recent-label">{r.label}</span><span className="recent-meta">{r.meta}</span>
                </button>
              </li>
            ))}
          </ul>
        </Collapsible>
      )}
    </GeometryPanel>
  ), [lang, info, geom, busy, predicting, form, features, recent, truth, pred]);

  const sourceName = (s: Source) => t(lang, s === 'fe' ? 'fe' : s === 'model' ? 'model' : 'diff');
  // what the numbers are: Ritz frequencies + figures of merit from the predicted p2 fields, or the stored p3 labels
  const qoiSrc = source === 'fe' ? truth?.qoi_source : pred?.qoi_source;
  const qoiNote = qoiSrc === 'p2_model' ? t(lang, 'qoiP2Model') : qoiSrc === 'p3_labels' ? t(lang, 'qoiP3Labels') : null;
  const inspector = useMemo(() => (<>
    {shown ? (
      <Collapsible title={cmp ? t(lang, 'cmpTitle') : t(lang, 'modes')} className="panel-section"
                   actions={<IconButton icon="table" label={t(lang, 'exportCsv')} className="flat" onClick={exportCsv} testid="csv-btn" />}>
        <ModeTable lang={lang} modes={shown.modes} cmp={cmp} selected={mode} onSelect={setMode} />
        {qoiNote && <div className="note qoi-source" data-testid="qoi-source">{qoiNote}</div>}
      </Collapsible>
    ) : <div className="empty-note">{!geom ? t(lang, 'empty') : predicting !== null ? t(lang, 'predicting')
                                      : t(lang, geom.can_predict ? 'noPrediction' : 'predictBlockedSee')}</div>}
    {shown && row && (
      <Collapsible title={`${t(lang, 'selected')}: ${row.mode} · ${sourceName(source)}`} className="panel-section">
        <ModeDetails lang={lang} row={row}
                     cmp={cmp && source !== 'fe' ? cmp[mode] : null}
                     timings={source === 'fe' ? null : pred?.time_s}
                     size={source !== 'fe' && pred?.n_dof ? `${int(pred.n_dof)} DOF (p2) · ${int(pred.n_tets)} ${t(lang, 'curvedTets')}` : null}
                     exportId={source === 'fe' ? truth?.id : pred?.id}
                     label={source === 'fe' ? truth?.labels?.[mode] ?? null : null} />
      </Collapsible>
    )}
    {source === 'diff' && display === 'field' && <div className="note pad">{t(lang, 'diffNote')}</div>}
    {display === 'feature' && <div className="note pad">{t(lang, `fkn_${fKind}` as Key)}</div>}
    <div className="note disclaimer pad">{t(lang, 'disclaimer')}</div>
  </>), [lang, shown, cmp, mode, row, source, pred, truth, display, fKind, geom, qoiNote, predicting]);
  /* eslint-enable react-hooks/exhaustive-deps */

  const plot = A.onField && bundle ? (
    <AxisPlot z={bundle.axis.z} ez={bundle.axis.Ez[mode]} title={t(lang, 'axisPlot')} note={t(lang, 'axisNote')}
              scale={(bundle && mode < bundle.K ? (refBundle ?? bundle).Emax[mode] : null) ?? 1} empty={t(lang, 'noAxisField')} />
  ) : null;

  const sideTab = (open: boolean, set: (b: boolean) => void, icon: IconName, title: string, side: 'left' | 'right', body: React.ReactNode) => (
    <aside className={`side ${side} ${open ? '' : 'collapsed'}`} aria-label={title}>
      {open ? (<>
        <div className="side-head">
          <Icon name={icon} size={15} /><span className="side-title">{title}</span>
          <IconButton icon={side === 'left' ? 'left' : 'right'} label={t(lang, 'panelHide')} onClick={() => set(false)} className="flat" testid={`hide-${side}`} />
        </div>
        <div className="side-body">{body}</div>
      </>) : (
        <button type="button" className="side-tab" title={`${t(lang, 'panelShow')}: ${title}`} onClick={() => set(true)} data-testid={`show-${side}`}>
          <Icon name={icon} size={16} /><span className="side-tab-label">{title}</span>
        </button>
      )}
    </aside>
  );

  const badges = [...(typed(field) !== null ? [t(lang, 'scaleManual')] : sc.lock ? [t(lang, 'scaleLocked')] : [])];
  const paneBar = splitOn && (
    <div className="pane-bar" data-testid="pane-bar">
      <div className="pane-bar-half">
        <span className="chip tiny accent">A</span>
        <span className="pane-bar-text">{sourceName(source)} · {t(lang, 'statusMode')} {mode}{bundle ? ` · ${bundle.f[mode]?.toFixed(4)} GHz` : ''}</span>
      </div>
      <div className="pane-bar-half">
        <span className="chip tiny accent">B</span>
        {sources.length > 1 && (
          <Seg value={source2} label={t(lang, 'source')} onChange={setSource2}
               options={sources.map((s) => ({ value: s, label: t(lang, s === 'fe' ? 'srcFe' : s === 'model' ? 'model' : 'diffShort'), title: sourceName(s), testid: `src2-${s}` }))} />
        )}
        <select className="tb-select" aria-label={t(lang, 'mode')} value={mode2} data-testid="mode2" onChange={(e) => setMode2(Number(e.target.value))}>
          {(shown2?.modes ?? []).map((m) => <option key={m.mode} value={m.mode}>{t(lang, 'statusMode')} {m.mode} · {m.f_GHz.toFixed(4)} GHz</option>)}
        </select>
      </div>
    </div>
  );

  const pane = (id: 'A' | 'B', vref: Ref<ViewerHandle>, V: typeof view, b: FieldBundle | null, src: Source, m: number, sid: string | null | undefined,
                sh: typeof shown, note: string | null) => (
    <div className="pane" data-pane={id}>
      <Viewer3D ref={vref} surface={geom?.surface ?? null} surfaceScalars={V.surfS} plane={V.plane}
                planeScalars={V.planeS} volume={V.volume} arrows={V.arrows}
                range={V.range} gain={V.gain} signed={V.signed} logScale={V.log > 0}
                surfaceMode={surfaceMode} theme={theme} viewAxis={axis}
                oblique={display === 'field' && view3d === 'volume'}
                neutralPlane={V.neutral} cells={V.cells} segments={V.segments}
                pickMode={probeOnNow} onPick={onPick} marker={marker} topInset={compact ? 0 : TOOLBAR_INSET} />
      {id === 'A' && !geom && <div className="viewer-empty"><Icon name="cube" size={34} /><span>{t(lang, 'empty')}</span><span className="sub">{t(lang, 'dropHint')}</span></div>}
      {geom && !sh && display === 'field' && (
        <div className="viewer-empty">
          {predicting !== null ? <><span className="spinner lg" aria-hidden="true" /><span>{t(lang, 'predicting')}</span></>
            : <span>{t(lang, geom.can_predict ? 'noPrediction' : 'predictBlockedSee')}</span>}
        </div>
      )}
      {sh && display === 'field' && sid && !b && <div className="viewer-empty"><span>{t(lang, 'loadingFields')}</span></div>}
      {(sh || display === 'feature') && geom && (
        <ColorBar gradient={gradient} range={V.phys} unit={V.unit} neutral={V.neutral} scaleTag={scaleTag(V)} badges={badges}
                  title={`${splitOn ? `${id} · ` : ''}${paneTitle(V, src, m, b)}`}
                  probe={note} onClearProbe={() => setProbe(null)} clearLabel={t(lang, 'probeClear')} />
      )}
    </div>
  );

  const stage = (
    <div className={`stage ${splitOn ? 'split' : ''}`} data-testid="stage">
      {paneBar}
      <div className="canvas-area">
        <div className="panes">
          {pane('A', viewer, view, bundle, source, mode, srcId, shown, probeA)}
          {splitOn && pane('B', viewer2, view2, bundle2, source2, mode2, srcId2, shown2, probeB)}
        </div>
        {geom && !compact && <div className="ov-top">{toolbar}</div>}
        {geom && (
          <div className="viewstrip" role="toolbar" aria-label={t(lang, 'viewTools')}>
            <IconButton icon="home" label={`${t(lang, 'viewHome')} (H)`} onClick={homeBtn} testid="view-home" />
            {PRESETS.map((pr, i) => (
              <button key={pr.id} type="button" className="ibtn txt" title={`${t(lang, pr.tip)} (${i + 1})`}
                      aria-label={t(lang, pr.tip)} data-testid={`view-${pr.id}`} onClick={() => presetBtn(pr.id)}>
                {pr.label}
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );

  const statusBar = (
    <footer className="statusbar" aria-label="status">
      {geom ? (<>
        <span>{describe(geom)}</span>
        <span className="num">{int(geom.check.n_tets)} {t(lang, 'statusTets')}</span>
        <span className="num">{int(geom.check.n_nodes)} {t(lang, 'statusNodes')}</span>
        {pred?.n_dof ? <span className="num" title="field3d: HCurl p2">{int(pred.n_dof)} DOF (p2) · {int(pred.n_tets)} {t(lang, 'curvedTets')}</span> : null}
        {shown && row && <span className="num">{t(lang, 'statusMode')} {row.mode} · {row.f_GHz.toFixed(4)} GHz</span>}
        {predicting !== null && <span className="busy-text"><span className="spinner" aria-hidden="true" />{t(lang, 'predicting')}</span>}
        {pred && predicting === null && <span className="num">{t(lang, 'timings')} {fmtTimings(lang, pred.time_s)}</span>}
      </>) : <span>{t(lang, 'statusNone')}</span>}
      <span className="grow" />
      {shown && <span>{t(lang, 'statusSource')}: {sourceName(source)}</span>}
      {info && <span className="num">{info.model.model_type ? `${info.model.model_type} · ` : ''}{info.model.run ? `${info.model.run} · ` : ''}{(info.model.n_params / 1e6).toFixed(2)} M · {info.model.device}{info.model.untrained ? ` · ${t(lang, 'untrainedChip')}` : ''}</span>}
    </footer>
  );

  const sheetTitle: Record<Exclude<Sheet, null>, string> = {
    geom: t(lang, 'sheetGeometry'), controls: t(lang, 'sheetControls'), modes: t(lang, 'sheetModes'),
  };

  return (
    <div className={`ws ${compact ? 'compact' : ''}`} hidden={hidden} data-testid="workspace"
         onDragEnter={onDragEnter} onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onDrop}>
      {dropDepth > 0 && (
        <div className="drop-overlay" data-testid="drop-overlay" aria-hidden="true">
          <Icon name="upload" size={30} />
          <strong>{t(lang, 'dropFile')}</strong>
          <span>{t(lang, 'dropWith')} {form.unit} · {t(lang, 'meshSize').split(' ')[0]} {form.meshSize} · {t(lang, 'nCells')} {form.nCells}</span>
        </div>
      )}
      <div className="ws-main">
        {!compact && sideTab(leftOpen, setLeftOpen, 'cube', t(lang, 'geometry'), 'left', geomPanel)}
        <div className="ws-center">
          {stage}
          {!compact && plot && (
            <section className={`dock ${dockOpen ? 'open' : ''}`}>
              <div className="dock-head">
                <Icon name="field" size={15} /><span className="side-title">{t(lang, 'axisPlot')}</span>
                <IconButton icon={dockOpen ? 'down' : 'up'} label={t(lang, dockOpen ? 'dockHide' : 'dockShow')}
                            onClick={() => setDockOpen(!dockOpen)} className="flat" testid="toggle-dock" />
              </div>
              {dockOpen && <div className="dock-body">{plot}</div>}
            </section>
          )}
        </div>
        {!compact && sideTab(rightOpen, setRightOpen, 'list', t(lang, 'inspector'), 'right', inspector)}
      </div>
      {!compact && statusBar}
      {compact && (<>
        <div className="dockbar" role="tablist">
          {(['geom', 'controls', 'modes'] as const).map((s) => (
            <button key={s} type="button" role="tab" aria-selected={sheet === s} className={sheet === s ? 'on' : ''}
                    data-testid={`sheet-${s}`} onClick={() => setSheet(sheet === s ? null : s)}>
              <Icon name={s === 'geom' ? 'cube' : s === 'controls' ? 'sliders' : 'list'} size={16} />{sheetTitle[s]}
            </button>
          ))}
        </div>
        {sheet && (<>
          <div className="sheet-backdrop" onClick={() => setSheet(null)} />
          <section className="sheet" role="dialog" aria-label={sheetTitle[sheet]}>
            <div className="sheet-head">
              <span className="side-title">{sheetTitle[sheet]}</span>
              <IconButton icon="x" label={t(lang, 'sheetClose')} onClick={() => setSheet(null)} className="flat" />
            </div>
            <div className="sheet-body">
              {sheet === 'geom' && geomPanel}
              {sheet === 'controls' && (geom ? toolbar : <div className="empty-note">{t(lang, 'empty')}</div>)}
              {sheet === 'modes' && (<>{inspector}{plot && <Collapsible title={t(lang, 'axisPlot')} className="panel-section" defaultOpen={false}>{plot}</Collapsible>}</>)}
            </div>
          </section>
        </>)}
      </>)}
    </div>
  );
}

/** The colour bar's CSS gradient for the active colour map. */
function legendGradient(signed: boolean, volume: boolean, theme: Theme): string {
  return cssGradient(signed ? divergingStops(theme) : volume ? volumeStops(theme) : sequentialStops(theme), signed);
}
