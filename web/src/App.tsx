import { useCallback, useDeferredValue, useEffect, useMemo, useReducer, useRef, useState } from 'react';
import { api, type FeatureView, type Features, type Geometry, type Info, type Prediction } from './api';
import { fillGrid, gather, gridPoints, magnitude, modeField, rasterize, sampleArrows, sampleGrid, sampleSlice, scaled,
         shareMesh, sliceArrows, slicePlane, thinGrid, type FieldBundle, type Grid, type Slice } from './fieldlab';
import GeometryPanel from './components/GeometryPanel';
import GeometryInfo from './components/GeometryInfo';
import ModeTable from './components/ModeTable';
import ModeDetails from './components/ModeDetails';
import AxisPlot from './components/AxisPlot';
import DatasetPage from './components/DatasetPage';
import GeneratePage from './components/GeneratePage';
import TrainPage from './components/TrainPage';
import TrubaPage from './components/TrubaPage';
import Viewer3D, { phaseFactor, volumeStops, type Arrows, type SurfaceMode, type VolumeScalars } from './components/Viewer3D';
import { cssGradient, divergingStops, sequentialStops, type Theme } from './colors';
import { t, type Key, type Lang } from './i18n';
import { sci } from './format';

const AXES = ['x', 'y', 'z'] as const;
type Axis = (typeof AXES)[number];
type Source = 'fe' | 'model' | 'diff';
type Display = 'field' | 'feature';
type FeatureKind = 'position' | 'wall' | 'torsion' | 'mesh' | 'raw';
const FEATURE_KINDS: FeatureKind[] = ['position', 'wall', 'torsion', 'mesh', 'raw'];
type View3D = 'cut' | 'volume';                    // field on a cut plane | |field| in the whole volume
const MAX_ARROWS = 700;                           // in the volume
const CUT_ARROWS = 300;                           // on a cut

// cuts and the 3D grid per mesh (the FE, model and difference solutions of a geometry share one: shareMesh)
const cutCache = new WeakMap<Uint32Array, Map<string, Slice>>();
const gridCache = new WeakMap<Uint32Array, Grid>();
function cutOf(b: FieldBundle, axis: 0 | 1 | 2, pos: number): Slice {
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
function gridOf(b: FieldBundle): Grid {
  let g = gridCache.get(b.tets);
  if (!g) { g = rasterize(b); gridCache.set(b.tets, g); }
  return g;
}
interface ArrowBase { points: Float32Array; vectors: Float32Array; vmax: number; length: number }
function arrowBase(points: Float32Array, vectors: Float32Array, length: number): ArrowBase {
  let vmax = 0;
  for (let i = 0; i < vectors.length; i += 3) vmax = Math.max(vmax, Math.hypot(vectors[i], vectors[i + 1], vectors[i + 2]));
  return { points, vectors, vmax, length };
}

function initialTheme(): Theme {
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export default function App() {
  const [lang, setLang] = useState<Lang>(navigator.language?.startsWith('tr') ? 'tr' : 'en');
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const [page, setPage] = useState<'work' | 'data' | 'gen' | 'train' | 'truba'>('work');
  const [info, setInfo] = useState<Info | null>(null);
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
  const [error, setError] = useState<string | null>(null);

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

  useEffect(() => { document.documentElement.dataset.theme = theme; }, [theme]);
  // the theme attribute before the re-render: the 3D view reads its colours (CSS variables) in its effects,
  // which run before this component's
  const changeTheme = (th: Theme) => { document.documentElement.dataset.theme = th; setTheme(th); };
  useEffect(() => { document.documentElement.lang = lang; }, [lang]);
  useEffect(() => { api.info().then(setInfo).catch((e) => setError(String(e.message ?? e))); }, []);

  const run = useCallback(async <T,>(fn: () => Promise<T>): Promise<T | undefined> => {
    setBusy(true); setError(null);
    try { return await fn(); } catch (e) { setError((e as Error).message); return undefined; } finally { setBusy(false); }
  }, []);

  const truth = geom?.truth ?? null;
  const newGeometry = (g: Geometry | undefined) => {
    if (!g) return;
    setGeom(g); setPred(null); setPos(null);
    fields.current.clear();
    setFeatures(null); setFView(null); setFSurf(null);
    const acc = g.truth?.modes.find((m) => m.accelerating);
    setMode(acc ? acc.mode : 0);
    setSource(g.truth ? 'fe' : 'model');
    api.features(g.id).then(setFeatures).catch(() => setFeatures(null));
  };

  const predict = () => geom && run(async () => {
    const p = await api.predict(geom.id, truth?.id ?? null);
    setPred(p);
    setSource('model');
    if (!truth) {
      const acc = p.modes.find((m) => m.accelerating);
      setMode(acc ? acc.mode : 0);
    }
  });

  const openItem = (did: string, sid: number) => run(() => api.datasetOpen(did, sid)).then((g) => {
    newGeometry(g);
    if (g) setPage('work');
  });

  const srcId = source === 'fe' ? truth?.id : source === 'diff' ? pred?.comparison?.diff_id : pred?.id;
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
    }).catch((e) => { cache.delete(id); setError(e.message); });
    cache.set(id, job);
  }, []);
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
        .then((v) => live && setFView(v)).catch((e) => setError(e.message));
    }, 120);
    if (fKind === 'raw') api.featureSurface(geom.id, feature).then((v) => live && setFSurf(v)).catch(() => undefined);
    else setFSurf(null);
    return () => { live = false; clearTimeout(h); };
  }, [geom, fKind, feature, axis, pos, display]);

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

  // the shown field, computed here: a cut / the grid per mesh, the vectors per (solution, mode, E/H), and per
  // phase step only the arrows and the gain of the colour maps
  const isVol = view3d === 'volume';
  const onField = display === 'field' && !!bundle && mode < bundle.K;
  const F = useMemo(() => (onField && bundle ? modeField(bundle, field, mode) : null), [onField, bundle, field, mode]);
  const cut = useMemo(() => (onField && bundle && !isVol ? cutOf(bundle, ai as 0 | 1 | 2, cutPos) : null),
                      [onField, bundle, isVol, ai, cutPos]);
  const grid = useMemo(() => (onField && bundle && isVol ? gridOf(bundle) : null), [onField, bundle, isVol]);
  const cutV = useMemo(() => (cut && F ? sampleSlice(cut, F) : null), [cut, F]);
  const gridV = useMemo(() => (grid && F ? sampleGrid(grid, F) : null), [grid, F]);
  const planeAmp = useMemo(() => (cutV ? magnitude(cutV) : null), [cutV]);
  const surfAmp = useMemo(() => (bundle && F ? magnitude(gather(F, bundle.vid)) : null), [bundle, F]);
  const volume = useMemo<VolumeScalars | null>(() => (grid && gridV ? {
    dims: grid.dims, origin: grid.origin, spacing: grid.spacing, scalars: fillGrid(grid, magnitude(gridV)),
  } : null), [grid, gridV]);
  const cutSel = useMemo(() => (cut && arrowsOn ? sliceArrows(cut, CUT_ARROWS) : null), [cut, arrowsOn]);
  const gridSel = useMemo(() => {
    if (!grid || !arrowsOn) return null;
    const { sel, step } = thinGrid(grid, MAX_ARROWS);
    return { sel, step, points: gridPoints(grid, sel) };
  }, [grid, arrowsOn]);
  const arrows0 = useMemo(() => {
    if (gridSel && gridV && grid) return arrowBase(gridSel.points, gather(gridV, gridSel.sel), 1.3 * gridSel.step * grid.spacing);
    if (cutSel && cutV) return arrowBase(cutSel.points, sampleArrows(cutV, cutSel.ids, cutSel.w), 1.3 * cutSel.step);
    return null;
  }, [gridSel, gridV, grid, cutSel, cutV]);
  const a = phaseFactor(field, phase);
  const arrows = useMemo<Arrows | null>(() => (arrows0 ? { ...arrows0, vectors: scaled(arrows0.vectors, a) } : null),
                                        [arrows0, a]);
  const scaleOf = (b: FieldBundle | null, f: 'E' | 'H') => (b && mode < b.K ? (f === 'E' ? b.Emax : b.Hmax)[mode] : null);
  const lim = scaleOf(refBundle, field) ?? scaleOf(bundle, field) ?? 1;     // differences on the FE scale

  // what the viewer colours
  const view = useMemo(() => {
    if (display === 'feature') {
      return { plane: fView?.plane ?? null, planeS: fView?.plane?.scalars ?? null, surfS: fSurf,
               volume: null as VolumeScalars | null, arrows: null as Arrows | null,
               range: (fView?.range ?? [0, 1]) as [number, number], gain: 1, signed: fView?.signed ?? false,
               label: fView?.legend ?? '', unit: '', neutral: fView?.neutral ?? false,
               cells: fView?.cells ?? null, segments: fView?.segments ?? [] };
    }
    // colour = |E| or |H| (the scalar field) at the phase: the amplitude times the gain; the arrows carry the
    // direction
    return {
      plane: isVol ? null : cut, planeS: isVol ? null : planeAmp, surfS: surfAmp, volume, arrows,
      range: [0, lim] as [number, number], gain: Math.abs(a), signed: false,
      label: `|${source === 'diff' ? 'Δ' : ''}${field}|, U = 1 J`,
      unit: field === 'E' ? 'V/m' : 'A/m', neutral: false, cells: null, segments: [],
    };
  }, [display, fView, fSurf, isVol, cut, planeAmp, surfAmp, volume, arrows, lim, a, source, field]);
  const legend = useMemo(() => cssGradient(view.signed ? divergingStops(theme) : view.volume ? volumeStops(theme)
    : sequentialStops(theme), view.signed), [view.signed, view.volume, theme]);

  const shown = source === 'fe' ? truth : pred;      // model + diff rows come from the model solution
  const cmp = pred?.comparison?.rows ?? null;
  const row = shown?.modes[mode];

  return (
    <div className="app">
      <header>
        <div>
          <h1>{t(lang, 'title')}</h1>
          <div className="sub">{t(lang, 'subtitle')}</div>
        </div>
        <nav className="seg" role="tablist">
          <button role="tab" aria-selected={page === 'work'} className={page === 'work' ? 'on' : ''} onClick={() => setPage('work')}>{t(lang, 'workspace')}</button>
          <button role="tab" aria-selected={page === 'data'} className={page === 'data' ? 'on' : ''} onClick={() => setPage('data')}>{t(lang, 'datasetTab')}</button>
          <button role="tab" aria-selected={page === 'gen'} className={page === 'gen' ? 'on' : ''} onClick={() => setPage('gen')}>{t(lang, 'generateTab')}</button>
          <button role="tab" aria-selected={page === 'train'} className={page === 'train' ? 'on' : ''} onClick={() => setPage('train')}>{t(lang, 'trainTab')}</button>
          <button role="tab" aria-selected={page === 'truba'} className={page === 'truba' ? 'on' : ''} onClick={() => setPage('truba')}>{t(lang, 'trubaTab')}</button>
        </nav>
        <div className="header-right">
          {info && <span className="badge">{info.model.run ? `${info.model.run} · ` : ''}{(info.model.n_params / 1e6).toFixed(2)} M · {info.model.device}</span>}
          <button className="ghost" onClick={() => setLang(lang === 'tr' ? 'en' : 'tr')}>{lang === 'tr' ? 'EN' : 'TR'}</button>
          <button className="ghost" onClick={() => changeTheme(theme === 'light' ? 'dark' : 'light')} aria-label="theme">{theme === 'light' ? '☾' : '☀'}</button>
        </div>
      </header>

      {info?.model.untrained && <div className="banner warning"><span className="icon">!</span>{t(lang, 'untrained')}{info.model.load_error ? ` — ${t(lang, 'ckptFailed')}: ${info.model.load_error}` : ''}</div>}
      {pred?.warnings.filter((w) => !w.startsWith('untrained')).map((w) =>
        <div key={w} className="banner warning"><span className="icon">!</span>{w}</div>)}
      {error && <div className="banner critical"><span className="icon">✕</span>{error}</div>}

      {page === 'truba' ? <TrubaPage lang={lang} /> :
       page === 'train' ? <TrainPage lang={lang} model={info?.model ?? null} onModel={() => { api.info().then(setInfo).catch(() => undefined); setPred(null); }} /> :
       page === 'gen' ? <GeneratePage lang={lang} onShowDatasets={() => setPage('data')} /> :
       page === 'data' ? <DatasetPage lang={lang} onOpen={openItem} busy={busy} /> : (
        <main>
          <div className="leftcol">
            <GeometryPanel lang={lang} info={info} geometry={geom} busy={busy}
                           onUpload={(f, u, m, n) => run(() => api.upload(f, u, m, n)).then(newGeometry)}
                           onSample={(fam, id, m) => run(() => api.sample(fam, id, m)).then(newGeometry)}
                           onPredict={predict} />
            {geom && <GeometryInfo lang={lang} geometry={geom} features={features} />}
          </div>

          <section className="center">
            <div className="viewer-card">
              <Viewer3D surface={geom?.surface ?? null} surfaceScalars={view.surfS} plane={view.plane}
                        planeScalars={view.planeS} volume={view.volume} arrows={view.arrows}
                        range={view.range} gain={view.gain} signed={view.signed}
                        surfaceMode={surfaceMode} theme={theme} viewAxis={axis}
                        oblique={display === 'field' && view3d === 'volume'}
                        neutralPlane={view.neutral} cells={view.cells} segments={view.segments} />
              {!geom && <div className="viewer-empty">{t(lang, 'empty')}</div>}
              {geom && !shown && display === 'field' && <div className="viewer-empty">{t(lang, 'noPrediction')}</div>}
              {shown && display === 'field' && srcId && !bundle && <div className="viewer-empty">{t(lang, 'loadingFields')}</div>}
              {(shown || display === 'feature') && geom && (
                <div className="legend">
                  {!view.neutral && <>
                    <span>{sci(view.range[0])}</span>
                    <div className="legend-bar" style={{ background: legend }} />
                    <span>{sci(view.range[1])} {view.unit}</span>
                  </>}
                  <span className="muted">{view.label}</span>
                </div>
              )}
            </div>

            <div className="controls">
              <label>{t(lang, 'display')}
                <select value={display} onChange={(e) => setDisplay(e.target.value as Display)}>
                  <option value="field">{t(lang, 'fieldMode')}</option>
                  <option value="feature">{t(lang, 'featureMode')}</option>
                </select>
              </label>
              {display === 'field' ? (<>
                {sources.length > 1 && (
                  <div className="seg" role="group" aria-label={t(lang, 'source')}>
                    {sources.map((s) => <button key={s} className={source === s ? 'on' : ''} onClick={() => setSource(s)}>{t(lang, s)}</button>)}
                  </div>
                )}
                <div className="seg" role="group" aria-label={t(lang, 'field')}>
                  {(['E', 'H'] as const).map((f) => <button key={f} className={field === f ? 'on' : ''} onClick={() => setField(f)}>{f}</button>)}
                </div>
                <label>{t(lang, 'view3d')}
                  <select value={view3d} onChange={(e) => setView3d(e.target.value as View3D)}>
                    <option value="cut">{t(lang, 'viewCut')}</option>
                    <option value="volume">{t(lang, 'viewVolume')}</option>
                  </select>
                </label>
                <label className="check"><input type="checkbox" checked={arrowsOn} onChange={(e) => setArrowsOn(e.target.checked)} /><span>{t(lang, 'arrowsOn')}</span></label>
              </>) : (<>
                <label>{t(lang, 'featureMode')}
                  <select value={fKind} onChange={(e) => setFKind(e.target.value as FeatureKind)}>
                    {FEATURE_KINDS.map((k) => <option key={k} value={k}>{t(lang, `fk_${k}` as Key)}</option>)}
                  </select>
                </label>
                {fKind === 'raw' && (
                  <label>{t(lang, 'channel')}
                    <select value={feature} onChange={(e) => setFeature(e.target.value)}>
                      {(features?.names ?? []).map((n) => <option key={n} value={n}>{n}</option>)}
                    </select>
                  </label>
                )}
              </>)}
              <label>{t(lang, display === 'field' && view3d === 'volume' ? 'viewDir' : 'cut')}
                <select value={axis} onChange={(e) => { setAxis(e.target.value as Axis); setPos(null); }}>
                  {AXES.map((a) => <option key={a} value={a}>⟂ {a}</option>)}
                </select>
              </label>
              {(display === 'feature' || view3d === 'cut') && (
                <label className="grow">{t(lang, 'position')} {posValue.toFixed(1)} mm
                  <input type="range" min={range[0]} max={range[1]} step={(range[1] - range[0]) / 200 || 1}
                         value={posValue} onChange={(e) => setPos(Number(e.target.value))} />
                </label>
              )}
              {display === 'field' && (<>
                <label className="grow">{t(lang, 'phase')} {phase}°
                  <input type="range" min={0} max={180} step={5} value={phase} onChange={(e) => setPhase(Number(e.target.value))} />
                </label>
                <button onClick={() => setPlaying(!playing)} disabled={!shown}>{playing ? `❚❚ ${t(lang, 'pause')}` : `▶ ${t(lang, 'play')}`}</button>
              </>)}
              <label>{t(lang, 'surface')}
                <select value={surfaceMode} onChange={(e) => setSurfaceMode(e.target.value as SurfaceMode)}>
                  <option value="ghost">{t(lang, 'ghost')}</option>
                  <option value="field">{t(lang, 'wallField')}</option>
                  <option value="mesh">{t(lang, 'meshView')}</option>
                  <option value="hidden">{t(lang, 'hidden')}</option>
                </select>
              </label>
            </div>
            {source === 'diff' && display === 'field' && <div className="note">{t(lang, 'diffNote')}</div>}
            {display === 'feature' && <div className="note">{t(lang, `fkn_${fKind}` as Key)}</div>}

            {onField && bundle && <AxisPlot z={bundle.axis.z} ez={bundle.axis.Ez[mode]} title={t(lang, 'axisPlot')} note={t(lang, 'axisNote')}
                                            scale={scaleOf(refBundle, 'E') ?? scaleOf(bundle, 'E') ?? 1}
                                            empty={t(lang, 'noAxisField')} />}
          </section>

          <aside className="panel right">
            {shown && <ModeTable lang={lang} modes={shown.modes} cmp={cmp} selected={mode} onSelect={setMode} />}
            {shown && row && (
              <ModeDetails lang={lang} row={row} source={t(lang, source === 'fe' ? 'fe' : 'model')}
                           cmp={cmp && source !== 'fe' ? cmp[mode] : null}
                           timings={source === 'fe' ? null : pred?.time_s}
                           exportId={source === 'fe' ? truth?.id : pred?.id}
                           label={source === 'fe' ? truth?.labels?.[mode] ?? null : null} />
            )}
            <div className="note disclaimer">{t(lang, 'disclaimer')}</div>
          </aside>
        </main>
      )}
    </div>
  );
}
