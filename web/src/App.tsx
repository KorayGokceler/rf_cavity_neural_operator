import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api, type FeatureView, type Features, type Geometry, type Info, type ModeData, type Plane, type Prediction, type Volume } from './api';
import GeometryPanel from './components/GeometryPanel';
import GeometryInfo from './components/GeometryInfo';
import ModeTable from './components/ModeTable';
import ModeDetails from './components/ModeDetails';
import AxisPlot from './components/AxisPlot';
import DatasetPage from './components/DatasetPage';
import GeneratePage from './components/GeneratePage';
import TrainPage from './components/TrainPage';
import TrubaPage from './components/TrubaPage';
import Viewer3D, { fieldMagnitude, phaseFactor, volumeStops, type Arrows, type SurfaceMode, type VolumeScalars } from './components/Viewer3D';
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
const MAX_ARROWS = 700;                           // in the volume; ~300 on a cut

/** One point per cell of size s (first come): an even subset of scattered points [P,3]. */
function thin(points: Float32Array, s: number, keep: (i: number) => boolean = () => true): number[] {
  const seen = new Set<string>();
  const out: number[] = [];
  for (let i = 0; i < points.length / 3; i++) {
    if (!keep(i)) continue;
    const key = `${Math.floor(points[3 * i] / s)},${Math.floor(points[3 * i + 1] / s)},${Math.floor(points[3 * i + 2] / s)}`;
    if (!seen.has(key)) { seen.add(key); out.push(i); }
  }
  return out;
}

/** Arrows at the points idx of P [P,3] mm with vectors a·F; the largest amplitude |F| drawn `length` mm long. */
function arrowsAt(P: Float32Array, F: Float32Array, idx: number[], a: number, length: number): Arrows {
  const points = new Float32Array(3 * idx.length);
  const vectors = new Float32Array(3 * idx.length);
  let vmax = 0;
  idx.forEach((i, j) => {
    for (let c = 0; c < 3; c++) { points[3 * j + c] = P[3 * i + c]; vectors[3 * j + c] = a * F[3 * i + c]; }
    vmax = Math.max(vmax, Math.hypot(F[3 * i], F[3 * i + 1], F[3 * i + 2]));
  });
  return { points, vectors, vmax, length };
}

function maxMag(F: Float32Array): number {
  let m = 0;
  for (let i = 0; i < F.length; i += 3) m = Math.max(m, Math.hypot(F[i], F[i + 1], F[i + 2]));
  return m;
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
  const [modeData, setModeData] = useState<ModeData | null>(null);
  const [truthData, setTruthData] = useState<ModeData | null>(null);
  const [plane, setPlane] = useState<Plane | null>(null);
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
  const [vol, setVol] = useState<Volume | null>(null);
  const [axis, setAxis] = useState<Axis>('y');
  const [pos, setPos] = useState<number | null>(null);
  const [phase, setPhase] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [surfaceMode, setSurfaceMode] = useState<SurfaceMode>('ghost');

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
    setGeom(g); setPred(null); setModeData(null); setTruthData(null); setPlane(null); setVol(null); setPos(null);
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

  // field data of the shown solution (+ the FE one for the difference colour scale)
  useEffect(() => {
    if (!srcId || display !== 'field') return;
    let live = true;
    api.mode(srcId, mode).then((d) => live && setModeData(d)).catch((e) => setError(e.message));
    return () => { live = false; };
  }, [srcId, mode, display]);
  useEffect(() => {
    if (source !== 'diff' || !truth) return;
    let live = true;
    api.mode(truth.id, mode).then((d) => live && setTruthData(d)).catch(() => undefined);
    return () => { live = false; };
  }, [source, truth, mode]);
  useEffect(() => {
    if (!srcId || display !== 'field') return;
    let live = true;
    const h = setTimeout(() => {
      api.plane(srcId, mode, axis, pos).then((p) => live && setPlane(p)).catch((e) => setError(e.message));
    }, 120);
    return () => { live = false; clearTimeout(h); };
  }, [srcId, mode, axis, pos, display]);
  // the whole volume (3D view)
  useEffect(() => {
    if (!srcId || display !== 'field' || view3d !== 'volume') return;
    let live = true;
    setVol(null);
    api.volume(srcId, mode).then((v) => live && setVol(v)).catch((e) => setError(e.message));
    return () => { live = false; };
  }, [srcId, mode, display, view3d]);
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

  // what the viewer colours
  const view = useMemo(() => {
    if (display === 'feature') {
      return { plane: fView?.plane ?? null, planeS: fView?.plane?.scalars ?? null, surfS: fSurf,
               volume: null as VolumeScalars | null, arrows: null as Arrows | null,
               range: (fView?.range ?? [0, 1]) as [number, number], signed: fView?.signed ?? false,
               label: fView?.legend ?? '', unit: '', neutral: fView?.neutral ?? false,
               cells: fView?.cells ?? null, segments: fView?.segments ?? [] };
    }
    // colour = |E| or |H| (the scalar field); the arrows carry the direction
    const ref = source === 'diff' ? truthData : modeData;
    const surfMax = ref ? (field === 'E' ? ref.surface.E_max : ref.surface.H_max) : 1;
    const isVol = view3d === 'volume';
    const a = phaseFactor(field, phase);
    const VF = vol ? (field === 'E' ? vol.E : vol.H) : null;
    const PF = plane ? (field === 'E' ? plane.E : plane.H) : null;
    const localMax = isVol ? (VF ? maxMag(VF) : 0) : (PF ? maxMag(PF) : 0);
    const lim = source === 'diff' ? surfMax : Math.max(surfMax, localMax);      // differences on the FE scale
    let volume: VolumeScalars | null = null;
    let arrows: Arrows | null = null;
    if (isVol && vol && VF) {
      volume = { dims: vol.dims, origin: vol.origin, spacing: vol.spacing, scalars: fieldMagnitude(VF, field, phase) };
      if (arrowsOn) {
        const st = Math.max(1, Math.round(Math.cbrt(vol.nInside / MAX_ARROWS)));
        const [nx, ny, nz] = vol.dims;
        const P = new Float32Array(3 * nx * ny * nz);
        for (let k = 0; k < nz; k++) for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
          const g = i + nx * (j + ny * k);
          P[3 * g] = vol.origin[0] + i * vol.spacing; P[3 * g + 1] = vol.origin[1] + j * vol.spacing;
          P[3 * g + 2] = vol.origin[2] + k * vol.spacing;
        }
        const idx: number[] = [];
        for (let k = 0; k < nz; k += st) for (let j = 0; j < ny; j += st) for (let i = 0; i < nx; i += st) {
          const g = i + nx * (j + ny * k);
          if (vol.inside[g]) idx.push(g);
        }
        arrows = arrowsAt(P, VF, idx, a, 1.3 * st * vol.spacing);
      }
    } else if (!isVol && plane && PF && arrowsOn) {
      const n = plane.points.length / 3;
      const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      for (let i = 0; i < n; i++) for (let c = 0; c < 3; c++) {
        lo[c] = Math.min(lo[c], plane.points[3 * i + c]); hi[c] = Math.max(hi[c], plane.points[3 * i + c]);
      }
      const ext = [0, 1, 2].map((c) => hi[c] - lo[c]).sort((x, y) => y - x);
      const s = Math.sqrt((ext[0] * ext[1]) / Math.min(n, 300)) || 1;          // ~300 arrows on the cut
      arrows = arrowsAt(plane.points, PF, thin(plane.points, s), a, 1.3 * s);
    }
    return {
      plane: isVol ? null : plane, planeS: !isVol && PF ? fieldMagnitude(PF, field, phase) : null,
      surfS: modeData ? fieldMagnitude(field === 'E' ? modeData.surface.E : modeData.surface.H, field, phase) : null,
      volume, arrows, range: [0, lim] as [number, number], signed: false,
      label: `|${source === 'diff' ? 'Δ' : ''}${field}|, U = 1 J`,
      unit: field === 'E' ? 'V/m' : 'A/m', neutral: false, cells: null, segments: [],
    };
  }, [display, fView, fSurf, source, truthData, modeData, plane, vol, view3d, arrowsOn, field, phase]);
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
                        range={view.range} signed={view.signed}
                        surfaceMode={surfaceMode} theme={theme} viewAxis={axis}
                        oblique={display === 'field' && view3d === 'volume'}
                        neutralPlane={view.neutral} cells={view.cells} segments={view.segments} />
              {!geom && <div className="viewer-empty">{t(lang, 'empty')}</div>}
              {geom && !shown && display === 'field' && <div className="viewer-empty">{t(lang, 'noPrediction')}</div>}
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

            {modeData && display === 'field' && <AxisPlot z={modeData.axis.z_mm} ez={modeData.axis.Ez} title={t(lang, 'axisPlot')} note={t(lang, 'axisNote')}
                                                          scale={(source === 'diff' ? truthData : modeData)?.surface.E_max ?? modeData.surface.E_max}
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
