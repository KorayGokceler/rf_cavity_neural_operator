import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api, type FeatureView, type Features, type Geometry, type Info, type ModeData, type Plane, type Prediction } from './api';
import GeometryPanel from './components/GeometryPanel';
import GeometryInfo from './components/GeometryInfo';
import ModeTable from './components/ModeTable';
import ModeDetails from './components/ModeDetails';
import AxisPlot from './components/AxisPlot';
import DatasetPage from './components/DatasetPage';
import Viewer3D, { fieldScalars, type Comp, type SurfaceMode } from './components/Viewer3D';
import { cssGradient, divergingStops, sequentialStops, type Theme } from './colors';
import { t, type Key, type Lang } from './i18n';
import { sci } from './format';

const AXES = ['x', 'y', 'z'] as const;
type Axis = (typeof AXES)[number];
type Source = 'fe' | 'model' | 'diff';
type Display = 'field' | 'feature';
type FeatureKind = 'position' | 'wall' | 'torsion' | 'mesh' | 'raw';
const FEATURE_KINDS: FeatureKind[] = ['position', 'wall', 'torsion', 'mesh', 'raw'];

function initialTheme(): Theme {
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export default function App() {
  const [lang, setLang] = useState<Lang>(navigator.language?.startsWith('tr') ? 'tr' : 'en');
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const [page, setPage] = useState<'work' | 'data'>('work');
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
  const [comp, setComp] = useState<Comp>('abs');
  const [axis, setAxis] = useState<Axis>('y');
  const [pos, setPos] = useState<number | null>(null);
  const [phase, setPhase] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [surfaceMode, setSurfaceMode] = useState<SurfaceMode>('ghost');

  useEffect(() => { document.documentElement.dataset.theme = theme; }, [theme]);
  useEffect(() => { document.documentElement.lang = lang; }, [lang]);
  useEffect(() => { api.info().then(setInfo).catch((e) => setError(String(e.message ?? e))); }, []);

  const run = useCallback(async <T,>(fn: () => Promise<T>): Promise<T | undefined> => {
    setBusy(true); setError(null);
    try { return await fn(); } catch (e) { setError((e as Error).message); return undefined; } finally { setBusy(false); }
  }, []);

  const truth = geom?.truth ?? null;
  const newGeometry = (g: Geometry | undefined) => {
    if (!g) return;
    setGeom(g); setPred(null); setModeData(null); setTruthData(null); setPlane(null); setPos(null);
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
               range: (fView?.range ?? [0, 1]) as [number, number], signed: fView?.signed ?? false,
               label: fView?.legend ?? '', unit: '', neutral: fView?.neutral ?? false,
               cells: fView?.cells ?? null, segments: fView?.segments ?? [] };
    }
    const ref = source === 'diff' ? truthData : modeData;
    const lim = ref ? (field === 'E' ? ref.surface.E_max : ref.surface.H_max) : 1;
    const signed = comp !== 'abs';
    return {
      plane, planeS: plane ? fieldScalars(field === 'E' ? plane.E : plane.H, field, comp, phase) : null,
      surfS: modeData ? fieldScalars(field === 'E' ? modeData.surface.E : modeData.surface.H, field, comp, phase) : null,
      range: (signed ? [-lim, lim] : [0, lim]) as [number, number], signed,
      label: `${source === 'diff' ? 'Δ' : ''}${signed ? `${field}${comp}` : `|${field}|`}, U = 1 J`,
      unit: field === 'E' ? 'V/m' : 'A/m', neutral: false, cells: null, segments: [],
    };
  }, [display, fView, fSurf, source, truthData, modeData, plane, field, comp, phase]);
  const legend = useMemo(() => cssGradient(view.signed ? divergingStops(theme) : sequentialStops(theme), view.signed),
    [view.signed, theme]);

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
        </nav>
        <div className="header-right">
          {info && <span className="badge">{info.model.field}-field · {(info.model.n_params / 1e6).toFixed(2)} M · {info.model.device}</span>}
          <button className="ghost" onClick={() => setLang(lang === 'tr' ? 'en' : 'tr')}>{lang === 'tr' ? 'EN' : 'TR'}</button>
          <button className="ghost" onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')} aria-label="theme">{theme === 'light' ? '☾' : '☀'}</button>
        </div>
      </header>

      {info?.model.untrained && <div className="banner warning"><span className="icon">!</span>{t(lang, 'untrained')}</div>}
      {pred?.warnings.filter((w) => !w.startsWith('untrained')).map((w) =>
        <div key={w} className="banner warning"><span className="icon">!</span>{w}</div>)}
      {error && <div className="banner critical"><span className="icon">✕</span>{error}</div>}

      {page === 'data' ? <DatasetPage lang={lang} onOpen={openItem} busy={busy} /> : (
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
                        planeScalars={view.planeS} range={view.range} signed={view.signed}
                        surfaceMode={surfaceMode} theme={theme} viewAxis={axis}
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
                <label>{t(lang, 'component')}
                  <select value={comp} onChange={(e) => setComp(e.target.value as Comp)}>
                    <option value="abs">|·|</option><option value="x">x</option><option value="y">y</option><option value="z">z</option>
                  </select>
                </label>
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
              <label>{t(lang, 'cut')}
                <select value={axis} onChange={(e) => { setAxis(e.target.value as Axis); setPos(null); }}>
                  {AXES.map((a) => <option key={a} value={a}>⟂ {a}</option>)}
                </select>
              </label>
              <label className="grow">{t(lang, 'position')} {posValue.toFixed(1)} mm
                <input type="range" min={range[0]} max={range[1]} step={(range[1] - range[0]) / 200 || 1}
                       value={posValue} onChange={(e) => setPos(Number(e.target.value))} />
              </label>
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
