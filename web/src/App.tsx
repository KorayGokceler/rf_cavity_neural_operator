import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api, type Geometry, type Info, type ModeData, type Plane, type Prediction } from './api';
import GeometryPanel from './components/GeometryPanel';
import ModeTable from './components/ModeTable';
import ModeDetails from './components/ModeDetails';
import AxisPlot from './components/AxisPlot';
import Viewer3D, { type Comp, type SurfaceMode } from './components/Viewer3D';
import { cssGradient, divergingStops, sequentialStops, type Theme } from './colors';
import { t, type Lang } from './i18n';
import { sci } from './format';

const AXES = ['x', 'y', 'z'] as const;
type Axis = (typeof AXES)[number];

function initialTheme(): Theme {
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export default function App() {
  const [lang, setLang] = useState<Lang>(navigator.language?.startsWith('tr') ? 'tr' : 'en');
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const [info, setInfo] = useState<Info | null>(null);
  const [geom, setGeom] = useState<Geometry | null>(null);
  const [pred, setPred] = useState<Prediction | null>(null);
  const [mode, setMode] = useState(0);
  const [modeData, setModeData] = useState<ModeData | null>(null);
  const [plane, setPlane] = useState<Plane | null>(null);
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

  const newGeometry = (g: Geometry | undefined) => {
    if (!g) return;
    setGeom(g); setPred(null); setModeData(null); setPlane(null); setMode(0); setPos(null);
  };

  const predict = () => geom && run(async () => {
    const p = await api.predict(geom.id);
    setPred(p);
    const acc = p.modes.find((m) => m.accelerating);
    setMode(acc ? acc.mode : 0);
  });

  // mode data (surface fields + axis profile)
  useEffect(() => {
    if (!pred) return;
    let live = true;
    api.mode(pred.id, mode).then((d) => live && setModeData(d)).catch((e) => setError(e.message));
    return () => { live = false; };
  }, [pred, mode]);

  // cut plane (debounced on the position slider)
  useEffect(() => {
    if (!pred) return;
    let live = true;
    const h = setTimeout(() => {
      api.plane(pred.id, mode, axis, pos).then((p) => live && setPlane(p)).catch((e) => setError(e.message));
    }, 120);
    return () => { live = false; clearTimeout(h); };
  }, [pred, mode, axis, pos]);

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
  const limit = modeData ? (field === 'E' ? modeData.surface.E_max : modeData.surface.H_max) : 1;
  const unit = field === 'E' ? 'V/m' : 'A/m';
  const signed = comp !== 'abs';
  const legend = useMemo(() => cssGradient(signed ? divergingStops(theme) : sequentialStops(theme), signed),
    [signed, theme]);
  const row = pred?.modes[mode];

  return (
    <div className="app">
      <header>
        <div>
          <h1>{t(lang, 'title')}</h1>
          <div className="sub">{t(lang, 'subtitle')}</div>
        </div>
        <div className="header-right">
          {info && <span className="badge">{info.model.field}-field · {(info.model.n_params / 1e6).toFixed(2)} M · {info.model.device}</span>}
          <button className="ghost" onClick={() => setLang(lang === 'tr' ? 'en' : 'tr')}>{lang === 'tr' ? 'EN' : 'TR'}</button>
          <button className="ghost" onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}
                  aria-label="theme">{theme === 'light' ? '☾' : '☀'}</button>
        </div>
      </header>

      {info?.model.untrained && <div className="banner warning"><span className="icon">!</span>{t(lang, 'untrained')}</div>}
      {pred?.warnings.filter((w) => !w.startsWith('untrained')).map((w) =>
        <div key={w} className="banner warning"><span className="icon">!</span>{w}</div>)}
      {error && <div className="banner critical"><span className="icon">✕</span>{error}</div>}

      <main>
        <GeometryPanel lang={lang} info={info} geometry={geom} busy={busy}
                       onUpload={(f, u, m, n) => run(() => api.upload(f, u, m, n)).then(newGeometry)}
                       onSample={(fam, id, m) => run(() => api.sample(fam, id, m)).then(newGeometry)}
                       onPredict={predict} />

        <section className="center">
          <div className="viewer-card">
            <Viewer3D surface={geom?.surface ?? null}
                      surfaceFields={modeData ? { E: modeData.surface.E, H: modeData.surface.H } : null}
                      plane={plane} field={field} comp={comp} phaseDeg={phase} limit={limit}
                      surfaceMode={surfaceMode} theme={theme} viewAxis={axis} />
            {!geom && <div className="viewer-empty">{t(lang, 'empty')}</div>}
            {geom && !pred && <div className="viewer-empty">{t(lang, 'noPrediction')}</div>}
            {pred && (
              <div className="legend">
                <span>{signed ? `−${sci(limit)}` : '0'}</span>
                <div className="legend-bar" style={{ background: legend }} />
                <span>{sci(limit)} {unit}</span>
                <span className="muted">{signed ? `${field}${comp}` : `|${field}|`}, U = 1 J</span>
              </div>
            )}
          </div>

          <div className="controls" aria-disabled={!pred}>
            <div className="seg" role="group" aria-label={t(lang, 'field')}>
              {(['E', 'H'] as const).map((f) =>
                <button key={f} className={field === f ? 'on' : ''} onClick={() => setField(f)}>{f}</button>)}
            </div>
            <label>{t(lang, 'component')}
              <select value={comp} onChange={(e) => setComp(e.target.value as Comp)}>
                <option value="abs">|·|</option><option value="x">x</option><option value="y">y</option><option value="z">z</option>
              </select>
            </label>
            <label>{t(lang, 'cut')}
              <select value={axis} onChange={(e) => { setAxis(e.target.value as Axis); setPos(null); }}>
                {AXES.map((a) => <option key={a} value={a}>⟂ {a}</option>)}
              </select>
            </label>
            <label className="grow">{t(lang, 'position')} {posValue.toFixed(1)} mm
              <input type="range" min={range[0]} max={range[1]} step={(range[1] - range[0]) / 200 || 1}
                     value={posValue} onChange={(e) => setPos(Number(e.target.value))} />
            </label>
            <label className="grow">{t(lang, 'phase')} {phase}°
              <input type="range" min={0} max={180} step={5} value={phase} onChange={(e) => setPhase(Number(e.target.value))} />
            </label>
            <button onClick={() => setPlaying(!playing)} disabled={!pred}>{playing ? `❚❚ ${t(lang, 'pause')}` : `▶ ${t(lang, 'play')}`}</button>
            <label>{t(lang, 'surface')}
              <select value={surfaceMode} onChange={(e) => setSurfaceMode(e.target.value as SurfaceMode)}>
                <option value="ghost">{t(lang, 'ghost')}</option>
                <option value="field">{t(lang, 'wallField')}</option>
                <option value="hidden">{t(lang, 'hidden')}</option>
              </select>
            </label>
          </div>

          {modeData && <AxisPlot z={modeData.axis.z_mm} ez={modeData.axis.Ez} title={t(lang, 'axisPlot')} note={t(lang, 'axisNote')} />}
        </section>

        <aside className="panel right">
          {pred && <ModeTable lang={lang} modes={pred.modes} selected={mode} onSelect={setMode} />}
          {pred && row && <ModeDetails lang={lang} pred={pred} row={row} />}
          <div className="note disclaimer">{t(lang, 'disclaimer')}</div>
        </aside>
      </main>
    </div>
  );
}
