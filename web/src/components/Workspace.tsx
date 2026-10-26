// The workspace page: geometry panel (left), the 3D viewer as the main canvas with a floating display toolbar,
// the inspector (right: modes, selected-mode values, comparison), the axis plot dock and the status bar.
// The field engine runs in the browser (fieldlab.ts): no interaction waits for the network.
import { useCallback, useDeferredValue, useEffect, useImperativeHandle, useMemo, useReducer, useRef, useState,
         type Ref } from 'react';
import { api, type FeatureView, type Features, type Geometry, type Info, type Prediction } from '../api';
import { gather, magnitude, modeField, fillGrid, gridPoints, sampleArrows, sampleGrid, sampleSlice, scaled, shareMesh,
         sliceArrows, thinGrid, type FieldBundle } from '../fieldlab';
import { cssGradient, divergingStops, sequentialStops, type Theme } from '../colors';
import { t, type Key, type Lang } from '../i18n';
import { int, sci } from '../format';
import { usePersisted } from '../store';
import { useOverlay } from '../overlay';
import { AXES, CUT_ARROWS, MAX_ARROWS, arrowBase, cutOf, gridOf, type Axis, type Display, type FeatureKind, type Source,
         type View3D } from '../workspace';
import GeometryPanel, { type GeomForm } from './GeometryPanel';
import GeometryInfo from './GeometryInfo';
import ModeTable from './ModeTable';
import ModeDetails from './ModeDetails';
import AxisPlot from './AxisPlot';
import ColorBar from './ColorBar';
import DisplayToolbar from './DisplayToolbar';
import Icon, { type IconName } from './Icon';
import { Collapsible, IconButton, useMedia } from './ui';
import Viewer3D, { PRESETS, phaseFactor, volumeStops, type Arrows, type SurfaceMode, type ViewerHandle,
                   type VolumeScalars } from './Viewer3D';
import { useToast } from './Toasts';

export interface WorkspaceHandle { openDataset: (did: string, sid: number) => Promise<boolean> }

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

  const [leftOpen, setLeftOpen] = usePersisted('ws.left', true);
  const [rightOpen, setRightOpen] = usePersisted('ws.right', true);
  const [dockOpen, setDockOpen] = usePersisted('ws.dock', true);
  const [sheet, setSheet] = useState<Sheet>(null);
  useOverlay(sheet !== null, () => setSheet(null));
  const [form, setForm] = useState<GeomForm>({ family: 'elliptical', sid: null, unit: 'mm', meshSize: 0.1, nCells: 1 });
  const viewer = useRef<ViewerHandle>(null);

  const run = useCallback(async <T,>(fn: () => Promise<T>, fail: string): Promise<T | undefined> => {
    setBusy(true);
    try { return await fn(); } catch (e) { toast('error', `${fail}: ${(e as Error).message}`); return undefined; } finally { setBusy(false); }
  }, [toast]);

  const truth = geom?.truth ?? null;
  const geomId = useRef<string | null>(null);              // the open geometry: late answers for another one are dropped
  const newGeometry = (g: Geometry | undefined): boolean => {
    if (!g) return false;
    geomId.current = g.id;
    setGeom(g); setPred(null); setPos(null);
    fields.current.clear();
    setFeatures(null); setFView(null); setFSurf(null);
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
    setGeom(null); setPred(null); setPos(null); setPlaying(false);
    fields.current.clear();
    setFeatures(null); setFView(null); setFSurf(null);
    setMode(0); setSource('model');
  };

  const predict = () => geom && run(async () => {
    const p = await api.predict(geom.id, truth?.id ?? null);
    if (geomId.current !== geom.id) return;               // closed (or replaced) meanwhile
    setPred(p);
    setSource('model');
    if (!truth) {
      const acc = p.modes.find((m) => m.accelerating);
      setMode(acc ? acc.mode : 0);
    }
    const sec = Object.values(p.time_s).reduce((a, b) => a + b, 0);
    toast('ok', `${t(lang, 'toastPredOk')} · ${sec.toFixed(1)} s`);
  }, t(lang, 'toastPredFail'));

  const buildSample = (family: string, id: number | null, meshSize: number) =>
    run(() => api.sample(family, id, meshSize), t(lang, 'toastGeomFail')).then(newGeometry);
  const upload = (file: File) =>
    run(() => api.upload(file, form.unit, form.meshSize, form.nCells), t(lang, 'toastGeomFail')).then(newGeometry);

  useImperativeHandle(ref, () => ({
    openDataset: (did: string, sid: number) =>
      run(() => api.datasetOpen(did, sid), t(lang, 'toastGeomFail')).then(newGeometry),
  }));

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

  const presetBtn = (id: string) => viewer.current?.preset(id);
  const homeBtn = () => viewer.current?.home();

  const toolbar = (
    <DisplayToolbar lang={lang} display={display} setDisplay={setDisplay} sources={sources} source={source}
                    setSource={setSource} field={field} setField={setField} view3d={view3d} setView3d={setView3d}
                    arrowsOn={arrowsOn} setArrowsOn={setArrowsOn} axis={axis}
                    setAxis={(x) => { setAxis(x); setPos(null); }} posValue={posValue} posRange={range} setPos={setPos}
                    phase={phase} setPhase={setPhase} playing={playing} setPlaying={setPlaying} canPlay={!!shown}
                    surfaceMode={surfaceMode} setSurfaceMode={setSurfaceMode} fKind={fKind} setFKind={setFKind}
                    feature={feature} setFeature={setFeature} featureNames={features?.names ?? []} />
  );

  const geomPanel = (
    <GeometryPanel lang={lang} info={info} geometry={geom} busy={busy} form={form} setForm={setForm}
                   onUpload={upload} onSample={buildSample} onPredict={predict} onClose={closeGeometry}
                   after={geom ? <GeometryInfo lang={lang} geometry={geom} features={features} /> : null} />
  );

  const sourceName = t(lang, source === 'fe' ? 'fe' : source === 'model' ? 'model' : 'diff');
  const inspector = (<>
    {shown ? (
      <Collapsible title={cmp ? t(lang, 'cmpTitle') : t(lang, 'modes')} className="panel-section">
        <ModeTable lang={lang} modes={shown.modes} cmp={cmp} selected={mode} onSelect={setMode} />
      </Collapsible>
    ) : <div className="empty-note">{geom ? t(lang, 'noPrediction') : t(lang, 'empty')}</div>}
    {shown && row && (
      <Collapsible title={`${t(lang, 'selected')}: ${row.mode} · ${sourceName}`} className="panel-section">
        <ModeDetails lang={lang} row={row}
                     cmp={cmp && source !== 'fe' ? cmp[mode] : null}
                     timings={source === 'fe' ? null : pred?.time_s}
                     exportId={source === 'fe' ? truth?.id : pred?.id}
                     label={source === 'fe' ? truth?.labels?.[mode] ?? null : null} />
      </Collapsible>
    )}
    {source === 'diff' && display === 'field' && <div className="note pad">{t(lang, 'diffNote')}</div>}
    {display === 'feature' && <div className="note pad">{t(lang, `fkn_${fKind}` as Key)}</div>}
    <div className="note disclaimer pad">{t(lang, 'disclaimer')}</div>
  </>);

  const plot = onField && bundle ? (
    <AxisPlot z={bundle.axis.z} ez={bundle.axis.Ez[mode]} title={t(lang, 'axisPlot')} note={t(lang, 'axisNote')}
              scale={scaleOf(refBundle, 'E') ?? scaleOf(bundle, 'E') ?? 1} empty={t(lang, 'noAxisField')} />
  ) : null;

  const qtyTitle = display === 'feature' ? view.label
    : `${view.label} · ${t(lang, 'statusMode')} ${mode}${bundle ? ` · ${bundle.f[mode]?.toFixed(4)} GHz` : ''} · ${sourceName}`;

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

  const stage = (
    <div className="stage" data-testid="stage">
      <div className="panes">
        <div className="pane">
          <Viewer3D ref={viewer} surface={geom?.surface ?? null} surfaceScalars={view.surfS} plane={view.plane}
                    planeScalars={view.planeS} volume={view.volume} arrows={view.arrows}
                    range={view.range} gain={view.gain} signed={view.signed}
                    surfaceMode={surfaceMode} theme={theme} viewAxis={axis}
                    oblique={display === 'field' && view3d === 'volume'}
                    neutralPlane={view.neutral} cells={view.cells} segments={view.segments} />
          {!geom && <div className="viewer-empty"><Icon name="cube" size={34} /><span>{t(lang, 'empty')}</span><span className="sub">{t(lang, 'dropHint')}</span></div>}
          {geom && !shown && display === 'field' && <div className="viewer-empty"><span>{t(lang, 'noPrediction')}</span></div>}
          {shown && display === 'field' && srcId && !bundle && <div className="viewer-empty"><span>{t(lang, 'loadingFields')}</span></div>}
          {(shown || display === 'feature') && geom && (
            <ColorBar gradient={legend} range={view.range} unit={view.unit} title={qtyTitle} neutral={view.neutral}
                      scaleTag={t(lang, 'scaleLinear')} />
          )}
        </div>
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
  );

  const statusBar = (
    <footer className="statusbar" aria-label="status">
      {geom ? (<>
        <span>{describe(geom)}</span>
        <span className="num">{int(geom.check.n_tets)} {t(lang, 'statusTets')}</span>
        <span className="num">{int(geom.check.n_nodes)} {t(lang, 'statusNodes')}</span>
        {features && <span className="num">{int(features.n_edges)} DOF (N0)</span>}
        {shown && row && <span className="num">{t(lang, 'statusMode')} {row.mode} · {row.f_GHz.toFixed(4)} GHz</span>}
        {pred && <span className="num">{Object.entries(pred.time_s).map(([k, v]) => `${k} ${v.toFixed(2)} s`).join(' · ')}</span>}
      </>) : <span>{t(lang, 'statusNone')}</span>}
      <span className="grow" />
      {shown && <span>{t(lang, 'statusSource')}: {sourceName}</span>}
      {info && <span className="num">{info.model.run ? `${info.model.run} · ` : ''}{(info.model.n_params / 1e6).toFixed(2)} M · {info.model.device}{info.model.untrained ? ` · ${t(lang, 'untrainedChip')}` : ''}</span>}
    </footer>
  );

  const sheetTitle: Record<Exclude<Sheet, null>, string> = {
    geom: t(lang, 'sheetGeometry'), controls: t(lang, 'sheetControls'), modes: t(lang, 'sheetModes'),
  };

  return (
    <div className={`ws ${compact ? 'compact' : ''}`} hidden={hidden} data-testid="workspace">
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
