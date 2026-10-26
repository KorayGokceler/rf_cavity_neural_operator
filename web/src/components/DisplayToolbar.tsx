// Floating display toolbar of the viewer: what is shown (field / input feature, E / H, cut / volume, arrows, axis,
// surface), where (cut position) and when (phase). Used as an overlay on the canvas and inside the phone sheet.
import { t, type Key, type Lang } from '../i18n';
import { AXES, FEATURE_KINDS, type Axis, type Display, type FeatureKind, type Source, type View3D } from '../workspace';
import type { SurfaceMode } from './Viewer3D';
import Icon from './Icon';
import { IconButton, Menu, Seg } from './ui';

export interface ToolbarProps {
  lang: Lang;
  display: Display; setDisplay: (d: Display) => void;
  sources: Source[]; source: Source; setSource: (s: Source) => void;
  field: 'E' | 'H'; setField: (f: 'E' | 'H') => void;
  view3d: View3D; setView3d: (v: View3D) => void;
  arrowsOn: boolean; setArrowsOn: (b: boolean) => void;
  axis: Axis; setAxis: (a: Axis) => void;
  posValue: number; posRange: number[]; setPos: (p: number) => void;
  phase: number; setPhase: (p: number) => void; playing: boolean; setPlaying: (p: boolean) => void; canPlay: boolean;
  surfaceMode: SurfaceMode; setSurfaceMode: (m: SurfaceMode) => void;
  fKind: FeatureKind; setFKind: (k: FeatureKind) => void;
  feature: string; setFeature: (f: string) => void; featureNames: string[];
  extra?: React.ReactNode;                 // more tool groups (colour scale, probe, split, export)
}

const SURFACES: { value: SurfaceMode; key: Key }[] = [
  { value: 'ghost', key: 'ghost' }, { value: 'field', key: 'wallField' }, { value: 'mesh', key: 'meshView' },
  { value: 'hidden', key: 'hidden' },
];

export default function DisplayToolbar(p: ToolbarProps) {
  const { lang } = p;
  const isField = p.display === 'field';
  const showPos = !isField || p.view3d === 'cut';
  return (
    <div className="toolbar" role="toolbar" aria-label={t(lang, 'dispTools')}>
      <div className="tb-row">
        <Seg value={p.display} label={t(lang, 'displayMode')} onChange={p.setDisplay}
             options={[{ value: 'field', icon: 'field', title: t(lang, 'dispField'), testid: 'display-field' },
                       { value: 'feature', icon: 'cpu', title: t(lang, 'dispFeature'), testid: 'display-feature' }]} />
        {isField ? (<>
          {p.sources.length > 1 && (
            <Seg value={p.source} label={t(lang, 'source')} onChange={p.setSource}
                 options={p.sources.map((s) => ({ value: s, label: t(lang, s === 'fe' ? 'srcFe' : s === 'model' ? 'model' : 'diffShort'),
                                                  title: t(lang, s), testid: `src-${s}` }))} />
          )}
          <Seg value={p.field} label={t(lang, 'field')} onChange={p.setField}
               options={[{ value: 'E', label: 'E', title: t(lang, 'fieldE'), testid: 'field-btn' },
                         { value: 'H', label: 'H', title: t(lang, 'fieldH'), testid: 'field-btn' }]} />
          <Seg value={p.view3d} label={t(lang, 'view3d')} onChange={p.setView3d}
               options={[{ value: 'cut', icon: 'slice', label: t(lang, 'cutShort'), title: t(lang, 'showCut'), testid: 'view-cut' },
                         { value: 'volume', icon: 'cube', label: t(lang, 'volShort'), title: t(lang, 'showVolume'), testid: 'view-volume' }]} />
          <IconButton icon="arrows" label={t(lang, 'arrowsTip')} active={p.arrowsOn} onClick={() => p.setArrowsOn(!p.arrowsOn)} testid="arrows" />
        </>) : (<>
          <select className="tb-select" aria-label={t(lang, 'featureMode')} value={p.fKind}
                  onChange={(e) => p.setFKind(e.target.value as FeatureKind)}>
            {FEATURE_KINDS.map((k) => <option key={k} value={k}>{t(lang, `fk_${k}` as Key)}</option>)}
          </select>
          {p.fKind === 'raw' && (
            <select className="tb-select" aria-label={t(lang, 'channel')} value={p.feature} onChange={(e) => p.setFeature(e.target.value)}>
              {p.featureNames.map((n) => <option key={n} value={n}>{n}</option>)}
            </select>
          )}
        </>)}
        <Seg value={p.axis} label={t(lang, isField && p.view3d === 'volume' ? 'axisLook' : 'axisCut')} onChange={p.setAxis}
             options={AXES.map((a) => ({ value: a, label: a, title: `${t(lang, isField && p.view3d === 'volume' ? 'axisLook' : 'axisCut')}: ⟂ ${a}`, testid: `axis-${a}` }))} />
        <Menu icon="layers" label={t(lang, 'surfaceTip')} testid="surface-menu">
          {(close) => (
            <div className="menu-list" role="menu">
              {SURFACES.map((s) => (
                <button key={s.value} type="button" role="menuitemradio" aria-checked={p.surfaceMode === s.value}
                        className={p.surfaceMode === s.value ? 'on' : ''} data-testid={`surface-${s.value}`}
                        onClick={() => { p.setSurfaceMode(s.value); close(); }}>
                  <span className="menu-check">{p.surfaceMode === s.value && <Icon name="check" size={14} />}</span>{t(lang, s.key)}
                </button>
              ))}
            </div>
          )}
        </Menu>
      </div>
      <div className="tb-row sliders">
        {showPos && (
          <label className="slider">
            <span className="slider-label">{t(lang, 'cutPos')}</span>
            <input type="range" min={p.posRange[0]} max={p.posRange[1]} step={(p.posRange[1] - p.posRange[0]) / 200 || 1}
                   value={p.posValue} data-testid="cutpos" aria-label={t(lang, 'cutPos')}
                   onChange={(e) => p.setPos(Number(e.target.value))} />
            <span className="slider-val num">{p.posValue.toFixed(1)} mm</span>
          </label>
        )}
        {isField && (
          <label className="slider">
            <span className="slider-label">{t(lang, 'phaseTip')}</span>
            <input type="range" min={0} max={180} step={5} value={p.phase} data-testid="phase" aria-label={t(lang, 'phase')}
                   onChange={(e) => p.setPhase(Number(e.target.value))} />
            <span className="slider-val num">{p.phase}°</span>
          </label>
        )}
        {isField && (
          <IconButton icon={p.playing ? 'pause' : 'play'} label={t(lang, 'playTip')} active={p.playing} disabled={!p.canPlay}
                      onClick={() => p.setPlaying(!p.playing)} testid="play" className="play" />
        )}
        {p.extra && <div className="tb-tools">{p.extra}</div>}
      </div>
    </div>
  );
}
