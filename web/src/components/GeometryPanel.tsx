// Geometry panel of the workspace: example cavity or file upload, the checks, the predict button.
import { useState } from 'react';
import type { Geometry, Info } from '../api';
import { t, type Lang } from '../i18n';
import { int, sci } from '../format';
import Icon from './Icon';
import { NumField, Seg, numOk } from './ui';

/** Form state (lifted to the workspace: a dropped file uses the same unit / mesh size / cells as the form). */
export interface GeomForm { family: string; sid: number | null; unit: string; meshSize: number; nCells: number }

interface Props {
  lang: Lang;
  info: Info | null;
  geometry: Geometry | null;
  busy: boolean;
  form: GeomForm;
  setForm: (f: GeomForm) => void;
  onUpload: (file: File) => void;
  onSample: (family: string, id: number | null, meshSize: number) => void;
  onPredict: () => void;
  onClose: () => void;
  children?: React.ReactNode;           // sections below the form (recent geometries …)
  after?: React.ReactNode;              // below the checks (geometry info)
}

export default function GeometryPanel({ lang, info, geometry, busy, form, setForm, onUpload, onSample, onPredict,
                                        onClose, children, after }: Props) {
  const [tab, setTab] = useState<'sample' | 'upload'>('sample');
  const [file, setFile] = useState<File | null>(null);
  const accept = info ? [...info.limits.cad, ...info.limits.mesh].join(',') : '';
  const chk = geometry?.check;
  const meshOk = numOk(form.meshSize, { min: 0.05, max: 0.5 });
  const cellsOk = numOk(form.nCells, { min: 1, max: 30, integer: true });
  const idOk = numOk(form.sid, { min: 0, integer: true, allowEmpty: true });
  const set = (p: Partial<GeomForm>) => setForm({ ...form, ...p });

  return (
    <div className="geom-panel">
      <Seg value={tab} label={t(lang, 'geomSource')} onChange={setTab} className="full"
           options={[{ value: 'sample', icon: 'cube', label: t(lang, 'sample') },
                     { value: 'upload', icon: 'upload', label: t(lang, 'upload') }]} />

      {tab === 'sample' ? (
        <form onSubmit={(e) => { e.preventDefault(); if (meshOk && idOk) onSample(form.family, form.sid, form.meshSize); }}>
          <label className="field">
            <span className="field-label">{t(lang, 'family')}</span>
            <select value={form.family} onChange={(e) => set({ family: e.target.value })} data-testid="family">
              <optgroup label={t(lang, 'trainFamilies')}>
                {info?.families.train.map((f) => <option key={f} value={f}>{f}</option>)}
              </optgroup>
              <optgroup label={t(lang, 'oodFamilies')}>
                {info?.families.ood.map((f) => <option key={f} value={f}>{f}</option>)}
              </optgroup>
            </select>
          </label>
          <div className="row2">
            <NumField lang={lang} label={t(lang, 'sampleId')} value={form.sid} min={0} integer allowEmpty placeholder="—"
                      testid="sample-id" onChange={(v) => set({ sid: v })} />
            <NumField lang={lang} label={t(lang, 'meshSize')} value={form.meshSize} min={0.05} max={0.5} step={0.01}
                      onChange={(v) => set({ meshSize: v ?? 0 })} />
          </div>
          <button type="submit" className="btn" disabled={busy || !meshOk || !idOk}>{t(lang, 'build')}</button>
        </form>
      ) : (
        <form onSubmit={(e) => { e.preventDefault(); if (file && meshOk && cellsOk) onUpload(file); }}>
          <label className="dropzone">
            <Icon name="upload" size={20} />
            <span className="dz-main">{file ? file.name : t(lang, 'browse')}</span>
            <span className="dz-sub">{t(lang, 'dropHint')}</span>
            <input type="file" accept={accept} data-testid="file-input" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          </label>
          <div className="row2">
            <label className="field">
              <span className="field-label">{t(lang, 'unit')}</span>
              <select value={form.unit} onChange={(e) => set({ unit: e.target.value })}>
                <option value="mm">mm</option><option value="cm">cm</option><option value="m">m</option>
              </select>
            </label>
            <NumField lang={lang} label={t(lang, 'nCells')} value={form.nCells} min={1} max={30} integer
                      onChange={(v) => set({ nCells: v ?? 0 })} />
          </div>
          <NumField lang={lang} label={t(lang, 'meshSize')} value={form.meshSize} min={0.05} max={0.5} step={0.01}
                    onChange={(v) => set({ meshSize: v ?? 0 })} />
          <button type="submit" className="btn" disabled={busy || !file || !meshOk || !cellsOk}>{t(lang, 'load')}</button>
          {info && <div className="note">≤ {info.limits.max_upload_mb} MB · {info.limits.cad.join(' ')} · {info.limits.mesh.join(' ')}</div>}
        </form>
      )}

      {children}

      {chk && (
        <section className="checks panel-section" data-testid="checks">
          <div className="section-head">
            <h3>{t(lang, 'checks')}</h3>
            <button type="button" className="btn ghost small close" onClick={onClose} title={t(lang, 'closeGeomTip')}
                    data-testid="close-geometry"><Icon name="x" size={14} />{t(lang, 'closeGeom')}</button>
          </div>
          <ul>
            <li className={chk.ok ? 'good' : 'critical'}>
              <Icon name={chk.ok ? 'check' : 'x'} size={14} />
              <span>{chk.ok ? t(lang, 'closed') : chk.problems.join('; ')}</span>
            </li>
            {chk.betti1 !== undefined && <li><span className="dot" />{t(lang, 'handles')}: {chk.betti1}</li>}
            <li><span className="dot" />{int(chk.n_tets)} {t(lang, 'tets')}, {int(chk.n_nodes)} {t(lang, 'nodes')}</li>
            {chk.volume_cm3 !== undefined && <li><span className="dot" />{t(lang, 'volume')}: {sci(chk.volume_cm3)} cm³</li>}
            {geometry?.source && 'ood' in geometry.source && geometry.source.ood === true &&
              <li className="warning"><Icon name="warn" size={14} />OOD</li>}
          </ul>
          <button type="button" className="btn primary" disabled={busy || !chk.ok} onClick={onPredict} data-testid="predict">
            {busy ? t(lang, 'working') : geometry?.truth ? t(lang, 'compare') : t(lang, 'predict')}
          </button>
        </section>
      )}
      {after}
    </div>
  );
}
