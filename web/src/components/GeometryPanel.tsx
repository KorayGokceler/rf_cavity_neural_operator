import { useState } from 'react';
import type { Geometry, Info } from '../api';
import { t, type Lang } from '../i18n';
import { int, sci } from '../format';

interface Props {
  lang: Lang;
  info: Info | null;
  geometry: Geometry | null;
  busy: boolean;
  onUpload: (file: File, unit: string, meshSize: number, nCells: number) => void;
  onSample: (family: string, id: number | null, meshSize: number) => void;
  onPredict: () => void;
}

export default function GeometryPanel({ lang, info, geometry, busy, onUpload, onSample, onPredict }: Props) {
  const [tab, setTab] = useState<'sample' | 'upload'>('sample');
  const [file, setFile] = useState<File | null>(null);
  const [unit, setUnit] = useState('mm');
  const [meshSize, setMeshSize] = useState(0.1);
  const [nCells, setNCells] = useState(1);
  const [family, setFamily] = useState('elliptical');
  const [sid, setSid] = useState('');
  const accept = info ? [...info.limits.cad, ...info.limits.mesh].join(',') : '';
  const chk = geometry?.check;

  return (
    <aside className="panel">
      <h2>{t(lang, 'geometry')}</h2>
      <div className="tabs" role="tablist">
        <button role="tab" aria-selected={tab === 'sample'} className={tab === 'sample' ? 'on' : ''}
                onClick={() => setTab('sample')}>{t(lang, 'sample')}</button>
        <button role="tab" aria-selected={tab === 'upload'} className={tab === 'upload' ? 'on' : ''}
                onClick={() => setTab('upload')}>{t(lang, 'upload')}</button>
      </div>

      {tab === 'sample' ? (
        <form onSubmit={(e) => { e.preventDefault(); onSample(family, sid === '' ? null : Number(sid), meshSize); }}>
          <label>{t(lang, 'family')}
            <select value={family} onChange={(e) => setFamily(e.target.value)}>
              <optgroup label={t(lang, 'trainFamilies')}>
                {info?.families.train.map((f) => <option key={f} value={f}>{f}</option>)}
              </optgroup>
              <optgroup label={t(lang, 'oodFamilies')}>
                {info?.families.ood.map((f) => <option key={f} value={f}>{f}</option>)}
              </optgroup>
            </select>
          </label>
          <label>{t(lang, 'sampleId')}
            <input type="number" min={0} value={sid} onChange={(e) => setSid(e.target.value)} placeholder="—" />
          </label>
          <label>{t(lang, 'meshSize')}
            <input type="number" step={0.01} min={0.05} max={0.5} value={meshSize}
                   onChange={(e) => setMeshSize(Number(e.target.value))} />
          </label>
          <button type="submit" disabled={busy}>{t(lang, 'build')}</button>
        </form>
      ) : (
        <form onSubmit={(e) => { e.preventDefault(); if (file) onUpload(file, unit, meshSize, nCells); }}>
          <label>{t(lang, 'file')}
            <input type="file" accept={accept} onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          </label>
          <div className="row2">
            <label>{t(lang, 'unit')}
              <select value={unit} onChange={(e) => setUnit(e.target.value)}>
                <option value="mm">mm</option><option value="cm">cm</option><option value="m">m</option>
              </select>
            </label>
            <label>{t(lang, 'nCells')}
              <input type="number" min={1} max={30} value={nCells} onChange={(e) => setNCells(Number(e.target.value))} />
            </label>
          </div>
          <label>{t(lang, 'meshSize')}
            <input type="number" step={0.01} min={0.05} max={0.5} value={meshSize}
                   onChange={(e) => setMeshSize(Number(e.target.value))} />
          </label>
          <button type="submit" disabled={busy || !file}>{t(lang, 'load')}</button>
          {info && <div className="note">≤ {info.limits.max_upload_mb} MB · {info.limits.cad.join(' ')} · {info.limits.mesh.join(' ')}</div>}
        </form>
      )}

      {chk && (
        <section className="checks">
          <h3>{t(lang, 'checks')}</h3>
          <ul>
            <li className={chk.ok ? 'good' : 'critical'}>
              <span className="icon">{chk.ok ? '✓' : '✕'}</span>
              {chk.ok ? t(lang, 'closed') : chk.problems.join('; ')}
            </li>
            {chk.betti1 !== undefined && <li><span className="icon">◦</span>{t(lang, 'handles')}: {chk.betti1}</li>}
            <li><span className="icon">◦</span>{int(chk.n_tets)} {t(lang, 'tets')}, {int(chk.n_nodes)} {t(lang, 'nodes')}</li>
            {chk.volume_cm3 !== undefined && <li><span className="icon">◦</span>{t(lang, 'volume')}: {sci(chk.volume_cm3)} cm³</li>}
            {geometry?.source && 'ood' in geometry.source && geometry.source.ood === true &&
              <li className="warning"><span className="icon">!</span>OOD</li>}
          </ul>
          <button className="primary" disabled={busy || !chk.ok} onClick={onPredict}>
            {busy ? t(lang, 'working') : geometry?.truth ? t(lang, 'compare') : t(lang, 'predict')}
          </button>
        </section>
      )}
    </aside>
  );
}
