// Dataset browser: registered H5 / PKL datasets → family overview, frequency / size histograms,
// a sortable, filterable geometry table; "open" sends a geometry (with its FE modes) to the workspace.
import { useEffect, useMemo, useState } from 'react';
import { api, type DatasetRef, type DatasetRow, type DatasetStats } from '../api';
import { t, type Lang } from '../i18n';
import { int } from '../format';
import { BarList, Histogram } from './Charts';

interface Props { lang: Lang; onOpen: (did: string, sid: number) => void; busy: boolean }

const PAGE = 50;
type SortKey = 'id' | 'family' | 'n_edges' | 'n_nodes' | 'betti1' | 'f0';

export default function DatasetPage({ lang, onOpen, busy }: Props) {
  const [list, setList] = useState<DatasetRef[] | null>(null);
  const [did, setDid] = useState<string>('');
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [family, setFamily] = useState('');
  const [sort, setSort] = useState<SortKey>('id');
  const [desc, setDesc] = useState(false);
  const [offset, setOffset] = useState(0);
  const [rows, setRows] = useState<{ total: number; items: DatasetRow[] } | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => { api.datasets().then((l) => { setList(l); if (l.length) setDid(l[0].id); }).catch((e) => setErr(e.message)); }, []);
  useEffect(() => {
    if (!did) return;
    setStats(null); setFamily(''); setOffset(0);
    api.datasetStats(did).then(setStats).catch((e) => setErr(e.message));
  }, [did]);
  useEffect(() => {
    if (!did) return;
    api.datasetItems(did, { family, offset, limit: PAGE, sort, desc }).then(setRows).catch((e) => setErr(e.message));
  }, [did, family, offset, sort, desc]);

  const fams = useMemo(() => stats ? Object.entries(stats.families).map(([label, v]) => ({ label, value: v.count })) : [], [stats]);
  const f0 = useMemo(() => !stats ? [] : family ? stats.families[family]?.f0 ?? [] : Object.values(stats.families).flatMap((v) => v.f0), [stats, family]);
  const ne = useMemo(() => !stats ? [] : family ? stats.families[family]?.n_edges ?? [] : Object.values(stats.families).flatMap((v) => v.n_edges), [stats, family]);
  const nModes = Math.min(stats?.n_modes ?? 0, 6);

  const th = (k: SortKey, label: string, num = false) => (
    <th className={num ? 'num' : ''} aria-sort={sort === k ? (desc ? 'descending' : 'ascending') : 'none'}>
      <button className="th" onClick={() => { if (sort === k) setDesc(!desc); else { setSort(k); setDesc(false); } setOffset(0); }}>
        {label}{sort === k ? (desc ? ' ↓' : ' ↑') : ''}
      </button>
    </th>
  );

  if (list && !list.length) {
    return <div className="page"><div className="card">{t(lang, 'noDatasets')}</div></div>;
  }
  return (
    <div className="page">
      {err && <div className="banner critical"><span className="icon">✕</span>{err}</div>}
      <div className="filters">
        <label>{t(lang, 'dataset')}
          <select value={did} onChange={(e) => setDid(e.target.value)}>
            {list?.map((d) => <option key={d.id} value={d.id}>{d.name} · {d.kind}</option>)}
          </select>
        </label>
        <label>{t(lang, 'family')}
          <select value={family} onChange={(e) => { setFamily(e.target.value); setOffset(0); }}>
            <option value="">{t(lang, 'all')}</option>
            {fams.map((f) => <option key={f.label} value={f.label}>{f.label}</option>)}
          </select>
        </label>
        {stats && <span className="muted">{int(stats.total)} {t(lang, 'geometries')} · {stats.fields.join(', ')}-field · {stats.n_modes} {t(lang, 'modesStored')}</span>}
      </div>

      <div className="overview">
        <BarList data={fams} title={t(lang, 'perFamily')} picked={family}
                 onPick={(l) => { setFamily(family === l ? '' : l); setOffset(0); }} />
        <Histogram values={f0} title={t(lang, 'f0Hist')} unit="GHz" />
        <Histogram values={ne} title={t(lang, 'edgesHist')} unit="" />
      </div>

      <div className="card">
        <table className="modes data">
          <thead>
            <tr>
              {th('id', 'id')}{th('family', t(lang, 'family'))}{th('n_nodes', t(lang, 'nodes'), true)}{th('n_edges', 'N0', true)}
              {th('betti1', 'b1', true)}{th('f0', 'f₀ [GHz]', true)}
              {Array.from({ length: Math.max(0, nModes - 1) }, (_, i) => <th key={i} className="num">f{i + 1}</th>)}
              <th />
            </tr>
          </thead>
          <tbody>
            {rows?.items.map((r) => (
              <tr key={r.id} onDoubleClick={() => onOpen(did, r.id)}>
                <td className="num">{r.id}</td>
                <td>{r.family}{r.deformed ? <span className="tag" title={t(lang, 'deformed')}>~</span> : null}</td>
                <td className="num">{int(r.n_nodes)}</td>
                <td className="num">{int(r.n_edges)}</td>
                <td className="num">{r.betti1}</td>
                {Array.from({ length: Math.max(nModes, 1) }, (_, i) =>
                  <td key={i} className="num">{r.f_GHz[i] !== undefined ? r.f_GHz[i].toFixed(4) : '—'}</td>)}
                <td><button disabled={busy} onClick={() => onOpen(did, r.id)}>{t(lang, 'open')}</button></td>
              </tr>
            ))}
          </tbody>
        </table>
        {rows && (
          <div className="pager">
            <button disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>‹</button>
            <span className="muted">{rows.total ? offset + 1 : 0}–{Math.min(offset + PAGE, rows.total)} / {int(rows.total)}</span>
            <button disabled={offset + PAGE >= rows.total} onClick={() => setOffset(offset + PAGE)}>›</button>
          </div>
        )}
      </div>
    </div>
  );
}
