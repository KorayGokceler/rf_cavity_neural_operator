// Dataset browser: registered H5 datasets → family overview, frequency / size histograms,
// a sortable, filterable geometry table; "open" sends a geometry (with its FE modes) to the workspace.
// Every dataset is a field-label H5 dataset (kind 'field'): its geometries can be predicted and compared.
import { useEffect, useMemo, useState } from 'react';
import { api, type DatasetQuality, type DatasetRef, type DatasetRow, type DatasetStats, type Job, type RefPlan, type RefResult } from '../api';
import { t, type Key, type Lang } from '../i18n';
import { int } from '../format';
import { BarList, Histogram } from './Charts';
import { Collapsible, NumField, numOk } from './ui';
import { useToast } from './Toasts';

interface Props { lang: Lang; onOpen: (did: string, sid: number) => Promise<unknown> | void }

const PAGE = 50;
type SortKey = 'id' | 'family' | 'n_dof' | 'betti1' | 'f0';

const sciF = (v: number | null | undefined) => (v === null || v === undefined || !Number.isFinite(v) ? '—' : v.toExponential(1));

/** p4 reference check of the selected dataset: start button + options, the job, and the results table. */
function ReferenceCard({ lang, did, name }: { lang: Lang; did: string; name: string }) {
  const toast = useToast();
  const [per, setPer] = useState<number | null>(1);
  const [order, setOrder] = useState<number | null>(4);
  const [nd, setNd] = useState<number | null>(700000);
  const [cross, setCross] = useState(false);
  const [plan, setPlan] = useState<RefPlan | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [res, setRes] = useState<RefResult | null>(null);
  const specs = { per: { min: 1, max: 20, integer: true }, order: { min: 3, max: 6, integer: true },
                  nd: { min: 10000, max: 20000000, integer: true } };
  const ok = numOk(per, specs.per) && numOk(order, specs.order) && numOk(nd, specs.nd);

  useEffect(() => {
    if (!ok || !nd) return;
    const h = setTimeout(() => api.referencePlan(nd).then(setPlan).catch(() => setPlan(null)), 250);
    return () => clearTimeout(h);
  }, [nd, ok]);
  // the newest reference job of this dataset; polled while it runs
  const active = job?.status === 'queued' || job?.status === 'running';
  useEffect(() => {
    let live = true;
    setJob(null); setRes(null);
    const tick = () => api.jobs().then((l) => {
      if (!live) return;
      const mine = l.find((j) => j.params.kind === 'reference' && j.params.dataset === name) ?? null;
      setJob((prev) => (prev && mine && prev.id === mine.id && prev.status === mine.status && prev.progress.done === mine.progress.done && prev.stage === mine.stage ? prev : mine));
    }).catch(() => undefined);
    tick();
    const h = setInterval(tick, active ? 2500 : 8000);
    return () => { live = false; clearInterval(h); };
  }, [name, active]);
  const jid = job?.id;
  useEffect(() => {
    if (!jid) return;
    let live = true;
    api.referenceResult(jid).then((r) => { if (live) setRes(r); }).catch(() => { if (live) setRes(null); });
    return () => { live = false; };
  }, [jid, job?.status, job?.progress.done]);

  const start = async () => {
    try {
      const j = await api.startReference({ dataset: did, per_family: per ?? 1, order: order ?? 4, max_ndof: nd ?? 700000, cross_check: cross });
      setJob(j); setRes(null);
      toast('info', `${t(lang, 'toastJobStarted')}: ${t(lang, 'refKind')}`);
    } catch (e) { toast('error', (e as Error).message); }
  };
  const rows = useMemo(() => [...(res?.samples ?? [])].sort((a, b) => a.family.localeCompare(b.family) || a.sample - b.sample), [res]);
  const pct = job && job.progress.total ? (100 * job.progress.done) / job.progress.total : 0;
  const cls = job?.status === 'done' ? 'good' : job?.status === 'failed' ? 'critical' : job?.status === 'cancelled' ? 'muted' : 'running';
  return (
    <div className="card pad ref-card" data-testid="ref-card">
      <div className="card-title">{t(lang, 'verifyTitle')}</div>
      <div className="note">{t(lang, 'verifyNote')}</div>
      <Collapsible title={t(lang, 'advanced')} defaultOpen={false} className="advanced">
        <div className="grid2">
          <NumField lang={lang} label={t(lang, 'refPerFamily')} value={per} onChange={setPer} {...specs.per} testid="ref-per" />
          <NumField lang={lang} label={t(lang, 'refOrder')} value={order} onChange={setOrder} {...specs.order} testid="ref-order" />
          <NumField lang={lang} label={t(lang, 'refMaxNdof')} value={nd} onChange={setNd} step={10000} {...specs.nd} testid="ref-ndof" />
        </div>
        <label className="check"><input type="checkbox" checked={cross} onChange={(e) => setCross(e.target.checked)} /><span>{t(lang, 'refCross')}</span></label>
      </Collapsible>
      {plan?.warnings.includes('memory') &&
        <div className="banner warning" data-testid="ref-mem-warn"><span className="icon">!</span>
          {t(lang, 'verifyMemWarn').replace('{need}', String(plan.mem_need_gb)).replace('{have}', String(plan.mem_gb))}</div>}
      <div className="ref-actions">
        <button type="button" className="btn primary" disabled={!ok || !did || active} onClick={start} data-testid="ref-start">{t(lang, 'verifyP4')}</button>
        {job && <span className="muted">
          <span className={`status ${cls}`}>{job.status === 'done' ? '✓' : job.status === 'failed' ? '✕' : job.status === 'cancelled' ? '–' : '◷'}</span>{' '}
          {job.status} · {int(job.progress.done)}/{int(job.progress.total)}
        </span>}
        {active && <button type="button" className="btn ghost small" onClick={() => api.cancelJob(job!.id)}>{t(lang, 'cancel')}</button>}
      </div>
      {job && <div className="progress"><span style={{ width: `${pct}%` }} className={cls} /></div>}
      {job?.error && <div className="banner critical" role="alert"><span className="icon">✕</span>{t(lang, 'refJobFail')}: {job.error}</div>}
      {!job && <div className="muted">{t(lang, 'refNoJob')}</div>}
      {job && !rows.length && active && <div className="muted">{t(lang, 'refWaiting')}</div>}
      {rows.length > 0 && (
        <div className="ref-table" data-testid="ref-table">
          <div className="group-title">{t(lang, 'refResults')} · {rows.length}{res?.summary?.n_skipped ? ` · ${t(lang, 'refSkippedN')} ${res.summary.n_skipped}` : ''}</div>
          <table className="modes data">
            <thead><tr>
              <th>{t(lang, 'family')}</th><th className="num">{t(lang, 'refSample')}</th><th className="num">{t(lang, 'refTets')}</th>
              <th className="num">{t(lang, 'refEst')}</th><th className="num">{t(lang, 'refMax')}</th><th className="num">{t(lang, 'refMedian')}</th>
              <th className="num" title={t(lang, 'refEstTrueTip')}>{t(lang, 'refEstTrue')}</th><th className="num">{t(lang, 'refTime')}</th><th>{t(lang, 'refSkipped')}</th>
            </tr></thead>
            <tbody>
              {rows.map((r) => (
                <tr key={`${r.file}-${r.sample}`}>
                  <td>{r.family}</td><td className="num">{r.sample}</td><td className="num">{int(r.n_tets)}</td>
                  <td className="num">{sciF(r.est_rel_f)}</td><td className="num">{sciF(r.max_rel_err)}</td><td className="num">{sciF(r.median_rel_err)}</td>
                  <td className="num">{r.est_over_true === null ? '—' : r.est_over_true.toFixed(2)}</td>
                  <td className="num">{r.t_ref_s === null ? '—' : r.t_ref_s.toFixed(0)}</td>
                  <td>{r.skipped ? <span className="status warning">{r.skipped}</span> : ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default function DatasetPage({ lang, onOpen }: Props) {
  const toast = useToast();
  const [busy, setBusy] = useState(false);
  const open = (did: string, sid: number) => { setBusy(true); Promise.resolve(onOpen(did, sid)).finally(() => setBusy(false)); };
  const [list, setList] = useState<DatasetRef[] | null>(null);
  const [did, setDid] = useState<string>('');
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [family, setFamily] = useState('');
  const [sort, setSort] = useState<SortKey>('id');
  const [desc, setDesc] = useState(false);
  const [offset, setOffset] = useState(0);
  const [rows, setRows] = useState<{ total: number; items: DatasetRow[] } | null>(null);
  const [quality, setQuality] = useState<DatasetQuality | null>(null);
  const setErr = (m: string) => toast('error', m);

  useEffect(() => { api.datasets().then((l) => { setList(l); if (l.length) setDid(l[0].id); }).catch((e) => setErr(e.message)); }, []);
  useEffect(() => {
    if (!did) return;
    setStats(null); setFamily(''); setOffset(0); setQuality(null);
    api.datasetStats(did).then(setStats).catch((e) => setErr(e.message));
    api.datasetQuality(did).then(setQuality).catch((e) => setErr(e.message));
  }, [did]);
  useEffect(() => {
    if (!did) return;
    api.datasetItems(did, { family, offset, limit: PAGE, sort, desc }).then(setRows).catch((e) => setErr(e.message));
  }, [did, family, offset, sort, desc]);

  const fams = useMemo(() => stats ? Object.entries(stats.families).map(([label, v]) => ({ label, value: v.count })) : [], [stats]);
  const f0 = useMemo(() => !stats ? [] : family ? stats.families[family]?.f0 ?? [] : Object.values(stats.families).flatMap((v) => v.f0), [stats, family]);
  const ne = useMemo(() => !stats ? [] : family ? stats.families[family]?.n_dof ?? [] : Object.values(stats.families).flatMap((v) => v.n_dof), [stats, family]);
  const nModes = Math.min(stats?.n_modes ?? 0, 6);
  const sci = sciF;
  const p2 = quality?.kind === 'p2';

  const th = (k: SortKey, label: string, num = false) => (
    <th className={num ? 'num' : ''} aria-sort={sort === k ? (desc ? 'descending' : 'ascending') : 'none'}>
      <button className="th" onClick={() => { if (sort === k) setDesc(!desc); else { setSort(k); setDesc(false); } setOffset(0); }}>
        {label}{sort === k ? (desc ? ' ↓' : ' ↑') : ''}
      </button>
    </th>
  );

  if (list && !list.length) {
    return <div className="page"><div className="card pad">{t(lang, 'noDatasets')}</div></div>;
  }
  return (
    <div className="page">
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
        {stats && <span className="muted">{int(stats.total)} {t(lang, 'geometries')} · {stats.n_modes} {t(lang, 'modesStored')}</span>}
      </div>

      {quality && (
        <div className="card pad quality" data-testid="ds-quality">
          <span className="group-title">{t(lang, 'labelQuality')}</span>
          <span className="tag" data-testid="ds-kind">{t(lang, quality.kind === 'p2' ? 'lq_p2' : 'lq_p3')}</span>
          {p2 && <>
            <span title={t(lang, 'lqBadTip')} className={quality.n_bad ? 'status warning' : 'status good'}>
              {t(lang, 'lqBad')}: <strong>{quality.n_bad ?? 0}</strong> / {int(quality.n)}
              {quality.bad_share === null || quality.bad_share === undefined ? '' : ` (${(100 * quality.bad_share).toFixed(1)} %)`}
            </span>
            <span>{t(lang, 'lqEst')}: <strong>{sci(quality.est_median)}</strong> / <strong>{sci(quality.est_max)}</strong></span>
            <span>{t(lang, 'lqTets')}: <strong>{quality.tets_min === null || quality.tets_min === undefined ? '—' : int(quality.tets_min)}–{quality.tets_max === null || quality.tets_max === undefined ? '—' : int(quality.tets_max)}</strong></span>
          </>}
        </div>
      )}

      <div className="overview">
        <BarList data={fams} title={t(lang, 'perFamily')} picked={family}
                 onPick={(l) => { setFamily(family === l ? '' : l); setOffset(0); }} />
        <Histogram values={f0} title={t(lang, 'f0Hist')} unit="GHz" />
        <Histogram values={ne} title={t(lang, 'dofHist')} unit="" />
      </div>

      <div className="card">
        <table className="modes data">
          <thead>
            <tr>
              {th('id', 'id')}{th('family', t(lang, 'family'))}
              <th className="num" title={t(lang, 'modelMeshTip')}>{t(lang, 'tetsModel')}</th>
              {!p2 && <th className="num" title={t(lang, 'labelMeshTip')}>{t(lang, 'tetsLabel')}</th>}
              {th('n_dof', 'DOF p2', true)}
              {!p2 && <th className="num" title={t(lang, 'adaptItersTip')}>{t(lang, 'adaptIters')}</th>}
              {!p2 && <th>{t(lang, 'adaptStop')}</th>}
              <th className="num" title={t(lang, 'estErrTip')}>{t(lang, 'estErr')}</th>
              {p2 && <th title={t(lang, 'labelOkTip')}>{t(lang, 'labelOk')}</th>}
              {p2 && <th className="num" title={t(lang, 'curvSafetyTip')}>{t(lang, 'curvSafety')}</th>}
              <th className="num" title={t(lang, 'edgesTip')}>{t(lang, 'edgesRS')}</th>
              {th('betti1', 'b1', true)}{th('f0', 'f₀ [GHz]', true)}
              {Array.from({ length: Math.max(0, nModes - 1) }, (_, i) => <th key={i} className="num">f{i + 1}</th>)}
              <th />
            </tr>
          </thead>
          <tbody>
            {rows?.items.map((r) => (
              <tr key={r.id} onDoubleClick={() => open(did, r.id)}>
                <td className="num">{r.id}</td>
                <td>{r.family}{r.deformed ? <span className="tag" title={t(lang, 'deformed')}>~</span> : null}</td>
                <td className="num">{int(r.n_tets)}</td>
                {!p2 && <td className="num">{r.field.n_tets_label === null ? '—' : int(r.field.n_tets_label)}</td>}
                <td className="num">{int(r.n_dof)}</td>
                {!p2 && <td className="num">{r.field.adapt_iters ?? '—'}</td>}
                {!p2 && <td><span className={`status ${r.field.adapt_stop === 'converged' ? 'good' : r.field.adapt_stop ? 'warning' : 'muted'}`}>{r.field.adapt_stop ? t(lang, `stop_${r.field.adapt_stop}` as Key) : t(lang, 'oneSolve')}</span></td>}
                <td className="num">{sci(r.field.est_rel_f)}</td>
                {p2 && <td><span className={`status ${r.field.label_ok === null ? 'muted' : r.field.label_ok ? 'good' : 'warning'}`}>{r.field.label_ok === null ? '—' : r.field.label_ok ? '✓ 1' : '! 0'}</span></td>}
                {p2 && <td className="num">{r.field.curvaturesafety_used ?? '—'}</td>}
                <td className="num">{r.field.rounded_edges ?? 0} / {r.field.sharp_edges ? <strong title={t(lang, 'sharpTip')}>{r.field.sharp_edges}</strong> : 0}</td>
                <td className="num">{r.betti1}</td>
                {Array.from({ length: Math.max(nModes, 1) }, (_, i) =>
                  <td key={i} className="num">{r.f_GHz[i] !== undefined ? r.f_GHz[i].toFixed(4) : '—'}</td>)}
                <td><button type="button" className="btn small" disabled={busy} onClick={() => open(did, r.id)} data-testid="open-item">{t(lang, 'open')}</button></td>
              </tr>
            ))}
          </tbody>
        </table>
        {rows && (
          <div className="pager">
            <button type="button" className="btn small" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}>‹</button>
            <span className="muted">{rows.total ? offset + 1 : 0}–{Math.min(offset + PAGE, rows.total)} / {int(rows.total)}</span>
            <button type="button" className="btn small" disabled={offset + PAGE >= rows.total} onClick={() => setOffset(offset + PAGE)}>›</button>
          </div>
        )}
      </div>

      {did && <ReferenceCard lang={lang} did={did} name={list?.find((d) => d.id === did)?.name ?? ''} />}
    </div>
  );
}
