// Dataset generation from the UI: one background job per family, written to RFCAV_GEN_ROOT (e.g. a
// Google Drive folder) in the TRUBA layout <root>/<TAG>/h5/<family>/…, pkl/<family>.pkl.
// Labels: N0 (one process per core) or field (adaptive p3 labels, docs/29–30: H5 only; workers × NGSolve
// threads planned by the server from its cores and memory, /api/jobs/plan).
import { useEffect, useMemo, useState } from 'react';
import { api, type FieldPlan, type GenParams, type Job, type JobsConfig } from '../api';
import { int } from '../format';
import { t, type Key, type Lang } from '../i18n';

// rough single-core seconds per geometry at mesh 0.10, 10 modes (cluster/truba/README.md benchmark)
const SEC: Record<string, number> = {
  elliptical: 2.2, reentrant: 1.0, pillbox_pipes: 1.2, ridged_box: 1.0, composite: 0.9, freeform: 2.3,
  hwr: 1.8, spoke: 2.6, dtl: 3.9,
};

function fmtDur(s: number) {
  if (!Number.isFinite(s)) return '—';
  if (s < 90) return `${Math.max(1, Math.round(s))} s`;
  if (s < 5400) return `${Math.round(s / 60)} dk`;
  return `${(s / 3600).toFixed(1)} sa`;
}

interface Props { lang: Lang; onShowDatasets: () => void }

type Form = Omit<GenParams, 'families'>;

/** TAG as the server derives it (src/service/jobs.tag_of / field_tag_of = cluster/truba/config.sh). */
function defaultTag(p: Form) {
  if (p.labels === 'n0') return `E_ms${p.mesh_size}_k${p.n_modes}_v1`;
  return `F_p${p.label_order}p${p.model_order}_ms${p.mesh_size}_k${p.n_modes}_mf${p.min_fillet}_v1`
    .replace('_ms0.1_', '_ms0.10_') + (p.adapt ? '' : '_1s') + (p.round_edges ? '' : '_sharp');
}

export default function GeneratePage({ lang, onShowDatasets }: Props) {
  const [cfg, setCfg] = useState<JobsConfig | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [fams, setFams] = useState<string[]>(['elliptical']);
  const [p, setP] = useState<Form>({
    labels: 'n0', n_total: 100, mesh_size: 0.1, n_modes: 10, sampling: 'sobol', deform_prob: 0.5, deform_max: 0.5,
    seed: 0, shard_size: 256, workers: null, convert: true, lean: true, tag: null,
    threads: null, label_order: 3, model_order: 2, min_fillet: 0.05, maxh_factor: 3, field_max_elements: 30000,
    adapt: true, tol_f: 1e-6, tol_q: 1e-3, model_max_elements: 10000, max_ndof: null, round_edges: true, heal: true,
    sample_timeout: 7200,
  });
  const [plan, setPlan] = useState<FieldPlan | null>(null);
  const field = p.labels === 'field';

  useEffect(() => {
    api.jobsConfig().then((c) => {
      setCfg(c);
      if (c.field) setP((q) => ({ ...q, ...c.field.defaults, threads: null }));   // the machine's label budget
    }).catch((e) => setErr(e.message));
  }, []);
  useEffect(() => {                                     // workers × threads for this machine (field labels)
    if (!field || !cfg?.enabled) return;
    const h = setTimeout(() => api.jobsPlan({ max_ndof: p.max_ndof, threads: p.threads, workers: p.workers })
      .then(setPlan).catch((e) => setErr(e.message)), 250);
    return () => clearTimeout(h);
  }, [field, cfg, p.max_ndof, p.threads, p.workers]);
  const active = jobs.some((j) => j.status === 'queued' || j.status === 'running');
  useEffect(() => {
    if (!cfg?.enabled) return;
    let live = true;
    const tick = () => api.jobs().then((j) => live && setJobs(j)).catch(() => undefined);
    tick();
    const h = setInterval(tick, active ? 2000 : 8000);
    return () => { live = false; clearInterval(h); };
  }, [cfg, active]);

  const tag = p.tag || defaultTag(p);
  const noCad = cfg?.field?.no_cad ?? ['freeform'];
  const chosen = field ? fams.filter((f) => !noCad.includes(f)) : fams;
  const workers = field ? (plan?.workers ?? 1) : (p.workers ?? cfg?.cpu_count ?? 1);
  const est = useMemo(() => {
    if (field) {                                     // core-seconds per geometry (docs/29 §4) over all cores in use
      const core = (cfg?.field.core_s ?? 560) * (p.adapt ? cfg?.field.adapt_factor ?? 3 : 1)
        * (p.n_modes / 10) ** 0.3 * (p.label_order / 3) ** 3;
      return chosen.length * p.n_total * core / Math.max(1, (plan?.workers ?? 1) * (plan?.threads ?? 1));
    }
    return chosen.reduce((a, f) => a + p.n_total * (SEC[f] ?? 1.5)
      * Math.pow(0.1 / p.mesh_size, 2.5) * (p.n_modes / 10) ** 0.5, 0) / workers;
  }, [chosen, p, workers, plan, field, cfg]);
  const set = <K extends keyof Form>(k: K, v: Form[K]) => setP({ ...p, [k]: v });
  const num = (k: keyof Form, empty: number | null = null) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setP({ ...p, [k]: e.target.value === '' ? empty : Number(e.target.value) });
  const toggle = (f: string) => setFams(fams.includes(f) ? fams.filter((x) => x !== f) : [...fams, f]);
  const setLabels = (labels: Form['labels']) => setP({
    ...p, labels, tag: null,
    shard_size: labels === 'field' && p.shard_size === 256 ? 32 : labels === 'n0' && p.shard_size === 32 ? 256 : p.shard_size,
  });

  const submit = async () => {
    setErr(null);
    try {
      await api.generate({ ...p, families: chosen, tag: p.tag || null });
      setJobs(await api.jobs());
    } catch (e) { setErr((e as Error).message); }
  };

  if (cfg && !cfg.enabled) {
    return <div className="page"><div className="card">{t(lang, 'genOff')}</div></div>;
  }
  const famBox = (list: string[], label: string) => (
    <fieldset className="fams">
      <legend>{label}
        <button type="button" className="ghost small" onClick={() => setFams([...new Set([...fams, ...list])])}>{t(lang, 'all')}</button>
        <button type="button" className="ghost small" onClick={() => setFams(fams.filter((f) => !list.includes(f)))}>×</button>
      </legend>
      {list.map((f) => {
        const off = field && noCad.includes(f);             // no CAD solid: field labels need one
        return (
          <label key={f} className="check" title={off ? t(lang, 'noCadSolid') : undefined}>
            <input type="checkbox" checked={!off && fams.includes(f)} disabled={off} onChange={() => toggle(f)} />
            <span className={off ? 'muted' : ''}>{f}</span><span className="muted">blok {cfg?.blocks[f] ?? '—'}</span>
          </label>
        );
      })}
    </fieldset>
  );

  return (
    <div className="page">
      {err && <div className="banner critical"><span className="icon">✕</span>{err}</div>}
      <div className="gen">
        <div className="card">
          <div className="card-title">{t(lang, 'genTitle')}</div>
          <label>{t(lang, 'labelsKind')}
            <select value={p.labels} onChange={(e) => setLabels(e.target.value as Form['labels'])}>
              <option value="n0">{t(lang, 'lab_n0')}</option>
              <option value="field">{t(lang, 'lab_field')}</option>
            </select>
          </label>
          {cfg && <>{famBox(cfg.families.train, t(lang, 'trainFamilies'))}{famBox(cfg.families.ood, t(lang, 'oodFamilies'))}</>}
          <div className="grid2">
            <label>{t(lang, 'perFamilyN')}<input type="number" min={1} max={cfg?.max_total} value={p.n_total} onChange={(e) => set('n_total', Number(e.target.value))} /></label>
            <label>{t(lang, 'meshSize')}<input type="number" step={0.01} min={0.05} max={0.5} value={p.mesh_size} onChange={(e) => set('mesh_size', Number(e.target.value))} /></label>
            <label>{t(lang, 'nModes')}<input type="number" min={1} max={30} value={p.n_modes} onChange={(e) => set('n_modes', Number(e.target.value))} /></label>
            <label>{t(lang, 'sampling')}<select value={p.sampling} onChange={(e) => set('sampling', e.target.value as 'sobol' | 'random')}><option value="sobol">sobol</option><option value="random">random</option></select></label>
            <label>{t(lang, 'deformProb')}<input type="number" step={0.1} min={0} max={1} value={p.deform_prob} onChange={(e) => set('deform_prob', Number(e.target.value))} /></label>
            <label>{t(lang, 'deformMax')}<input type="number" step={0.05} min={0} max={0.9} value={p.deform_max} onChange={(e) => set('deform_max', Number(e.target.value))} /></label>
            <label>{t(lang, 'seed')}<input type="number" min={0} value={p.seed} onChange={(e) => set('seed', Number(e.target.value))} /></label>
            <label>{t(lang, 'shardSize')}<input type="number" min={16} max={4096} value={p.shard_size} onChange={(e) => set('shard_size', Number(e.target.value))} /></label>
            <label>{t(lang, field ? 'fieldWorkers' : 'workers')}<input type="number" min={1} max={256} placeholder={String(field ? plan?.workers ?? '' : cfg?.cpu_count ?? '')} value={p.workers ?? ''} onChange={num('workers')} /></label>
            <label>TAG<input type="text" placeholder={tag} value={p.tag ?? ''} onChange={(e) => set('tag', e.target.value || null)} /></label>
          </div>
          {field ? <>
            <div className="card-title">{t(lang, 'fieldCard')}</div>
            <label className="check"><input type="checkbox" checked={p.adapt} onChange={(e) => set('adapt', e.target.checked)} /><span>{t(lang, 'adaptOn')}</span></label>
            <div className="grid2">
              <label>{t(lang, 'tolF')}<input type="number" step={1e-7} min={1e-9} value={p.tol_f} disabled={!p.adapt} onChange={num('tol_f', 1e-6)} /></label>
              <label>{t(lang, 'tolQ')}<input type="number" step={1e-4} min={1e-6} value={p.tol_q} disabled={!p.adapt} onChange={num('tol_q', 1e-3)} /></label>
              <label>{t(lang, 'maxNdof')}<input type="number" min={10000} step={100000} placeholder={String(cfg?.field.defaults.max_ndof ?? '')} value={p.max_ndof ?? ''} onChange={num('max_ndof')} /></label>
              <label>{t(lang, 'modelMaxElements')}<input type="number" min={100} step={1000} value={p.model_max_elements} onChange={num('model_max_elements', 10000)} /></label>
              <label>{t(lang, 'labelOrder')}<input type="number" min={2} max={5} value={p.label_order} onChange={num('label_order', 3)} /></label>
              <label>{t(lang, 'modelOrder')}<input type="number" min={1} max={4} value={p.model_order} onChange={num('model_order', 2)} /></label>
              <label>{t(lang, 'minFillet')}<input type="number" step={0.01} min={0} max={0.3} value={p.min_fillet} onChange={num('min_fillet', 0.05)} /></label>
              <label>{t(lang, 'threads')}<input type="number" min={1} max={256} placeholder={String(plan?.threads ?? '')} value={p.threads ?? ''} onChange={num('threads')} /></label>
              <label>{t(lang, 'maxElements')}<input type="number" min={1000} step={1000} value={p.field_max_elements} onChange={num('field_max_elements', 30000)} /></label>
              <label>{t(lang, 'maxhFactor')}<input type="number" step={0.5} min={1} max={10} value={p.maxh_factor} onChange={num('maxh_factor', 3)} /></label>
              <label>{t(lang, 'sampleTimeout')}<input type="number" min={60} step={600} value={p.sample_timeout} onChange={num('sample_timeout', 7200)} /></label>
            </div>
            <label className="check"><input type="checkbox" checked={p.round_edges} onChange={(e) => set('round_edges', e.target.checked)} /><span>{t(lang, 'roundEdges')}</span></label>
            <label className="check"><input type="checkbox" checked={p.heal} onChange={(e) => set('heal', e.target.checked)} /><span>{t(lang, 'healOn')}</span></label>
            {plan && <div className="note">{t(lang, 'machine')}: {plan.cpus} {t(lang, 'cores')} · {plan.mem_gb ? `${plan.mem_gb} GB RAM` : 'RAM ?'} → <strong>{plan.workers} × {plan.threads}</strong> ({t(lang, 'workersThreads')}) · ≤ {plan.mem_per_worker_gb} GB / {t(lang, 'worker')} · {t(lang, 'total')} ≤ {plan.mem_total_gb} GB</div>}
            {plan?.warnings.map((w) => <div key={w} className="banner warning"><span className="icon">!</span>{t(lang, `plan_${w}` as Key)}</div>)}
            <div className="note">{t(lang, 'fieldH5Only')}</div>
          </> : <>
            <label className="check"><input type="checkbox" checked={p.convert} onChange={(e) => set('convert', e.target.checked)} /><span>{t(lang, 'convertPkl')}</span></label>
            <label className="check"><input type="checkbox" checked={p.lean} disabled={!p.convert} onChange={(e) => set('lean', e.target.checked)} /><span>{t(lang, 'leanPkl')}</span></label>
          </>}
          <div className="note">{t(lang, 'genTarget')}: <code>{cfg?.gen_root}/{tag}/h5/&lt;{t(lang, 'family')}&gt;/</code></div>
          <div className="note">{int(chosen.length * p.n_total)} {t(lang, 'geometries')} · {field ? `${workers} × ${plan?.threads ?? '…'}` : workers} {t(lang, 'workers').toLowerCase()} · ≈ {fmtDur(est)} ({t(lang, 'roughEst')})</div>
          <button className="primary" disabled={!chosen.length || !cfg} onClick={submit}>{t(lang, 'startGen')} ({chosen.length})</button>
        </div>

        <div className="card">
          <div className="card-title">{t(lang, 'jobs')}
            <button className="ghost small" onClick={async () => { await api.rescan(); onShowDatasets(); }}>{t(lang, 'toDatasets')} →</button>
          </div>
          {!jobs.length && <div className="muted">{t(lang, 'noJobs')}</div>}
          <ul className="jobs">
            {jobs.map((j) => {
              const pct = j.progress.total ? (100 * j.progress.done) / j.progress.total : 0;
              const el = (j.finished ?? Date.now() / 1000) - (j.started ?? j.created);
              const cls = j.status === 'done' ? 'good' : j.status === 'failed' ? 'critical' : j.status === 'cancelled' ? 'muted' : 'running';
              return (
                <li key={j.id}>
                  <div className="job-head">
                    <span className={`status ${cls}`}>{j.status === 'done' ? '✓' : j.status === 'failed' ? '✕' : j.status === 'cancelled' ? '–' : '◷'}</span>
                    <strong>{j.params.family}</strong>{j.params.labels === 'field' && <span className="tag">field{j.params.adapt ? ' · adapt' : ''} · {j.params.workers}×{j.params.threads}</span>}
                    <span className="muted">{j.params.tag} · {int(j.progress.done)}/{int(j.progress.total)}{j.progress.failed ? ` · ${j.progress.failed} ✕` : ''} · {fmtDur(el)}</span>
                    {(j.status === 'queued' || j.status === 'running') &&
                      <button className="ghost small" onClick={() => api.cancelJob(j.id).then(() => api.jobs().then(setJobs))}>{t(lang, 'cancel')}</button>}
                    <button className="ghost small" onClick={() => setOpen(open === j.id ? null : j.id)}>log</button>
                  </div>
                  <div className="progress"><span style={{ width: `${pct}%` }} className={cls} /></div>
                  <div className="muted small-text">{j.status === 'running' ? j.stage : j.status}{j.error ? ` — ${j.error}` : ''}</div>
                  {open === j.id && <pre className="log">{j.log.join('\n')}</pre>}
                </li>
              );
            })}
          </ul>
        </div>
      </div>
    </div>
  );
}
