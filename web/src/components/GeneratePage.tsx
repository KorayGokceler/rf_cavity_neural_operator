// Dataset generation from the UI: one background job per family, written to RFCAV_GEN_ROOT (e.g. a
// Google Drive folder) in the TRUBA layout <root>/<TAG>/h5/<family>/…
// Field labels only (p2 labels on the model mesh by default, docs/37; p3 adaptive labels projected onto p2, docs/29–30); workers ×
// NGSolve threads planned by the server from its cores and memory (/api/jobs/plan).
import { useEffect, useMemo, useRef, useState } from 'react';
import { api, type FieldPlan, type GenParams, type Job, type JobsConfig } from '../api';
import { int } from '../format';
import { t, type Key, type Lang } from '../i18n';
import { Collapsible, NumField, numOk } from './ui';
import { useToast } from './Toasts';

function fmtDur(s: number, lang: Lang) {
  if (!Number.isFinite(s)) return '—';
  if (s < 90) return `${Math.max(1, Math.round(s))} s`;
  if (s < 5400) return `${Math.round(s / 60)} ${t(lang, 'unitMin')}`;
  return `${(s / 3600).toFixed(1)} ${t(lang, 'unitHour')}`;
}

interface Props { lang: Lang; onShowDatasets: () => void }

type Form = Omit<GenParams, 'families'>;

/** TAG as the server derives it (src/service/jobs.field_tag_of = cluster/truba/config.sh). */
function defaultTag(p: Form) {
  const oneSolve = !p.adapt && p.label_order > p.model_order;     // p2 labels have no adaptive variant
  return `F_p${p.label_order}p${p.model_order}_ms${p.mesh_size}_k${p.n_modes}_mf${p.min_fillet}_v2`
    .replace('_ms0.1_', '_ms0.10_') + (oneSolve ? '_1s' : '') + (p.round_edges ? '' : '_sharp');
}

/** Label kind (docs/37): p2 = one solve in the model space on the model mesh; hf = adaptive p3 (test / CST sets);
 * hf1 = one p3 solve. */
type LabelKind = 'p2' | 'hf' | 'hf1';
const labelKind = (p: Form): LabelKind => (p.label_order <= p.model_order ? 'p2' : p.adapt ? 'hf' : 'hf1');

export default function GeneratePage({ lang, onShowDatasets }: Props) {
  const toast = useToast();
  const [cfg, setCfg] = useState<JobsConfig | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const setErr = (m: string | null) => { if (m) toast('error', m); };
  const lastStatus = useRef(new Map<string, Job['status']>());
  const [open, setOpen] = useState<string | null>(null);
  const [fams, setFams] = useState<string[]>(['elliptical']);
  const [p, setP] = useState<Form>({
    labels: 'field', n_total: 100, mesh_size: 0.1, n_modes: 10, sampling: 'sobol', deform_prob: 0.5, deform_max: 0.5,
    seed: 0, shard_size: 32, workers: null, tag: null,
    threads: null, label_order: 2, model_order: 2, min_fillet: 0.05, maxh_factor: 3, field_max_elements: 30000,
    adapt: true, tol_f: 1e-6, tol_f_p2: 1e-3, tol_q: 1e-3, model_max_elements: 20000, max_ndof: null, round_edges: true, heal: true,
    sample_timeout: 7200,
  });
  const [plan, setPlan] = useState<FieldPlan | null>(null);

  useEffect(() => {
    api.jobsConfig().then((c) => {
      setCfg(c);
      if (c.field) setP((q) => ({ ...q, ...c.field.defaults, threads: null }));   // the machine's label budget
    }).catch((e) => setErr(e.message));
  }, []);
  useEffect(() => {                                     // workers × threads for this machine (field labels)
    if (!cfg?.enabled) return;
    const h = setTimeout(() => api.jobsPlan({ max_ndof: p.max_ndof, threads: p.threads, workers: p.workers,
                                              label_order: p.label_order, model_order: p.model_order,
                                              model_max_elements: p.model_max_elements })
      .then(setPlan).catch((e) => setErr(e.message)), 250);
    return () => clearTimeout(h);
  }, [cfg, p.max_ndof, p.threads, p.workers, p.label_order, p.model_order, p.model_max_elements]);
  const active = jobs.some((j) => j.status === 'queued' || j.status === 'running');
  // a job that was queued / running and is now done / failed: one toast (the first poll only records the states)
  const notify = (list: Job[]) => {
    const seen = lastStatus.current;
    for (const j of list) {
      const prev = seen.get(j.id);
      if ((prev === 'queued' || prev === 'running') && (j.status === 'done' || j.status === 'failed')) {
        toast(j.status === 'done' ? 'ok' : 'error',
              `${t(lang, j.status === 'done' ? 'toastJobDone' : 'toastJobFail')}: ${j.params.family} · ${j.params.tag}${j.error ? ` — ${j.error}` : ''}`);
      }
      seen.set(j.id, j.status);
    }
  };
  useEffect(() => {
    if (!cfg?.enabled) return;
    let live = true;
    const tick = () => api.jobs().then((j) => { if (live) { setJobs(j); notify(j); } }).catch(() => undefined);
    tick();
    const h = setInterval(tick, active ? 2000 : 8000);
    return () => { live = false; clearInterval(h); };
  }, [cfg, active]);

  const tag = p.tag || defaultTag(p);
  const noCad = cfg?.field?.no_cad ?? ['freeform'];
  const chosen = fams.filter((f) => !noCad.includes(f));
  const workers = plan?.workers ?? 1;
  const kind = labelKind(p);
  const est = useMemo(() => {   // core-seconds per geometry (src/service/jobs.field_core_s) over all cores in use
    const core = labelKind(p) === 'p2'
      ? (cfg?.field.p2_core_s ?? 120) * (p.n_modes / 10) ** 0.3
      : (cfg?.field.core_s ?? 560) * (p.adapt ? cfg?.field.adapt_factor ?? 3 : 1)
        * (p.n_modes / 10) ** 0.3 * (p.label_order / 3) ** 3;
    return chosen.length * p.n_total * core / Math.max(1, (plan?.workers ?? 1) * (plan?.threads ?? 1));
  }, [chosen, p, plan, cfg]);
  const set = <K extends keyof Form>(k: K, v: Form[K]) => setP({ ...p, [k]: v });
  const toggle = (f: string) => setFams(fams.includes(f) ? fams.filter((x) => x !== f) : [...fams, f]);

  // numeric fields: bounds mirror the server's validation; the start button waits for valid values
  type NumKey = keyof Form & ('n_total' | 'mesh_size' | 'n_modes' | 'deform_prob' | 'deform_max' | 'seed' | 'shard_size'
    | 'workers' | 'tol_f' | 'tol_f_p2' | 'tol_q' | 'max_ndof' | 'model_max_elements' | 'label_order' | 'model_order' | 'min_fillet'
    | 'threads' | 'field_max_elements' | 'maxh_factor' | 'sample_timeout');
  interface Spec { min?: number; max?: number; step?: number; integer?: boolean; allowEmpty?: boolean }
  const SPEC: Record<NumKey, Spec> = {
    n_total: { min: 1, max: cfg?.max_total, integer: true }, mesh_size: { min: 0.05, max: 0.5, step: 0.01 },
    n_modes: { min: 1, max: 30, integer: true }, deform_prob: { min: 0, max: 1, step: 0.1 },
    deform_max: { min: 0, max: 0.9, step: 0.05 }, seed: { min: 0, integer: true },
    shard_size: { min: 16, max: 4096, integer: true }, workers: { min: 1, max: 256, integer: true, allowEmpty: true },
    tol_f: { min: 1e-9, step: 1e-7 }, tol_f_p2: { min: 1e-6, max: 0.1, step: 1e-4 }, tol_q: { min: 1e-6, step: 1e-4 },
    max_ndof: { min: 10000, step: 100000, integer: true, allowEmpty: true },
    model_max_elements: { min: 100, step: 1000, integer: true }, label_order: { min: 2, max: 5, integer: true },
    model_order: { min: 1, max: 4, integer: true }, min_fillet: { min: 0, max: 0.3, step: 0.01 },
    threads: { min: 1, max: 256, integer: true, allowEmpty: true }, field_max_elements: { min: 1000, step: 1000, integer: true },
    maxh_factor: { min: 1, max: 10, step: 0.5 }, sample_timeout: { min: 60, step: 600, integer: true },
  };
  const COMMON: NumKey[] = ['n_total', 'mesh_size', 'n_modes', 'deform_prob', 'deform_max', 'seed', 'shard_size', 'workers'];
  const FIELD: NumKey[] = ['tol_f', 'tol_f_p2', 'tol_q', 'max_ndof', 'model_max_elements', 'label_order', 'model_order', 'min_fillet',
                           'threads', 'field_max_elements', 'maxh_factor', 'sample_timeout'];
  const nf = (k: NumKey, label: string, placeholder?: string, disabled = false) => (
    <NumField lang={lang} label={label} value={p[k] as number | null} min={SPEC[k].min} max={SPEC[k].max}
              step={SPEC[k].step} integer={SPEC[k].integer} allowEmpty={SPEC[k].allowEmpty} placeholder={placeholder}
              disabled={disabled} testid={`gen-${k}`} onChange={(v) => setP({ ...p, [k]: v as never })} />
  );
  const setKind = (k: LabelKind) => setP({ ...p, label_order: k === 'p2' ? p.model_order : Math.max(3, p.model_order + 1),
                                            adapt: k !== 'hf1' });
  const formOk = [...COMMON, ...FIELD.filter((k) => kind === 'hf' || (k !== 'tol_f' && k !== 'tol_q'))
    .filter((k) => k !== 'tol_f_p2' || kind === 'p2')]
    .every((k) => numOk(p[k] as number | null, SPEC[k]));

  const submit = async () => {
    try {
      await api.generate({ ...p, families: chosen, tag: p.tag || null });
      const list = await api.jobs();
      setJobs(list); notify(list);
      toast('info', `${t(lang, 'toastJobStarted')}: ${chosen.length} ${t(lang, 'jobs').toLowerCase()}`);
    } catch (e) { setErr((e as Error).message); }
  };

  if (cfg && !cfg.enabled) {
    return <div className="page"><div className="card pad">{t(lang, 'genOff')}</div></div>;
  }
  const famBox = (list: string[], label: string) => (
    <fieldset className="fams">
      <legend>{label}
        <button type="button" className="btn ghost small" onClick={() => setFams([...new Set([...fams, ...list])])}>{t(lang, 'all')}</button>
        <button type="button" className="btn ghost small" onClick={() => setFams(fams.filter((f) => !list.includes(f)))}>×</button>
      </legend>
      {list.map((f) => {
        const off = noCad.includes(f);                      // no CAD solid: field labels need one
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
      <div className="gen">
        <div className="card">
          <div className="card-title">{t(lang, 'genTitle')}</div>
          <div className="note" data-testid="gen-labels">{t(lang, 'labelsKind')}: <strong>{t(lang, 'lab_field')}</strong></div>
          {cfg && <>{famBox(cfg.families.train, t(lang, 'trainFamilies'))}{famBox(cfg.families.ood, t(lang, 'oodFamilies'))}</>}
          <div className="group-title">{t(lang, 'basics')}</div>
          <div className="grid3">
            {nf('n_total', t(lang, 'perFamilyN'))}
            {nf('mesh_size', t(lang, 'meshSize'))}
            {nf('n_modes', t(lang, 'nModes'))}
          </div>
          <label>{t(lang, 'labelKind')}
            <select value={kind} onChange={(e) => setKind(e.target.value as LabelKind)} data-testid="gen-label-kind">
              <option value="p2">{t(lang, 'labelKindP2')}</option>
              <option value="hf">{t(lang, 'labelKindHf')}</option>
              <option value="hf1">{t(lang, 'labelKindHf1')}</option>
            </select>
          </label>
          <div className="note">{t(lang, kind === 'p2' ? 'labelKindP2Note' : 'labelKindHfNote')}</div>
          {kind === 'p2' && <div className="note" data-testid="gen-mesh-range">{t(lang, 'modelMeshRange')}: <strong>{t(lang, 'modelMeshRangeVal')}</strong> · <span title={t(lang, 'tolFP2Tip')}>{t(lang, 'tolFP2')}</span>: <strong>{p.tol_f_p2}</strong></div>}
          <Collapsible title={t(lang, 'advanced')} defaultOpen={false} className="advanced">
            <div className="grid2">
              <label>{t(lang, 'sampling')}<select value={p.sampling} onChange={(e) => set('sampling', e.target.value as 'sobol' | 'random')}><option value="sobol">sobol</option><option value="random">random</option></select></label>
              {nf('deform_prob', t(lang, 'deformProb'))}
              {nf('deform_max', t(lang, 'deformMax'))}
              {nf('seed', t(lang, 'seed'))}
              {nf('shard_size', t(lang, 'shardSize'))}
              {nf('workers', t(lang, 'fieldWorkers'), String(plan?.workers ?? ''))}
              <label>TAG<input type="text" placeholder={tag} value={p.tag ?? ''} onChange={(e) => set('tag', e.target.value || null)} /></label>
            </div>
            <div className="group-title">{t(lang, 'fieldCard')}</div>
              <div className="grid2">
                {nf('tol_f', t(lang, 'tolF'), undefined, kind !== 'hf')}
                {nf('tol_q', t(lang, 'tolQ'), undefined, kind !== 'hf')}
                {nf('tol_f_p2', t(lang, 'tolFP2'), undefined, kind !== 'p2')}
                {nf('max_ndof', t(lang, 'maxNdof'), String(cfg?.field.defaults.max_ndof ?? ''), kind === 'p2')}
                {nf('model_max_elements', t(lang, 'modelMaxElementsP3'), undefined, kind === 'p2')}
                {nf('label_order', t(lang, 'labelOrder'))}
                {nf('model_order', t(lang, 'modelOrder'))}
                {nf('min_fillet', t(lang, 'minFillet'))}
                {nf('threads', t(lang, 'threads'), String(plan?.threads ?? ''))}
                {nf('field_max_elements', t(lang, 'maxElements'))}
                {nf('maxh_factor', t(lang, 'maxhFactor'))}
                {nf('sample_timeout', t(lang, 'sampleTimeout'))}
              </div>
              <label className="check"><input type="checkbox" checked={p.round_edges} onChange={(e) => set('round_edges', e.target.checked)} /><span>{t(lang, 'roundEdges')}</span></label>
              <label className="check"><input type="checkbox" checked={p.heal} onChange={(e) => set('heal', e.target.checked)} /><span>{t(lang, 'healOn')}</span></label>
          </Collapsible>
          {plan && <div className="note">{t(lang, 'machine')}: {plan.cpus} {t(lang, 'cores')} · {plan.mem_gb ? `${plan.mem_gb} GB RAM` : 'RAM ?'} → <strong>{plan.workers} × {plan.threads}</strong> ({t(lang, 'workersThreads')}) · ≤ {plan.mem_per_worker_gb} GB / {t(lang, 'worker')} · {t(lang, 'total')} ≤ {plan.mem_total_gb} GB</div>}
          {plan?.warnings.map((w) => <div key={w} className="banner warning"><span className="icon">!</span>{t(lang, `plan_${w}` as Key)}</div>)}
          <div className="note">{t(lang, 'fieldH5Only')}</div>
          <div className="note">{t(lang, 'genTarget')}: <code>{cfg?.gen_root}/{tag}/h5/&lt;{t(lang, 'family')}&gt;/</code></div>
          <div className="note">{int(chosen.length * (Number(p.n_total) || 0))} {t(lang, 'geometries')} · {workers} × {plan?.threads ?? '…'} {t(lang, 'workers').toLowerCase()} · ≈ {fmtDur(est, lang)} ({t(lang, 'roughEst')})</div>
          {!formOk && <div className="field-error" role="alert">{t(lang, 'vFixForm')}</div>}
          <button type="button" className="btn primary" disabled={!chosen.length || !cfg || !formOk} onClick={submit} data-testid="gen-start">{t(lang, 'startGen')} ({chosen.length})</button>
        </div>

        <div className="card">
          <div className="card-title">{t(lang, 'jobs')}
            <button type="button" className="btn ghost small" onClick={async () => { await api.rescan(); onShowDatasets(); }}>{t(lang, 'toDatasets')} →</button>
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
                    <strong>{j.params.kind === 'reference' ? t(lang, 'refKind') : j.params.family}</strong>
                    <span className="tag">{j.params.kind === 'reference' ? `p${j.params.order} · ${j.params.threads} thr` : `field${j.params.adapt ? ' · adapt' : ''} · ${j.params.workers}×${j.params.threads}`}</span>
                    <span className="muted">{j.params.tag} · {int(j.progress.done)}/{int(j.progress.total)}{j.progress.failed ? ` · ${j.progress.failed} ✕` : ''} · {fmtDur(el, lang)}</span>
                    {(j.status === 'queued' || j.status === 'running') &&
                      <button type="button" className="btn ghost small" onClick={() => api.cancelJob(j.id).then(() => api.jobs().then(setJobs))}>{t(lang, 'cancel')}</button>}
                    <button type="button" className="btn ghost small" onClick={() => setOpen(open === j.id ? null : j.id)}>log</button>
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
