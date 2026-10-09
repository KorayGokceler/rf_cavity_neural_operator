// Dataset generation from the UI: one background job per family, written to RFCAV_GEN_ROOT (e.g. a
// Google Drive folder) in the TRUBA layout <root>/<TAG>/h5/<family>/…, pkl/<family>.pkl.
import { useEffect, useMemo, useState } from 'react';
import { api, type GenParams, type Job, type JobsConfig } from '../api';
import { int } from '../format';
import { t, type Lang } from '../i18n';

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

export default function GeneratePage({ lang, onShowDatasets }: Props) {
  const [cfg, setCfg] = useState<JobsConfig | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [err, setErr] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const [fams, setFams] = useState<string[]>(['elliptical']);
  const [p, setP] = useState<Omit<GenParams, 'families'>>({
    n_total: 100, mesh_size: 0.1, n_modes: 10, sampling: 'sobol', deform_prob: 0.5, deform_max: 0.5, seed: 0,
    shard_size: 256, workers: null, convert: true, lean: true, tag: null,
  });

  useEffect(() => { api.jobsConfig().then(setCfg).catch((e) => setErr(e.message)); }, []);
  const active = jobs.some((j) => j.status === 'queued' || j.status === 'running');
  useEffect(() => {
    if (!cfg?.enabled) return;
    let live = true;
    const tick = () => api.jobs().then((j) => live && setJobs(j)).catch(() => undefined);
    tick();
    const h = setInterval(tick, active ? 2000 : 8000);
    return () => { live = false; clearInterval(h); };
  }, [cfg, active]);

  const tag = p.tag || `${cfg?.field ?? 'E'}_ms${p.mesh_size}_k${p.n_modes}_v1`;
  const workers = p.workers ?? cfg?.cpu_count ?? 1;
  const est = useMemo(() => fams.reduce((a, f) => a + p.n_total * (SEC[f] ?? 1.5)
    * Math.pow(0.1 / p.mesh_size, 2.5) * (p.n_modes / 10) ** 0.5, 0) / workers, [fams, p, workers]);
  const set = <K extends keyof typeof p>(k: K, v: (typeof p)[K]) => setP({ ...p, [k]: v });
  const toggle = (f: string) => setFams(fams.includes(f) ? fams.filter((x) => x !== f) : [...fams, f]);

  const submit = async () => {
    setErr(null);
    try {
      await api.generate({ ...p, families: fams, tag: p.tag || null });
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
      {list.map((f) => (
        <label key={f} className="check">
          <input type="checkbox" checked={fams.includes(f)} onChange={() => toggle(f)} />
          <span>{f}</span><span className="muted">blok {cfg?.blocks[f] ?? '—'}</span>
        </label>
      ))}
    </fieldset>
  );

  return (
    <div className="page">
      {err && <div className="banner critical"><span className="icon">✕</span>{err}</div>}
      <div className="gen">
        <div className="card">
          <div className="card-title">{t(lang, 'genTitle')}</div>
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
            <label>{t(lang, 'workers')}<input type="number" min={1} max={256} placeholder={String(cfg?.cpu_count ?? '')} value={p.workers ?? ''} onChange={(e) => set('workers', e.target.value === '' ? null : Number(e.target.value))} /></label>
            <label>TAG<input type="text" placeholder={tag} value={p.tag ?? ''} onChange={(e) => set('tag', e.target.value || null)} /></label>
          </div>
          <label className="check"><input type="checkbox" checked={p.convert} onChange={(e) => set('convert', e.target.checked)} /><span>{t(lang, 'convertPkl')}</span></label>
          <label className="check"><input type="checkbox" checked={p.lean} disabled={!p.convert} onChange={(e) => set('lean', e.target.checked)} /><span>{t(lang, 'leanPkl')}</span></label>
          <div className="note">{t(lang, 'genTarget')}: <code>{cfg?.gen_root}/{tag}/h5/&lt;{t(lang, 'family')}&gt;/</code> · {cfg?.field}-field</div>
          <div className="note">{int(fams.length * p.n_total)} {t(lang, 'geometries')} · {workers} {t(lang, 'workers').toLowerCase()} · ≈ {fmtDur(est)} ({t(lang, 'roughEst')})</div>
          <button className="primary" disabled={!fams.length || !cfg} onClick={submit}>{t(lang, 'startGen')} ({fams.length})</button>
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
                    <strong>{j.params.family}</strong>
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
