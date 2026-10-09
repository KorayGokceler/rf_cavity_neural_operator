// Model training from the UI: one train.py run at a time on the server's GPU, checkpoints in
// RFCAV_RUNS_ROOT/<name>/ (Colab: Drive training_logs), live curves from progress.json.
import { useEffect, useMemo, useState } from 'react';
import { api, type DatasetRef, type ModelInfo, type RunDetail, type RunStatus, type RunSummary, type TrainConfig,
  type TrainParams } from '../api';
import { sci } from '../format';
import { t, type Key, type Lang } from '../i18n';
import { LineChart, type LineSeries } from './Charts';

const ST: Record<RunStatus, { key: Key; cls: string; icon: string }> = {
  running: { key: 'stRunning', cls: 'running', icon: '◷' },
  preparing: { key: 'stPreparing', cls: 'running', icon: '◷' },
  finished: { key: 'stFinished', cls: 'good', icon: '✓' },
  stopped: { key: 'stStopped', cls: 'stopped', icon: '■' },
  failed: { key: 'stFailed', cls: 'critical', icon: '✕' },
  checkpoint: { key: 'stCheckpoint', cls: 'muted', icon: '•' },
  empty: { key: 'stCheckpoint', cls: 'muted', icon: '•' },
};
const live = (s?: RunStatus) => s === 'running' || s === 'preparing';
const pct = (v: number) => `${(100 * v).toFixed(v < 0.01 ? 2 : 1)} %`;

function stamp() {
  const d = new Date(), p = (n: number) => String(n).padStart(2, '0');
  return `${String(d.getFullYear()).slice(2)}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}`;
}

interface Props { lang: Lang; model: ModelInfo | null; onModel: () => void }

export default function TrainPage({ lang, model, onModel }: Props) {
  const [cfg, setCfg] = useState<TrainConfig | null>(null);
  const [datasets, setDatasets] = useState<DatasetRef[]>([]);
  const [runs, setRuns] = useState<RunSummary[]>([]);
  const [sel, setSel] = useState<string | null>(null);
  const [run, setRun] = useState<RunDetail | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [showLog, setShowLog] = useState(false);
  const [more, setMore] = useState<number | null>(null);
  const [p, setP] = useState<TrainParams>({
    name: `small_${stamp()}`, datasets: [], preset: 'small', epochs: 100, batch_size: 4, lr: 2e-4, n_modes: 6,
    patience: 50, qoi_weight: 0, cache_operators: null,
  });
  const set = <K extends keyof TrainParams>(k: K, v: TrainParams[K]) => setP({ ...p, [k]: v });

  useEffect(() => {
    api.trainConfig().then(setCfg).catch((e) => setErr(e.message));
    api.datasets().then(setDatasets).catch(() => undefined);
  }, []);
  const anyLive = runs.some((r) => live(r.status));
  useEffect(() => {
    if (!cfg?.enabled) return;
    let on = true;
    const tick = () => api.trainRuns().then((r) => {
      if (!on) return;
      setRuns(r);
      setSel((s) => s ?? r[0]?.name ?? null);
    }).catch(() => undefined);
    tick();
    const h = setInterval(tick, anyLive ? 3000 : 10000);
    return () => { on = false; clearInterval(h); };
  }, [cfg, anyLive]);
  const selLive = live(runs.find((r) => r.name === sel)?.status);
  useEffect(() => {
    if (!sel) { setRun(null); return; }
    let on = true;
    const tick = () => api.trainRun(sel).then((r) => on && setRun(r)).catch(() => undefined);
    tick();
    const h = selLive ? setInterval(tick, 3000) : undefined;
    return () => { on = false; if (h) clearInterval(h); };
  }, [sel, selLive]);

  const act = async (fn: () => Promise<unknown>) => {
    setErr(null);
    try {
      await fn();
      setRuns(await api.trainRuns());
      if (sel) setRun(await api.trainRun(sel));
    } catch (e) { setErr((e as Error).message); }
  };
  const start = () => act(async () => {
    const r = await api.trainStart(p);
    setSel(r.name);
    setP({ ...p, name: `${p.preset}_${stamp()}` });
  });

  const curves = useMemo(() => {
    const h = run?.history ?? [];
    const ser = (key: string, label: string, cls: 's1' | 's2'): LineSeries =>
      ({ label, cls, pts: h.filter((e) => key in e).map((e) => [e.epoch, e[key]] as [number, number]) });
    const tr = t(lang, 'train'), va = t(lang, 'val');
    return {
      loss: [ser('train/loss', tr, 's1'), ser('val/loss', va, 's2')],
      field: [ser('train/field_rel_l2', tr, 's1'), ser('val/field_rel_l2', va, 's2')],
      freq: [ser('train/freq_rel_err', tr, 's1'), ser('val/freq_rel_err', va, 's2')],
      lr: [ser('lr-AdamW', t(lang, 'lr'), 's1')],
    };
  }, [run, lang]);

  if (cfg && !cfg.enabled) return <div className="page"><div className="card">{t(lang, 'trainOff')}</div></div>;
  const busy = !!cfg && runs.some((r) => live(r.status));
  const inUse = model?.run && run && model.run === run.name;
  const st = run ? ST[run.status] : null;
  const prog = run?.max_epochs ? Math.min(1, (run.epoch + (run.n_batches ? (run.batch ?? 0) / run.n_batches : 0)) / run.max_epochs) : 0;

  return (
    <div className="page">
      {err && <div className="banner critical"><span className="icon">✕</span>{err}</div>}
      {cfg && !cfg.gpu && <div className="banner warning"><span className="icon">!</span>{t(lang, 'noGpu')}</div>}
      <div className="train">
        <div className="train-side">
          <div className="card">
            <div className="card-title">{t(lang, 'newRun')}<span className="muted">{cfg?.gpu ?? 'CPU'}</span></div>
            <label>{t(lang, 'runName')}<input type="text" value={p.name} onChange={(e) => set('name', e.target.value)} /></label>
            <div>
              <div className="label">{t(lang, 'trainData')}</div>
              <div className="ds-list">
                {datasets.map((d) => (
                  <label key={d.id} className="check">
                    <input type="checkbox" checked={p.datasets.includes(d.id)}
                           onChange={() => set('datasets', p.datasets.includes(d.id) ? p.datasets.filter((x) => x !== d.id) : [...p.datasets, d.id])} />
                    <span>{d.name}</span><span className="muted">{d.kind}</span>
                  </label>
                ))}
                {!datasets.length && <span className="muted">{t(lang, 'noDatasets')}</span>}
              </div>
              <div className="note">{t(lang, 'trainDataNote')}</div>
            </div>
            <div>
              <div className="label">{t(lang, 'modelSize')}</div>
              <div className="presets seg">
                {cfg && Object.entries(cfg.presets).map(([k, v]) => (
                  <button key={k} className={p.preset === k ? 'on' : ''} title={`embed ${v.embed_dim} · ${v.n_layers} layers · m = ${v.n_basis}`}
                          onClick={() => setP({ ...p, preset: k, name: p.name.replace(/^[a-z]+_/, `${k}_`) })}>
                    <strong>{k}</strong><span className="muted">{v.params_m < 0.1 ? v.params_m.toFixed(2) : v.params_m.toFixed(1)} M</span>
                  </button>
                ))}
              </div>
            </div>
            <div className="grid2">
              <label>{t(lang, 'epochs')}<input type="number" min={1} value={p.epochs} onChange={(e) => set('epochs', Number(e.target.value))} /></label>
              <label>{t(lang, 'batchSize')}<input type="number" min={1} value={p.batch_size} onChange={(e) => set('batch_size', Number(e.target.value))} /></label>
              <label>{t(lang, 'lr')}<input type="number" step={1e-5} min={0} value={p.lr} onChange={(e) => set('lr', Number(e.target.value))} /></label>
              <label>{t(lang, 'ritzModes')}<input type="number" min={1} max={30} value={p.n_modes} onChange={(e) => set('n_modes', Number(e.target.value))} /></label>
              <label>{t(lang, 'patience')}<input type="number" min={1} value={p.patience} onChange={(e) => set('patience', Number(e.target.value))} /></label>
              <label>{t(lang, 'qoiWeight')}<input type="number" step={0.05} min={0} value={p.qoi_weight} onChange={(e) => set('qoi_weight', Number(e.target.value))} /></label>
              <label>{t(lang, 'cacheOps')}
                <select value={p.cache_operators === null ? 'auto' : p.cache_operators ? 'on' : 'off'}
                        onChange={(e) => set('cache_operators', e.target.value === 'auto' ? null : e.target.value === 'on')}>
                  <option value="auto">{t(lang, 'auto')}</option><option value="on">{t(lang, 'on')}</option><option value="off">{t(lang, 'off')}</option>
                </select>
              </label>
            </div>
            <button className="primary" disabled={!cfg || busy || !p.datasets.length || !p.name} onClick={start}>{t(lang, 'startTrain')}</button>
            <div className="note">{t(lang, 'colabNote')}</div>
          </div>
          <div className="card">
            <div className="card-title">{t(lang, 'runs')}</div>
            <div className="path-note">{cfg?.runs_root}</div>
            {!runs.length && <div className="muted">{t(lang, 'noRuns')}</div>}
            <ul className="runs">
              {runs.map((r) => {
                const s = ST[r.status];
                const f = r.max_epochs ? Math.min(1, r.epoch / r.max_epochs) : 0;
                return (
                  <li key={r.name}>
                    <button className={`run ${sel === r.name ? 'on' : ''}`} onClick={() => setSel(r.name)}>
                      <span className="run-head">
                        <span className={`status ${s.cls}`}>{s.icon}</span><strong>{r.name}</strong>
                        {model?.run === r.name && <span className="badge">{t(lang, 'inUse')}</span>}
                        <span className="muted">{r.epoch}{r.max_epochs ? `/${r.max_epochs}` : ''} · {r.best_score != null ? pct(r.best_score) : '—'}</span>
                      </span>
                      {r.max_epochs ? <span className="progress"><span style={{ width: `${f * 100}%` }} className={s.cls} /></span> : null}
                    </button>
                  </li>
                );
              })}
            </ul>
          </div>
        </div>

        <div className="train-main">
          {run && st && (
            <div className="card">
              <div className="card-title">
                <span><span className={`status ${st.cls}`}>{st.icon}</span> <strong>{run.name}</strong> · {t(lang, st.key)}{run.stage && live(run.status) && run.status === 'preparing' ? ` — ${run.stage}` : ''}</span>
                {inUse && <span className="badge">{t(lang, 'inUse')}</span>}
              </div>
              {run.max_epochs ? <div className="progress"><span style={{ width: `${prog * 100}%` }} className={st.cls} /></div> : null}
              <dl className="kv">
                <dt>{t(lang, 'epochs')}</dt><dd>{run.epoch}{run.max_epochs ? ` / ${run.max_epochs}` : ''}{run.n_batches && live(run.status) ? ` · batch ${run.batch}/${run.n_batches}` : ''}</dd>
                <dt>{t(lang, 'modelSize')}</dt><dd>{run.preset ?? '—'}{run.n_params ? ` · ${(run.n_params / 1e6).toFixed(2)} M` : ''}</dd>
                <dt>{t(lang, 'bestVal')}</dt><dd>{run.best_score != null ? `${pct(run.best_score)} (val/field_rel_l2)` : '—'}</dd>
                {run.params && <><dt>{t(lang, 'sources')}</dt><dd>{run.params.sources.join(', ')}</dd></>}
                {run.error && <><dt>{t(lang, 'stFailed')}</dt><dd className="critical-text">{run.error}</dd></>}
              </dl>
              {run.test && (
                <><div className="sub-title">{t(lang, 'testRes')}</div>
                <dl className="kv">
                  <dt>field rel L2</dt><dd>{pct(run.test['test/field_rel_l2'])}</dd>
                  {'test/freq_rel_err' in run.test && <><dt>freq rel err</dt><dd>{pct(run.test['test/freq_rel_err'])}</dd></>}
                  {'test/freq_mae_ghz' in run.test && <><dt>freq MAE</dt><dd>{sci(run.test['test/freq_mae_ghz'] * 1e3)} MHz</dd></>}
                </dl></>
              )}
              <div className="actions">
                {live(run.status) && <button onClick={() => act(api.trainStop)}>{t(lang, 'stop')}</button>}
                {!live(run.status) && run.resumable && (
                  <>
                    <label className="inline">{t(lang, 'toEpoch')}<input type="number" min={1} placeholder={String(run.max_epochs ?? '')}
                           value={more ?? ''} onChange={(e) => setMore(e.target.value === '' ? null : Number(e.target.value))} /></label>
                    <button disabled={busy} onClick={() => act(() => api.trainResume(run.name, more))}>{t(lang, 'resume')}</button>
                  </>
                )}
                {run.n_ckpt > 0 && !inUse && (
                  <button className="primary" onClick={() => act(async () => { await api.loadModel(run.name); onModel(); })}>{t(lang, 'useModel')}</button>
                )}
                {(run.log.length > 0) && <button className="ghost" onClick={() => setShowLog(!showLog)}>log</button>}
              </div>
              {showLog && <pre className="log">{run.log.join('\n')}</pre>}
            </div>
          )}
          {run && run.history.length > 0 && (
            <div className="curves">
              <LineChart title={t(lang, 'curveField')} series={curves.field} log fmt={pct} xLabel="epoch" />
              <LineChart title={t(lang, 'curveFreq')} series={curves.freq} log fmt={pct} xLabel="epoch" />
              <LineChart title={t(lang, 'curveLoss')} series={curves.loss} fmt={(v) => sci(v, 2)} xLabel="epoch" />
              <LineChart title={t(lang, 'curveLr')} series={curves.lr} log fmt={(v) => sci(v, 1)} xLabel="epoch" />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
