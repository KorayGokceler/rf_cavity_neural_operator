// TRUBA plug-in: the form below becomes ONE shell line for a TRUBA login node (cluster/truba/run.sh,
// built and validated server-side by src/service/truba.py). Nothing is sent to the cluster from here.
import { useEffect, useRef, useState, type ChangeEvent } from 'react';
import { api, type TrubaConfig, type TrubaForm, type TrubaResult } from '../api';
import { int } from '../format';
import { t, type Key, type Lang } from '../i18n';

const ACTIONS = ['all', 'dataset', 'train', 'status', 'setup'] as const;

interface Props { lang: Lang }

export default function TrubaPage({ lang }: Props) {
  const [cfg, setCfg] = useState<TrubaConfig | null>(null);
  const [f, setF] = useState<TrubaForm | null>(null);
  const [res, setRes] = useState<TrubaResult | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [extra, setExtra] = useState('');
  const pre = useRef<HTMLPreElement | null>(null);

  useEffect(() => {
    api.trubaConfig().then((c) => { setCfg(c); setF(c.form); }).catch((e) => setErr(e.message));
  }, []);
  useEffect(() => {
    if (!f) return;
    let live = true;
    const h = setTimeout(() => {
      api.trubaCommand({ ...f, train_extra: extra.split(/\s+/).filter(Boolean) })
        .then((r) => { if (live) { setRes(r); setErr(null); setCopied(false); } })
        .catch((e) => { if (live) { setRes(null); setErr(e.message); } });
    }, 200);
    return () => { live = false; clearTimeout(h); };
  }, [f, extra]);

  if (!f || !cfg) return <div className="page">{err ? <div className="banner critical"><span className="icon">✕</span>{err}</div> : null}</div>;

  const set = <K extends keyof TrubaForm>(k: K, v: TrubaForm[K]) => setF({ ...f, [k]: v });
  const num = (k: keyof TrubaForm, allowEmpty = false) => (e: ChangeEvent<HTMLInputElement>) =>
    set(k, (e.target.value === '' && allowEmpty ? '' : Number(e.target.value)) as never);
  const field = f.labels === 'field';
  const data = f.action === 'all' || f.action === 'dataset';
  const train = f.action === 'all' || f.action === 'train';
  const famUsable = (fam: string) => !(field && fam === 'freeform');
  const toggle = (fam: string) => set('families', f.families.includes(fam) ? f.families.filter((x) => x !== fam) : [...f.families, fam]);

  const copy = async () => {
    if (!res) return;
    try { await navigator.clipboard.writeText(res.command); setCopied(true); } catch {
      const r = document.createRange();                      // no clipboard API (http): select the text
      if (pre.current) { r.selectNodeContents(pre.current); window.getSelection()?.removeAllRanges(); window.getSelection()?.addRange(r); }
    }
  };

  const famBox = (list: string[], label: string) => (
    <fieldset className="fams">
      <legend>{label}
        <button type="button" className="ghost small" onClick={() => set('families', [...new Set([...f.families, ...list.filter(famUsable)])])}>{t(lang, 'all')}</button>
        <button type="button" className="ghost small" onClick={() => set('families', f.families.filter((x) => !list.includes(x)))}>×</button>
      </legend>
      {list.map((fam) => (
        <label key={fam} className="check">
          <input type="checkbox" disabled={!famUsable(fam)} checked={famUsable(fam) && f.families.includes(fam)} onChange={() => toggle(fam)} />
          <span>{fam}</span>
        </label>
      ))}
    </fieldset>
  );

  return (
    <div className="page">
      {err && <div className="banner critical"><span className="icon">✕</span>{err}</div>}
      <div className="card truba-out">
        <div className="card-title">{t(lang, 'trubaTitle')}
          <button className="primary small" disabled={!res} onClick={copy}>{copied ? `✓ ${t(lang, 'copied')}` : t(lang, 'copy')}</button>
        </div>
        <div className="note">{t(lang, 'trubaIntro')}</div>
        <pre ref={pre} className="cmd" aria-label={t(lang, 'command')}>{res?.command ?? '…'}</pre>
        {res && (
          <div className="note">
            {(data || f.action === 'status' || train) && <>{t(lang, 'outDir')}: <code>/arf/scratch/$USER/rfcav3d/{res.tag}/</code> · </>}
            {data && <>{int(res.estimate.geometries)} {t(lang, 'estGeoms')} · ≈ {int(res.estimate.core_hours)} {t(lang, 'estCore')} · ≈ {res.estimate.disk_gb} {t(lang, 'estDisk')} · {res.estimate.workers_per_job} {t(lang, 'estWorkers')}</>}
          </div>
        )}
        {res?.notes.map((n) => <div key={n} className="banner warning"><span className="icon">!</span>{t(lang, `note_${n}` as Key)}</div>)}
        {res && f.action !== 'status' && f.action !== 'setup' && (
          <div className="note">{t(lang, 'afterwards')}: <code>cd ~/{f.repo_dir} && bash cluster/truba/run.sh status{field ? ' LABELS=field' : ''}{f.test ? ' TEST=1' : ''}</code></div>
        )}
      </div>

      <div className="gen">
        <div className="card">
          <div className="card-title">{t(lang, 'action')}</div>
          <div className="seg" role="group" aria-label={t(lang, 'action')}>
            {ACTIONS.map((a) => <button key={a} className={f.action === a ? 'on' : ''} onClick={() => set('action', a)}>{t(lang, `act_${a}` as Key)}</button>)}
          </div>
          {f.action !== 'setup' && (
            <label>{t(lang, 'labelsKind')}
              <select value={f.labels} onChange={(e) => {
                const labels = e.target.value as TrubaForm['labels'];
                setF({ ...f, labels, exp: f.exp === 'field_base' && labels === 'n0' ? 'base' : f.exp === 'base' && labels === 'field' ? 'field_base' : f.exp,
                       families: labels === 'field' ? f.families.filter((x) => x !== 'freeform') : f.families });
              }}>
                <option value="field">{t(lang, 'lab_field')}</option>
                <option value="n0">{t(lang, 'lab_n0')}</option>
              </select>
            </label>
          )}
          <label className="check"><input type="checkbox" checked={f.dry_run} onChange={(e) => set('dry_run', e.target.checked)} /><span>{t(lang, 'dryRun')}</span></label>
          {f.action !== 'setup' && <label className="check"><input type="checkbox" checked={f.test} onChange={(e) => set('test', e.target.checked)} /><span>{t(lang, 'testRun')}</span></label>}

          <div className="card-title">{t(lang, 'repoCard')}</div>
          <label>{t(lang, 'repoUrl')}<input type="text" value={f.repo} onChange={(e) => set('repo', e.target.value)} /></label>
          <div className="grid2">
            <label>{t(lang, 'branch')}<input type="text" value={f.branch} onChange={(e) => set('branch', e.target.value)} /></label>
            <label>{t(lang, 'repoDir')}<input type="text" value={f.repo_dir} onChange={(e) => set('repo_dir', e.target.value)} /></label>
          </div>
          <div className="note">{t(lang, 'privateRepo')}</div>
        </div>

        {(data || f.action === 'train' || f.action === 'status') && (
          <div className="card">
            <div className="card-title">{t(lang, 'dataCard')}</div>
            {f.action !== 'status' && <>{famBox(cfg.families.train, t(lang, 'trainFamilies'))}{famBox(cfg.families.ood, t(lang, 'oodFamilies'))}</>}
            {train && <div className="note">{t(lang, 'trainOn')}</div>}
            <div className="grid2">
              {data && <label>{t(lang, 'perFamilyN')}<input type="number" min={1} value={f.n_per_family} onChange={num('n_per_family', true)} /></label>}
              <label>{t(lang, 'meshSize')}<input type="number" step={0.01} min={0.03} max={0.5} value={f.mesh_size} onChange={num('mesh_size')} /></label>
              <label>{t(lang, 'nModes')}<input type="number" min={1} max={40} value={f.n_modes} onChange={num('n_modes')} /></label>
              {field && <>
                <label>{t(lang, 'labelOrder')}<input type="number" min={2} max={4} value={f.label_order} onChange={num('label_order')} /></label>
                <label>{t(lang, 'modelOrder')}<input type="number" min={1} max={3} value={f.model_order} onChange={num('model_order')} /></label>
                <label>{t(lang, 'minFillet')}<input type="number" step={0.01} min={0} max={0.3} value={f.min_fillet} onChange={num('min_fillet', true)} /></label>
              </>}
              {data && field && <>
                <label>{t(lang, 'threads')}<input type="number" min={1} max={128} value={f.threads} onChange={num('threads')} /></label>
                <label>{t(lang, 'maxElements')}<input type="number" min={1000} step={1000} value={f.max_elements} onChange={num('max_elements')} /></label>
                <label>{t(lang, 'maxhFactor')}<input type="number" step={0.5} min={1} max={10} value={f.maxh_factor} onChange={num('maxh_factor')} /></label>
                <label>{t(lang, 'modelMaxElements')}<input type="number" min={100} step={1000} value={f.model_max_elements} onChange={num('model_max_elements')} /></label>
                <label>{t(lang, 'tolF')}<input type="number" step={1e-7} min={1e-9} value={f.tol_f} disabled={!f.adapt} onChange={num('tol_f')} /></label>
                <label>{t(lang, 'tolQ')}<input type="number" step={1e-4} min={1e-6} value={f.tol_q} disabled={!f.adapt} onChange={num('tol_q')} /></label>
                <label>{t(lang, 'maxNdof')}<input type="number" min={10000} step={100000} value={f.max_ndof} onChange={num('max_ndof')} /></label>
              </>}
              {data && <>
                <label>{t(lang, 'deformProb')}<input type="number" step={0.1} min={0} max={1} value={f.deform_prob} onChange={num('deform_prob')} /></label>
                <label>{t(lang, 'deformMax')}<input type="number" step={0.05} min={0} max={0.9} value={f.deform_max} onChange={num('deform_max')} /></label>
                <label>{t(lang, 'sampling')}<select value={f.sampling} onChange={(e) => set('sampling', e.target.value as TrubaForm['sampling'])}><option value="sobol">sobol</option><option value="random">random</option></select></label>
                <label>{t(lang, 'seed')}<input type="number" min={0} value={f.seed} onChange={num('seed')} /></label>
              </>}
              <label>TAG<input type="text" placeholder={res?.tag ?? ''} value={f.tag} onChange={(e) => set('tag', e.target.value)} /></label>
            </div>
            {data && field && <label className="check"><input type="checkbox" checked={f.adapt} onChange={(e) => set('adapt', e.target.checked)} /><span>{t(lang, 'adaptOn')}</span></label>}
            {data && <>
              <div className="card-title">{t(lang, 'slurmCard')}</div>
              <div className="grid2">
                <label>{t(lang, 'partition')}<input type="text" value={f.partition} onChange={(e) => set('partition', e.target.value)} /></label>
                <label>{t(lang, 'cpus')}<input type="number" min={1} step={56} value={f.cpus} onChange={num('cpus')} /></label>
                <label>{t(lang, 'wallTime')}<input type="text" value={f.time} onChange={(e) => set('time', e.target.value)} /></label>
                <label>{t(lang, 'account')}<input type="text" value={f.account} onChange={(e) => set('account', e.target.value)} /></label>
                <label>{t(lang, 'maxParallel')}<input type="number" min={1} value={f.max_parallel} onChange={num('max_parallel')} /></label>
              </div>
            </>}
          </div>
        )}

        {train && (
          <div className="card">
            <div className="card-title">{t(lang, 'trainCard')}</div>
            <div className="grid2">
              <label>{t(lang, 'expName')}<input type="text" value={f.exp} onChange={(e) => set('exp', e.target.value)} /></label>
              <label>{t(lang, 'modelSize')}<select value={f.model} onChange={(e) => set('model', e.target.value)}>{cfg.models.map((m) => <option key={m} value={m}>{m}</option>)}</select></label>
              <label>{t(lang, 'epochs')}<input type="number" min={1} value={f.epochs} onChange={num('epochs')} /></label>
              <label>{t(lang, 'batchOpt')}<input type="number" min={1} value={f.batch} placeholder={field ? '2' : '4'} onChange={num('batch', true)} /></label>
              <label>{t(lang, 'lr')}<input type="number" step={1e-5} min={1e-7} value={f.lr} onChange={num('lr')} /></label>
              <label>{t(lang, 'gpus')}<input type="number" min={1} max={16} value={f.gpus} onChange={num('gpus')} /></label>
              <label>{t(lang, 'gpuPartition')}<input type="text" value={f.gpu_partition} onChange={(e) => set('gpu_partition', e.target.value)} /></label>
              <label>{t(lang, 'gpuCpus')}<input type="number" min={1} value={f.gpu_cpus} onChange={num('gpu_cpus')} /></label>
              <label>{t(lang, 'gpuTime')}<input type="text" value={f.gpu_time} onChange={(e) => set('gpu_time', e.target.value)} /></label>
              <label>{t(lang, 'loaderWorkers')}<input type="number" min={0} value={f.num_workers} placeholder={field ? '14' : '6'} onChange={num('num_workers', true)} /></label>
            </div>
            {field && <label>{t(lang, 'cacheDir')}<input type="text" value={f.cache_dir} onChange={(e) => set('cache_dir', e.target.value)} /></label>}
            <label>{t(lang, 'trainExtra')}<input type="text" placeholder="training.patience=30 model.num_field_modes=8" value={extra} onChange={(e) => setExtra(e.target.value)} /></label>
          </div>
        )}
      </div>
    </div>
  );
}
