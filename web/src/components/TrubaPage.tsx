// TRUBA plug-in: the form below becomes ONE shell line for a TRUBA login node (cluster/truba/run.sh,
// built and validated server-side by src/service/truba.py). Nothing is sent to the cluster from here.
// Field labels only (LABELS=field: the field3d model's data); freeform has no CAD solid and is not offered.
import { useEffect, useRef, useState } from 'react';
import { api, type TrubaConfig, type TrubaForm, type TrubaResult } from '../api';
import { int } from '../format';
import { Collapsible, NumField } from './ui';
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
  const nfld = (k: keyof TrubaForm, allowEmpty = false) => (v: number | null) =>
    set(k, (v === null ? (allowEmpty ? '' : 0) : v) as never);
  const data = f.action === 'all' || f.action === 'dataset';
  const train = f.action === 'all' || f.action === 'train';
  const famUsable = (fam: string) => fam !== 'freeform';
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
        <button type="button" className="btn ghost small" onClick={() => set('families', [...new Set([...f.families, ...list.filter(famUsable)])])}>{t(lang, 'all')}</button>
        <button type="button" className="btn ghost small" onClick={() => set('families', f.families.filter((x) => !list.includes(x)))}>×</button>
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
          <button className="btn primary small" disabled={!res} onClick={copy}>{copied ? `✓ ${t(lang, 'copied')}` : t(lang, 'copy')}</button>
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
          <div className="note">{t(lang, 'afterwards')}: <code>cd ~/{f.repo_dir} && bash cluster/truba/run.sh status LABELS=field{f.test ? ' TEST=1' : ''}</code></div>
        )}
      </div>

      <div className="gen">
        <div className="card">
          <div className="card-title">{t(lang, 'action')}</div>
          <div className="seg" role="group" aria-label={t(lang, 'action')}>
            {ACTIONS.map((a) => <button key={a} className={f.action === a ? 'on' : ''} onClick={() => set('action', a)}>{t(lang, `act_${a}` as Key)}</button>)}
          </div>
          {f.action !== 'setup' && <div className="note" data-testid="truba-labels">{t(lang, 'labelsKind')}: <strong>{t(lang, 'lab_field')}</strong></div>}
          <label className="check"><input type="checkbox" checked={f.dry_run} onChange={(e) => set('dry_run', e.target.checked)} /><span>{t(lang, 'dryRun')}</span></label>
          {f.action !== 'setup' && <label className="check"><input type="checkbox" checked={f.test} onChange={(e) => set('test', e.target.checked)} /><span>{t(lang, 'testRun')}</span></label>}

          <Collapsible title={t(lang, 'repoCard')} defaultOpen={false} className="advanced">
            <label>{t(lang, 'repoUrl')}<input type="text" value={f.repo} onChange={(e) => set('repo', e.target.value)} /></label>
            <div className="grid2">
              <label>{t(lang, 'branch')}<input type="text" value={f.branch} onChange={(e) => set('branch', e.target.value)} /></label>
              <label>{t(lang, 'repoDir')}<input type="text" value={f.repo_dir} onChange={(e) => set('repo_dir', e.target.value)} /></label>
            </div>
            <div className="note">{t(lang, 'privateRepo')}</div>
          </Collapsible>
        </div>

        {(data || f.action === 'train' || f.action === 'status') && (
          <div className="card">
            <div className="card-title">{t(lang, 'dataCard')}</div>
            {f.action !== 'status' && <>{famBox(cfg.families.train, t(lang, 'trainFamilies'))}{famBox(cfg.families.ood, t(lang, 'oodFamilies'))}</>}
            {train && <div className="note">{t(lang, 'trainOn')}</div>}
            <div className="grid2">
              {data && <NumField lang={lang} label={t(lang, 'perFamilyN')} value={f.n_per_family} min={1} allowEmpty onChange={nfld('n_per_family', true)} />}
              <NumField lang={lang} label={t(lang, 'meshSize')} value={f.mesh_size} min={0.03} max={0.5} step={0.01} onChange={nfld('mesh_size')} />
              <NumField lang={lang} label={t(lang, 'nModes')} value={f.n_modes} min={1} max={40} onChange={nfld('n_modes')} />
              <>
                <NumField lang={lang} label={t(lang, 'labelOrder')} value={f.label_order} min={2} max={4} onChange={nfld('label_order')} />
                <NumField lang={lang} label={t(lang, 'modelOrder')} value={f.model_order} min={1} max={3} onChange={nfld('model_order')} />
                <NumField lang={lang} label={t(lang, 'minFillet')} value={f.min_fillet} min={0} max={0.3} step={0.01} allowEmpty onChange={nfld('min_fillet', true)} />
              </>
              <label>TAG<input type="text" placeholder={res?.tag ?? ''} value={f.tag} onChange={(e) => set('tag', e.target.value)} /></label>
            </div>
            {data && <label className="check"><input type="checkbox" checked={f.adapt} onChange={(e) => set('adapt', e.target.checked)} /><span>{t(lang, 'adaptOn')}</span></label>}
            {data && (
              <Collapsible title={t(lang, 'advanced')} defaultOpen={false} className="advanced">
                <div className="grid2">
                  {data && <>
                    <NumField lang={lang} label={t(lang, 'threads')} value={f.threads} min={1} max={128} onChange={nfld('threads')} />
                    <NumField lang={lang} label={t(lang, 'maxElements')} value={f.max_elements} min={1000} step={1000} onChange={nfld('max_elements')} />
                    <NumField lang={lang} label={t(lang, 'maxhFactor')} value={f.maxh_factor} min={1} max={10} step={0.5} onChange={nfld('maxh_factor')} />
                    <NumField lang={lang} label={t(lang, 'modelMaxElements')} value={f.model_max_elements} min={100} step={1000} onChange={nfld('model_max_elements')} />
                    <NumField lang={lang} label={t(lang, 'tolF')} value={f.tol_f} min={1e-9} step={1e-7} disabled={!f.adapt} onChange={nfld('tol_f')} />
                    <NumField lang={lang} label={t(lang, 'tolQ')} value={f.tol_q} min={1e-6} step={1e-4} disabled={!f.adapt} onChange={nfld('tol_q')} />
                    <NumField lang={lang} label={t(lang, 'maxNdof')} value={f.max_ndof} min={10000} step={100000} onChange={nfld('max_ndof')} />
                  </>}
                  {data && <>
                    <NumField lang={lang} label={t(lang, 'deformProb')} value={f.deform_prob} min={0} max={1} step={0.1} onChange={nfld('deform_prob')} />
                    <NumField lang={lang} label={t(lang, 'deformMax')} value={f.deform_max} min={0} max={0.9} step={0.05} onChange={nfld('deform_max')} />
                    <label>{t(lang, 'sampling')}<select value={f.sampling} onChange={(e) => set('sampling', e.target.value as TrubaForm['sampling'])}><option value="sobol">sobol</option><option value="random">random</option></select></label>
                    <NumField lang={lang} label={t(lang, 'seed')} value={f.seed} min={0} onChange={nfld('seed')} />
                  </>}
                </div>
                {data && <label className="check"><input type="checkbox" checked={f.round_edges} onChange={(e) => set('round_edges', e.target.checked)} /><span>{t(lang, 'roundEdges')}</span></label>}
                {data && <label className="check"><input type="checkbox" checked={f.heal} onChange={(e) => set('heal', e.target.checked)} /><span>{t(lang, 'healOn')}</span></label>}
              </Collapsible>
            )}
            {data && (
              <Collapsible title={t(lang, 'slurmCard')} defaultOpen={false} className="advanced">
                <div className="grid2">
                  <label>{t(lang, 'partition')}<input type="text" value={f.partition} onChange={(e) => set('partition', e.target.value)} /></label>
                  <NumField lang={lang} label={t(lang, 'cpus')} value={f.cpus} min={1} step={56} onChange={nfld('cpus')} />
                  <label>{t(lang, 'wallTime')}<input type="text" value={f.time} onChange={(e) => set('time', e.target.value)} /></label>
                  <label>{t(lang, 'account')}<input type="text" value={f.account} onChange={(e) => set('account', e.target.value)} /></label>
                  <NumField lang={lang} label={t(lang, 'maxParallel')} value={f.max_parallel} min={1} onChange={nfld('max_parallel')} />
                </div>
              </Collapsible>
            )}
          </div>
        )}

        {train && (
          <div className="card">
            <div className="card-title">{t(lang, 'trainCard')}</div>
            <div className="grid2">
              <label>{t(lang, 'expName')}<input type="text" value={f.exp} onChange={(e) => set('exp', e.target.value)} /></label>
              <label>{t(lang, 'modelSize')}<select value={f.model} onChange={(e) => set('model', e.target.value)}>{cfg.models.map((m) => <option key={m} value={m}>{m}</option>)}</select></label>
              <NumField lang={lang} label={t(lang, 'epochs')} value={f.epochs} min={1} onChange={nfld('epochs')} />
              <NumField lang={lang} label={t(lang, 'batchOpt')} value={f.batch} min={1} placeholder="2" allowEmpty onChange={nfld('batch', true)} />
              <NumField lang={lang} label={t(lang, 'lr')} value={f.lr} min={1e-7} step={1e-5} onChange={nfld('lr')} />
            </div>
            <Collapsible title={t(lang, 'advanced')} defaultOpen={false} className="advanced">
              <div className="grid2">
              <NumField lang={lang} label={t(lang, 'gpus')} value={f.gpus} min={1} max={16} onChange={nfld('gpus')} />
              <label>{t(lang, 'gpuPartition')}<input type="text" value={f.gpu_partition} onChange={(e) => set('gpu_partition', e.target.value)} /></label>
                <NumField lang={lang} label={t(lang, 'gpuCpus')} value={f.gpu_cpus} min={1} onChange={nfld('gpu_cpus')} />
                <label>{t(lang, 'gpuTime')}<input type="text" value={f.gpu_time} onChange={(e) => set('gpu_time', e.target.value)} /></label>
              <NumField lang={lang} label={t(lang, 'loaderWorkers')} value={f.num_workers} min={0} placeholder="14" allowEmpty onChange={nfld('num_workers', true)} />
              </div>
              <label>{t(lang, 'cacheDir')}<input type="text" value={f.cache_dir} onChange={(e) => set('cache_dir', e.target.value)} /></label>
              <label>{t(lang, 'trainExtra')}<input type="text" placeholder="training.patience=30 model.num_field_modes=8" value={extra} onChange={(e) => setExtra(e.target.value)} /></label>
            </Collapsible>
          </div>
        )}
      </div>
    </div>
  );
}
