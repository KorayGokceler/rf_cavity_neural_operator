import { api, type CmpRow, type ModeRow } from '../api';
import { QOI_DEFS, etaLabel, fmtF, sci } from '../format';
import { t, type Lang } from '../i18n';

interface Props {
  lang: Lang; row: ModeRow; source: string; cmp?: CmpRow | null; timings?: Record<string, number> | null;
  exportId?: string | null; label?: Record<string, number> | null;
}

const pct = (v: number | null | undefined) => (v === null || v === undefined || !Number.isFinite(v) ? '—' : `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)} %`);

export default function ModeDetails({ lang, row, source, cmp, timings, exportId, label }: Props) {
  const e = row.eta !== null && row.eta !== undefined ? etaLabel(lang, row.eta) : null;
  return (
    <div className="card">
      <div className="card-title">{t(lang, 'selected')}: {row.mode} · {source}</div>
      <div className="hero">
        <span className="hero-num">{fmtF(cmp ? cmp.f_model : row.f_GHz)}</span><span className="hero-unit">GHz</span>
        {cmp && <span className="muted">FE {fmtF(cmp.f_fe)} · Δ {pct(cmp.f_rel_err)}</span>}
      </div>
      {cmp ? (
        <table className="modes qoi-cmp">
          <thead><tr><th /><th>FE</th><th>model</th><th>Δ</th></tr></thead>
          <tbody>
            {QOI_DEFS.map((d) => {
              const q = cmp.qoi[d.key];
              const sc = d.scale ?? 1;
              return (
                <tr key={d.key}>
                  <td>{d.label}{d.unit && <span className="unit"> {d.unit}</span>}</td>
                  <td className="num">{sci(q?.fe === null || q?.fe === undefined ? null : q.fe * sc)}</td>
                  <td className="num">{sci(q?.model === null || q?.model === undefined ? null : q.model * sc)}</td>
                  <td className="num">{cmp.degenerate ? '≈' : pct(q?.rel_err)}</td>
                </tr>
              );
            })}
            <tr><td>{t(lang, 'relL2')}</td><td /><td className="num">{cmp.rel_l2.toFixed(3)}</td><td /></tr>
          </tbody>
        </table>
      ) : (
        <dl className="qoi">
          {QOI_DEFS.map((d) => {
            const v = (row as unknown as Record<string, number | null>)[d.key];
            const lab = label?.[d.key];
            return (
              <div key={d.key}>
                <dt>{d.label}</dt>
                <dd className="num" title={lab !== undefined ? `label: ${lab}` : undefined}>
                  {sci(v === null ? null : v * (d.scale ?? 1))}{d.unit && <span className="unit"> {d.unit}</span>}
                </dd>
              </div>
            );
          })}
          {e && (
            <div>
              <dt>{t(lang, 'confidence')}</dt>
              <dd><span className={`status ${e.cls}`}>{e.icon} {e.text}</span> <span className="unit">{((row.eta ?? 0) * 100).toFixed(2)} %</span></dd>
            </div>
          )}
        </dl>
      )}
      {row.degenerate && <div className="note warning-text">≈ {t(lang, 'degenerate')}: per-mode values depend on the pair's rotation.</div>}
      {timings && <div className="note">{t(lang, 'timings')}: {Object.entries(timings).map(([k, v]) => `${k} ${v.toFixed(2)} s`).join(' · ')}</div>}
      {exportId && <a className="button" href={api.exportUrl(exportId)} download>{t(lang, 'export')}</a>}
    </div>
  );
}
