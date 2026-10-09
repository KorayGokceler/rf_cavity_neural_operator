import type { ModeRow, Prediction } from '../api';
import { api } from '../api';
import { QOI_DEFS, etaLabel, fmtF, sci } from '../format';
import { t, type Lang } from '../i18n';

interface Props { lang: Lang; pred: Prediction; row: ModeRow }

export default function ModeDetails({ lang, pred, row }: Props) {
  const e = etaLabel(lang, row.eta);
  return (
    <div className="card">
      <div className="card-title">{t(lang, 'selected')}: {row.mode}</div>
      <div className="hero">
        <span className="hero-num">{fmtF(row.f_GHz)}</span><span className="hero-unit">GHz</span>
      </div>
      <dl className="qoi">
        {QOI_DEFS.map((d) => {
          const v = (row as unknown as Record<string, number | null>)[d.key];
          return (
            <div key={d.key}>
              <dt>{d.label}</dt>
              <dd className="num">{sci(v === null ? null : v * (d.scale ?? 1))}{d.unit && <span className="unit"> {d.unit}</span>}</dd>
            </div>
          );
        })}
        <div>
          <dt>{t(lang, 'confidence')}</dt>
          <dd><span className={`status ${e.cls}`}>{e.icon} {e.text}</span> <span className="unit">{(row.eta * 100).toFixed(2)} %</span></dd>
        </div>
      </dl>
      {row.degenerate && <div className="note warning-text">≈ {t(lang, 'degenerate')}: per-mode values depend on the pair's rotation.</div>}
      <div className="note">
        {t(lang, 'timings')}: {Object.entries(pred.time_s).map(([k, v]) => `${k} ${v.toFixed(2)} s`).join(' · ')}
      </div>
      <a className="button" href={api.exportUrl(pred.id)} download>{t(lang, 'export')}</a>
    </div>
  );
}
