import type { ModeRow } from '../api';
import { etaLabel, fmtF, sci } from '../format';
import { t, type Lang } from '../i18n';

interface Props { lang: Lang; modes: ModeRow[]; selected: number; onSelect: (k: number) => void }

export default function ModeTable({ lang, modes, selected, onSelect }: Props) {
  return (
    <div className="card">
      <div className="card-title">{t(lang, 'modes')}</div>
      <table className="modes">
        <thead>
          <tr><th>#</th><th>f [GHz]</th><th>Q₀</th><th>R/Q [Ω]</th><th>E<sub>pk</sub>/E<sub>acc</sub></th><th>η</th></tr>
        </thead>
        <tbody>
          {modes.map((m) => {
            const e = etaLabel(lang, m.eta);
            return (
              <tr key={m.mode} className={m.mode === selected ? 'sel' : ''} tabIndex={0}
                  onClick={() => onSelect(m.mode)} onKeyDown={(ev) => ev.key === 'Enter' && onSelect(m.mode)}>
                <td>{m.mode}{m.accelerating && <span className="tag acc" title={t(lang, 'accelerating')}>★</span>}</td>
                <td className="num">{fmtF(m.f_GHz)}{m.degenerate && <span className="tag" title={t(lang, 'degenerate')}>≈</span>}</td>
                <td className="num">{sci(m.Q0)}</td>
                <td className="num">{sci(m.R_over_Q_ohm)}</td>
                <td className="num">{sci(m.Epk_Eacc)}</td>
                <td><span className={`status ${e.cls}`} title={`η = ${(m.eta * 100).toFixed(2)} %`}>{e.icon}</span></td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div className="note">★ {t(lang, 'accelerating')} · ≈ {t(lang, 'degenerate')} · η: {t(lang, 'confidence')}</div>
    </div>
  );
}
