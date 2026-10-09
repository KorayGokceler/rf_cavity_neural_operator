import type { CmpRow, ModeRow } from '../api';
import { etaLabel, fmtF, sci } from '../format';
import { t, type Lang } from '../i18n';

interface Props {
  lang: Lang; modes: ModeRow[]; cmp?: CmpRow[] | null; selected: number; onSelect: (k: number) => void;
}

const pct = (v: number | null | undefined) => (v === null || v === undefined || !Number.isFinite(v) ? '—' : `${(v * 100).toFixed(2)} %`);

function Eta({ lang, eta }: { lang: Lang; eta: number | null }) {
  if (eta === null || eta === undefined) return <span className="muted">—</span>;
  const e = etaLabel(lang, eta);
  return <span className={`status ${e.cls}`} title={`η = ${(eta * 100).toFixed(2)} %`}>{e.icon}</span>;
}

export default function ModeTable({ lang, modes, cmp, selected, onSelect }: Props) {
  const row = (k: number, cells: React.ReactNode) => (
    <tr key={k} className={k === selected ? 'sel' : ''} tabIndex={0}
        onClick={() => onSelect(k)} onKeyDown={(ev) => ev.key === 'Enter' && onSelect(k)}>{cells}</tr>
  );
  const tags = (m: { accelerating: boolean; degenerate: boolean }) => (
    <>{m.accelerating && <span className="tag acc" title={t(lang, 'accelerating')}>★</span>}
      {m.degenerate && <span className="tag" title={t(lang, 'degenerate')}>≈</span>}</>
  );
  return (
    <div className="card">
      <div className="card-title">{cmp ? t(lang, 'cmpTitle') : t(lang, 'modes')}</div>
      <table className="modes">
        {cmp ? (
          <>
            <thead><tr><th>#</th><th>f FE</th><th>f model</th><th>Δf</th><th>rel-L2</th><th>η</th></tr></thead>
            <tbody>
              {cmp.map((c) => row(c.mode, <>
                <td>{c.mode}{tags(c)}</td>
                <td className="num">{fmtF(c.f_fe)}</td>
                <td className="num">{fmtF(c.f_model)}</td>
                <td className="num">{pct(c.f_rel_err)}</td>
                <td className="num">{c.rel_l2.toFixed(3)}</td>
                <td><Eta lang={lang} eta={c.eta} /></td>
              </>))}
            </tbody>
          </>
        ) : (
          <>
            <thead><tr><th>#</th><th>f [GHz]</th><th>Q₀</th><th>R/Q [Ω]</th><th>E<sub>pk</sub>/E<sub>acc</sub></th><th>η</th></tr></thead>
            <tbody>
              {modes.map((m) => row(m.mode, <>
                <td>{m.mode}{tags(m)}</td>
                <td className="num">{fmtF(m.f_GHz)}</td>
                <td className="num">{sci(m.Q0)}</td>
                <td className="num">{sci(m.R_over_Q_ohm)}</td>
                <td className="num">{sci(m.Epk_Eacc)}</td>
                <td><Eta lang={lang} eta={m.eta} /></td>
              </>))}
            </tbody>
          </>
        )}
      </table>
      <div className="note">★ {t(lang, 'accelerating')} · ≈ {t(lang, 'degenerate')} · η: {t(lang, 'confidence')}{cmp ? ` · rel-L2: ${t(lang, 'relL2')}` : ''}</div>
    </div>
  );
}
