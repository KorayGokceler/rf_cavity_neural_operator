// Colour bar with its title (quantity · mode · frequency · source) and the scale type.
import { sci } from '../format';

interface Props {
  gradient: string;                 // CSS linear-gradient
  range: [number, number];
  unit: string;
  title: string;
  neutral?: boolean;                // no colour mapping (mesh views): the title only
  scaleTag?: string;                // 'linear' | 'log10 · 3 dec'
  badges?: string[];                // e.g. locked
}

export default function ColorBar({ gradient, range, unit, title, neutral, scaleTag, badges = [] }: Props) {
  return (
    <div className="colorbar" role="group" aria-label={title}>
      <div className="colorbar-title">
        <span className="colorbar-text">{title}</span>
        {scaleTag && <span className="chip tiny">{scaleTag}</span>}
        {badges.map((b) => <span key={b} className="chip tiny accent">{b}</span>)}
      </div>
      {!neutral && (
        <div className="colorbar-row">
          <span className="num">{sci(range[0])}</span>
          <div className="colorbar-bar" style={{ background: gradient }} />
          <span className="num">{sci(range[1])} {unit}</span>
        </div>
      )}
    </div>
  );
}
