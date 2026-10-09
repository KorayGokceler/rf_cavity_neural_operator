// Small single-series SVG charts for the dataset overview (dataviz reference palette, slot 1).
import { useState } from 'react';
import { int } from '../format';

/** Horizontal bars: one per category, value labels at the bar end, hover tooltip. */
export function BarList({ data, title, onPick, picked }: {
  data: { label: string; value: number }[]; title: string; onPick?: (l: string) => void; picked?: string;
}) {
  const max = Math.max(1, ...data.map((d) => d.value));
  return (
    <div className="card">
      <div className="card-title">{title}</div>
      <div className="barlist">
        {data.map((d) => (
          <button key={d.label} className={`bar-row ${picked === d.label ? 'on' : ''}`} onClick={() => onPick?.(d.label)}
                  title={`${d.label}: ${int(d.value)}`}>
            <span className="bar-label">{d.label}</span>
            <span className="bar-track"><span className="bar-fill" style={{ width: `${(d.value / max) * 100}%` }} /></span>
            <span className="bar-value">{int(d.value)}</span>
          </button>
        ))}
      </div>
    </div>
  );
}

/** Histogram of values: 2px gaps between bins, recessive axis, hover tooltip per bin. */
export function Histogram({ values, title, unit, bins = 30 }: {
  values: number[]; title: string; unit: string; bins?: number;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const v = values.filter(Number.isFinite);
  if (!v.length) return <div className="card"><div className="card-title">{title}</div><div className="muted">—</div></div>;
  const lo = Math.min(...v), hi = Math.max(...v);
  const w = (hi - lo) / bins || 1;
  const counts = new Array(bins).fill(0);
  v.forEach((x) => { counts[Math.min(bins - 1, Math.floor((x - lo) / w))] += 1; });
  const cmax = Math.max(...counts);
  const W = 520, H = 150, M = { l: 36, r: 8, t: 8, b: 26 };
  const bw = (W - M.l - M.r) / bins;
  const sy = (c: number) => H - M.b - (c / cmax) * (H - M.t - M.b);
  const fmt = (x: number) => (Math.abs(x) >= 100 ? x.toFixed(0) : x.toFixed(2));
  return (
    <div className="card plot">
      <div className="card-title">{title}</div>
      <div className="plot-wrap">
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title} onPointerLeave={() => setHover(null)}>
          <line x1={M.l} x2={W - M.r} y1={H - M.b} y2={H - M.b} className="axis-base" />
          <line x1={M.l} x2={W - M.r} y1={sy(cmax)} y2={sy(cmax)} className="grid" />
          <text x={M.l - 4} y={sy(cmax) + 4} textAnchor="end" className="tick">{cmax}</text>
          <text x={M.l - 4} y={H - M.b + 4} textAnchor="end" className="tick">0</text>
          {counts.map((c, i) => (
            <g key={i} onPointerEnter={() => setHover(i)}>
              <rect x={M.l + i * bw} y={M.t} width={bw} height={H - M.t - M.b} fill="transparent" />
              {c > 0 && <rect x={M.l + i * bw + 1} y={sy(c)} width={Math.max(bw - 2, 1)} height={H - M.b - sy(c)}
                              rx={2} className={`hist-bar ${hover === i ? 'hover' : ''}`} />}
            </g>
          ))}
          <text x={M.l} y={H - 8} className="tick">{fmt(lo)}</text>
          <text x={W - M.r} y={H - 8} textAnchor="end" className="tick">{fmt(hi)} {unit}</text>
        </svg>
        {hover !== null && (
          <div className="tooltip" style={{ left: `${((M.l + (hover + 0.5) * bw) / W) * 100}%` }}>
            <strong>{counts[hover]}</strong>
            <span>{fmt(lo + hover * w)}–{fmt(lo + (hover + 1) * w)} {unit}</span>
          </div>
        )}
      </div>
    </div>
  );
}
