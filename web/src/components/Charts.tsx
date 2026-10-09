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

export interface LineSeries { label: string; cls: 's1' | 's2'; pts: [number, number][] }

/** Training curves: ≤ 2 series on one y-axis (optionally log), legend + end labels, crosshair tooltip. */
export function LineChart({ title, series, log = false, fmt, xLabel }: {
  title: string; series: LineSeries[]; log?: boolean; fmt: (v: number) => string; xLabel: string;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const W = 520, H = 190, M = { l: 52, r: 74, t: 10, b: 28 };
  const live = series.map((s) => ({ ...s, pts: s.pts.filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y) && (!log || y > 0)) }))
    .filter((s) => s.pts.length);
  if (!live.length) return <div className="card plot"><div className="card-title">{title}</div><div className="muted">—</div></div>;
  const xs = live.flatMap((s) => s.pts.map(([x]) => x)), ys = live.flatMap((s) => s.pts.map(([, y]) => y));
  const x0 = Math.min(...xs), x1 = Math.max(...xs);
  const tf = (y: number) => (log ? Math.log10(y) : y);
  let y0 = Math.min(...ys.map(tf)), y1 = Math.max(...ys.map(tf));
  if (y1 - y0 < 1e-12) { y0 -= log ? 0.5 : Math.abs(y0) * 0.1 + 1e-12; y1 += log ? 0.5 : Math.abs(y1) * 0.1 + 1e-12; }
  if (!log && y0 > 0 && y0 < 0.35 * y1) y0 = 0;                       // anchor at zero when it is close
  const sx = (x: number) => M.l + (x1 > x0 ? (x - x0) / (x1 - x0) : 0.5) * (W - M.l - M.r);
  const sy = (y: number) => M.t + (1 - (tf(y) - y0) / (y1 - y0)) * (H - M.t - M.b);
  const yt = log
    ? Array.from({ length: Math.floor(y1) - Math.ceil(y0) + 1 }, (_, i) => 10 ** (Math.ceil(y0) + i))
    : [0, 0.25, 0.5, 0.75, 1].map((f) => y0 + f * (y1 - y0));
  const ticksY = yt.length >= 2 ? yt : [10 ** y0, 10 ** y1];
  const xt = Array.from(new Set([x0, Math.round((x0 + x1) / 2), x1]));
  const path = (pts: [number, number][]) => pts.map(([x, y], i) => `${i ? 'L' : 'M'}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join('');
  const epochs = Array.from(new Set(xs)).sort((a, b) => a - b);
  // end labels (2 series): at the last point, pushed apart when they would overlap
  const ends = live.map((s) => sy(s.pts[s.pts.length - 1][1]) + 4);
  if (ends.length === 2 && Math.abs(ends[0] - ends[1]) < 12) {
    const mid = (ends[0] + ends[1]) / 2, up = ends[0] <= ends[1] ? 0 : 1;
    ends[up] = mid - 6; ends[1 - up] = mid + 6;
  }
  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const x = x0 + (((e.clientX - r.left) / r.width) * W - M.l) / (W - M.l - M.r) * (x1 - x0);
    setHover(epochs.reduce((b, v) => (Math.abs(v - x) < Math.abs(b - x) ? v : b), epochs[0]));
  };
  return (
    <div className="card plot">
      <div className="card-title">{title}
        {live.length > 1 && <span className="chart-legend">{live.map((s) => <span key={s.label}><span className={`sw ${s.cls}`} />{s.label}</span>)}</span>}
      </div>
      <div className="plot-wrap">
        <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title} onPointerMove={onMove} onPointerLeave={() => setHover(null)}>
          {ticksY.map((y, i) => (
            <g key={i}>
              <line x1={M.l} x2={W - M.r} y1={sy(y)} y2={sy(y)} className={i === 0 && !log ? 'axis-base' : 'grid'} />
              <text x={M.l - 6} y={sy(y) + 4} textAnchor="end" className="tick">{fmt(y)}</text>
            </g>
          ))}
          {xt.map((x, i) => <text key={x} x={sx(x)} y={H - 8} textAnchor="middle" className="tick">{i === xt.length - 1 ? `${x} ${xLabel}` : x}</text>)}
          {live.map((s, si) => (
            <g key={s.label}>
              {s.pts.length > 1 && <path d={path(s.pts)} className={`series-line ${s.cls}`} />}
              {s.pts.length <= 12 && s.pts.map(([x, y]) => <circle key={x} cx={sx(x)} cy={sy(y)} r={3} className={`series-dot ${s.cls}`} />)}
              {live.length > 1 && <text x={sx(s.pts[s.pts.length - 1][0]) + 6} y={ends[si]} className="end-label">{s.label}</text>}
            </g>
          ))}
          {hover !== null && (
            <g>
              <line x1={sx(hover)} x2={sx(hover)} y1={M.t} y2={H - M.b} className="crosshair" />
              {live.map((s) => {
                const p = s.pts.find(([x]) => x === hover);
                return p ? <circle key={s.label} cx={sx(p[0])} cy={sy(p[1])} r={4} className={`series-dot ${s.cls}`} /> : null;
              })}
            </g>
          )}
        </svg>
        {hover !== null && (
          <div className="tooltip" style={{ left: `${(sx(hover) / W) * 100}%` }}>
            <strong>{xLabel} {hover}</strong>
            {live.map((s) => {
              const p = s.pts.find(([x]) => x === hover);
              return p ? <span key={s.label}>{s.label}: {fmt(p[1])}</span> : null;
            })}
          </div>
        )}
      </div>
    </div>
  );
}
