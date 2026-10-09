// E_z on the beam axis: one series, 2px line, recessive grid, crosshair + tooltip; gaps outside Ω.
import { useMemo, useRef, useState } from 'react';

interface Props { z: number[]; ez: (number | null)[]; title: string; note: string; scale: number; empty: string }

const W = 560, H = 190, M = { l: 56, r: 12, t: 10, b: 30 };

function ticks(lo: number, hi: number, n = 5) {
  const span = hi - lo || 1;
  const step = Math.pow(10, Math.floor(Math.log10(span / n)));
  const err = (span / n) / step;
  const s = step * (err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1);
  const out: number[] = [];
  for (let v = Math.ceil(lo / s) * s; v <= hi + 1e-9 * span; v += s) out.push(+v.toPrecision(12));
  return out;
}

function fmtField(v: number) {
  const a = Math.abs(v);
  if (a >= 1e6) return `${(v / 1e6).toFixed(a >= 1e7 ? 0 : 1)} MV/m`;
  if (a >= 1e3) return `${(v / 1e3).toFixed(a >= 1e4 ? 0 : 1)} kV/m`;
  return `${v.toFixed(0)} V/m`;
}

export default function AxisPlot({ z, ez, title, note, scale, empty }: Props) {
  const ref = useRef<SVGSVGElement>(null);
  const [hover, setHover] = useState<number | null>(null);
  const d = useMemo(() => {
    const pts = z.map((x, i) => [x, ez[i]] as [number, number | null]).filter(([x]) => Number.isFinite(x));
    const vals = pts.map(([, y]) => y).filter((y): y is number => y !== null && Number.isFinite(y));
    if (!pts.length || !vals.length) return null;
    const x0 = pts[0][0], x1 = pts[pts.length - 1][0];
    const ymax = Math.max(...vals.map(Math.abs)) || 1;
    const sx = (x: number) => M.l + ((x - x0) / (x1 - x0 || 1)) * (W - M.l - M.r);
    const sy = (y: number) => M.t + (1 - (y + ymax) / (2 * ymax)) * (H - M.t - M.b);
    let path = '', pen = false;
    for (const [x, y] of pts) {
      if (y === null || !Number.isFinite(y)) { pen = false; continue; }
      path += `${pen ? 'L' : 'M'}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`;
      pen = true;
    }
    return { pts, x0, x1, ymax, sx, sy, path, xt: ticks(x0, x1), yt: ticks(-ymax, ymax, 4) };
  }, [z, ez]);

  if (!d) return <div className="card muted">{title}: —</div>;
  if (d.ymax < 1e-6 * scale) return <div className="card"><div className="card-title">{title}</div><div className="muted">{empty}</div></div>;

  const onMove = (e: React.PointerEvent) => {
    const r = ref.current!.getBoundingClientRect();
    const px = ((e.clientX - r.left) / r.width) * W;
    const x = d.x0 + ((px - M.l) / (W - M.l - M.r)) * (d.x1 - d.x0);
    let best = 0, bd = Infinity;
    d.pts.forEach(([zx], i) => { const dd = Math.abs(zx - x); if (dd < bd) { bd = dd; best = i; } });
    setHover(best);
  };
  const hv = hover !== null ? d.pts[hover] : null;

  return (
    <div className="card plot">
      <div className="card-title">{title}</div>
      <div className="plot-wrap">
        <svg ref={ref} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title}
             onPointerMove={onMove} onPointerLeave={() => setHover(null)}>
          {d.yt.map((y) => (
            <g key={`y${y}`}>
              <line x1={M.l} x2={W - M.r} y1={d.sy(y)} y2={d.sy(y)} className={y === 0 ? 'axis-base' : 'grid'} />
              <text x={M.l - 6} y={d.sy(y) + 4} textAnchor="end" className="tick">{fmtField(y)}</text>
            </g>
          ))}
          {d.xt.map((x) => (
            <text key={`x${x}`} x={d.sx(x)} y={H - 10} textAnchor="middle" className="tick">{x}</text>
          ))}
          <text x={W - M.r} y={H - 10} textAnchor="end" className="tick">s [mm]</text>
          <path d={d.path} className="series-line" />
          {hv && hv[1] !== null && (
            <g>
              <line x1={d.sx(hv[0])} x2={d.sx(hv[0])} y1={M.t} y2={H - M.b} className="crosshair" />
              <circle cx={d.sx(hv[0])} cy={d.sy(hv[1])} r={4} className="series-dot" />
            </g>
          )}
        </svg>
        {hv && (
          <div className="tooltip" style={{ left: `${(d.sx(hv[0]) / W) * 100}%` }}>
            <strong>{hv[1] === null ? '—' : fmtField(hv[1])}</strong>
            <span>s = {hv[0].toFixed(1)} mm</span>
          </div>
        )}
      </div>
      <div className="note">{note}</div>
    </div>
  );
}
