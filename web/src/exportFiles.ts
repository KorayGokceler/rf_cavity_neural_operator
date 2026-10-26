// File exports of the workspace: a PNG of the 3D view(s) with the colour bar, title and a legend line, and a CSV of
// the mode table. Both are built in the browser from data already there.
import type { CmpRow, ModeRow } from './api';
import { QOI_DEFS, sci } from './format';

export function saveBlob(blob: Blob, name: string): void {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

export const safeName = (s: string) => s.replace(/[^A-Za-z0-9._-]+/g, '_').replace(/^_+|_+$/g, '') || 'view';

// ── CSV ─────────────────────────────────────────────────────────────────────
const cell = (v: unknown): string => {
  if (v === null || v === undefined) return '';
  if (typeof v === 'number') return Number.isFinite(v) ? String(v) : '';
  if (typeof v === 'boolean') return v ? '1' : '0';
  const s = String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

/** The mode table as CSV: header row, '.' decimals, empty cells for missing values. Columns: every value of the
 *  table and the cavity figures; with a comparison also FE / model / Δ [%] per figure. */
export function modesCsv(modes: ModeRow[], cmp: CmpRow[] | null): string {
  const q = (m: ModeRow, k: string, sc = 1) => {
    const v = (m as unknown as Record<string, number | null>)[k];
    return v === null || v === undefined ? null : v * sc;
  };
  const col = (d: (typeof QOI_DEFS)[number]) => (d.scale === 1e-6 ? d.key.replace('_ohm', '_MOhm') : d.key);   // R_sh in MΩ
  const head = ['mode', 'f_GHz', 'degenerate', 'accelerating', 'eta', ...QOI_DEFS.map(col)];
  if (cmp) {
    head.push('f_FE_GHz', 'f_model_GHz', 'f_rel_err', 'rel_L2');
    for (const d of QOI_DEFS) head.push(`${col(d)}_FE`, `${col(d)}_model`, `${col(d)}_delta_pct`);
  }
  const lines = [head.map(cell).join(',')];
  for (const m of modes) {
    const row: unknown[] = [m.mode, m.f_GHz, m.degenerate, m.accelerating, m.eta, ...QOI_DEFS.map((d) => q(m, d.key, d.scale ?? 1))];
    if (cmp) {
      const c = cmp.find((x) => x.mode === m.mode);
      row.push(c?.f_fe ?? null, c?.f_model ?? null, c?.f_rel_err ?? null, c?.rel_l2 ?? null);
      for (const d of QOI_DEFS) {
        const e = c?.qoi[d.key], sc = d.scale ?? 1;
        row.push(e?.fe == null ? null : e.fe * sc, e?.model == null ? null : e.model * sc,
                 e?.rel_err == null || c?.degenerate ? null : 100 * e.rel_err);
      }
    }
    lines.push(row.map(cell).join(','));
  }
  return lines.join('\n') + '\n';
}

// ── PNG ─────────────────────────────────────────────────────────────────────
export interface PanePng {
  image: string;                        // data URL of the canvas
  title: string;                        // quantity · mode · frequency · source
  stops: [number, string][];            // colour stops, positions 0..1
  range: [number, number];              // shown physical range
  unit: string;
  scaleTag: string;                     // linear | log10 · n decades
  log: number;                          // decades (0: linear)
  neutral: boolean;                     // no colour mapping
  note?: string;                        // e.g. the probe readout
}

interface Theme { bg: string; fg: string; muted: string; line: string }

const loadImage = (src: string) => new Promise<HTMLImageElement>((res, rej) => {
  const im = new Image();
  im.onload = () => res(im);
  im.onerror = () => rej(new Error('image'));
  im.src = src;
});

/** Panes side by side (or stacked), each with its title above and its colour bar below, one legend line under all. */
export async function composePng(panes: PanePng[], legend: string, stacked: boolean, th: Theme, cssWidth: number): Promise<Blob> {
  const imgs = await Promise.all(panes.map((p) => loadImage(p.image)));
  const s = Math.max(1, imgs[0].naturalWidth / Math.max(1, cssWidth));         // device pixels per CSS pixel
  const px = (n: number) => Math.round(n * s);
  const headH = px(30), footH = px(panes.some((p) => p.note) ? 74 : 56), legH = px(26);
  const W = stacked ? Math.max(...imgs.map((i) => i.naturalWidth)) : imgs.reduce((a, i) => a + i.naturalWidth, 0);
  const paneH = (i: number) => headH + imgs[i].naturalHeight + footH;
  const H = (stacked ? imgs.reduce((a, _, i) => a + paneH(i), 0) : Math.max(...imgs.map((_, i) => paneH(i)))) + legH;
  const cv = document.createElement('canvas');
  cv.width = W; cv.height = H;
  const g = cv.getContext('2d')!;
  g.fillStyle = th.bg; g.fillRect(0, 0, W, H);
  const font = (w: number, size: number) => `${w} ${px(size)}px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;
  let x = 0, y = 0;
  imgs.forEach((im, i) => {
    const p = panes[i];
    g.fillStyle = th.fg; g.font = font(600, 13); g.textBaseline = 'middle';
    g.fillText(p.title, x + px(10), y + headH / 2);
    g.drawImage(im, x, y + headH);
    const fy = y + headH + im.naturalHeight;
    if (!p.neutral) {
      const bw = px(200), bh = px(12), bx = x + px(10), by = fy + px(10);
      const grad = g.createLinearGradient(bx, 0, bx + bw, 0);
      p.stops.forEach(([pos, c]) => grad.addColorStop(Math.min(1, Math.max(0, pos)), c));
      g.fillStyle = grad; g.fillRect(bx, by, bw, bh);
      g.strokeStyle = th.line; g.lineWidth = Math.max(1, s * 0.75); g.strokeRect(bx, by, bw, bh);
      g.fillStyle = th.fg; g.font = font(400, 11); g.textBaseline = 'top';
      if (p.log) {
        g.textAlign = 'center';
        for (let k = 0; k <= p.log; k++) {
          const tx = bx + (bw * k) / p.log;
          g.beginPath(); g.moveTo(tx, by + bh); g.lineTo(tx, by + bh + px(3)); g.stroke();
          g.fillText(sci(p.range[1] * 10 ** (k - p.log), 2), tx, by + bh + px(4));
        }
        g.textAlign = 'left';
      } else {
        g.textAlign = 'left'; g.fillText(sci(p.range[0]), bx, by + bh + px(4));
        g.textAlign = 'right'; g.fillText(sci(p.range[1]), bx + bw, by + bh + px(4));
        g.textAlign = 'left';
      }
      g.fillStyle = th.muted; g.textBaseline = 'middle';
      g.fillText(`${p.unit}  ·  ${p.scaleTag}`, bx + bw + px(12), by + bh / 2);
    }
    if (p.note) { g.fillStyle = th.fg; g.font = font(400, 12); g.textBaseline = 'middle'; g.fillText(p.note, x + px(10), fy + footH - px(14)); }
    if (stacked) y += paneH(i); else x += im.naturalWidth;
  });
  g.fillStyle = th.muted; g.font = font(400, 11); g.textBaseline = 'middle'; g.textAlign = 'left';
  g.fillText(legend, px(10), H - legH / 2);
  return new Promise((res, rej) => cv.toBlob((b) => (b ? res(b) : rej(new Error('png'))), 'image/png'));
}
