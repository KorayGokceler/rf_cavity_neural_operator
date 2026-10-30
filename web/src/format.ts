import type { Key, Lang } from './i18n';
import { t } from './i18n';

export function sci(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return '—';
  const a = Math.abs(v);
  if (a !== 0 && (a >= 1e5 || a < 1e-2)) {
    const [m, e] = v.toExponential(digits - 1).split('e');
    return `${m}·10${sup(Number(e))}`;
  }
  return Number(v.toPrecision(digits)).toLocaleString('en-US', { maximumSignificantDigits: digits });
}

const SUP: Record<string, string> = { '-': '⁻', '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴', '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹' };
const sup = (n: number) => String(n).split('').map((c) => SUP[c] ?? c).join('');

export const fmtF = (f: number) => (Number.isFinite(f) ? f.toFixed(4) : '—');

/** Confidence class of the label-free residual η (thresholds: 2 % / 10 %). */
export function etaClass(eta: number): 'good' | 'fair' | 'poor' {
  return eta < 0.02 ? 'good' : eta < 0.1 ? 'fair' : 'poor';
}

export function etaLabel(lang: Lang, eta: number) {
  const c = etaClass(eta);
  return { cls: c, icon: c === 'good' ? '✓' : c === 'fair' ? '!' : '✕', text: t(lang, c) };
}

export interface QoiDef { key: string; label: string; unit: string; scale?: number }
export const QOI_DEFS: QoiDef[] = [
  { key: 'Q0', label: 'Q₀', unit: '' },
  { key: 'G_ohm', label: 'G', unit: 'Ω' },
  { key: 'R_over_Q_ohm', label: 'R/Q', unit: 'Ω' },
  { key: 'R_sh_ohm', label: 'R_sh', unit: 'MΩ', scale: 1e-6 },
  { key: 'T_transit', label: 'T', unit: '' },
  { key: 'Epk_Eacc', label: 'E_pk/E_acc', unit: '' },
  { key: 'Bpk_Eacc_mT_per_MVm', label: 'B_pk/E_acc', unit: 'mT/(MV/m)' },
];

/** Integer with thin-space grouping (locale-neutral: '1 317', never '1.317' next to frequencies). */
export const int = (n: number) => (Number.isFinite(n) ? Math.round(n).toString().replace(/\B(?=(\d{3})+(?!\d))/g, '\u2009') : '—');

const TIMING_ORDER = ['mesh', 'operators', 'model', 'qoi', 'display'] as const;

/** Prediction timings (time_s: mesh, operators, model, qoi, display, total) as one line:
 *  "1.23 s (mesh 0.20 · operators 0.08 · …)"; the total is the server's own, never the sum of the parts. */
export function fmtTimings(lang: Lang, ts: Record<string, number> | null | undefined): string {
  if (!ts) return '';
  const parts = [...TIMING_ORDER.filter((k) => k in ts), ...Object.keys(ts).filter((k) => k !== 'total' && !(TIMING_ORDER as readonly string[]).includes(k))]
    .map((k) => `${(TIMING_ORDER as readonly string[]).includes(k) ? t(lang, `tm_${k}` as Key) : k} ${ts[k].toFixed(2)}`);
  const total = ts.total ?? Object.values(ts).reduce((a, b) => a + b, 0);
  return `${total.toFixed(2)} s${parts.length ? ` (${parts.join(' · ')})` : ''}`;
}
