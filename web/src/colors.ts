// Field colour maps (dataviz reference palette): magnitude = one-hue sequential ramp (near zero
// recedes toward the surface), signed component = diverging blue ↔ neutral gray ↔ red.
export type Theme = 'light' | 'dark';

const BLUE = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']; // 100…700
const RED = ['#f6c4c3', '#ee8e8d', '#e34948', '#b92f2f', '#7f1d1d'];
const MID = { light: '#f0efec', dark: '#383835' };

export function hexToRgb(h: string): [number, number, number] {
  const s = h.trim().replace('#', '');
  const n = parseInt(s.length === 3 ? s.split('').map((c) => c + c).join('') : s, 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

/** [position in [0,1], hex] stops for a magnitude map. */
export function sequentialStops(theme: Theme): [number, string][] {
  const ramp = theme === 'light' ? BLUE : [...BLUE].reverse();      // near zero ≈ surface tone
  return ramp.map((c, i) => [i / (ramp.length - 1), c]);
}

/** [position in [-1,1], hex] stops for a signed map. */
export function divergingStops(theme: Theme): [number, string][] {
  const neg = [BLUE[6], BLUE[5], BLUE[3], BLUE[1]];
  const pos = [RED[0], RED[2], RED[3], RED[4]];
  return [
    ...neg.map((c, i) => [-1 + (i * 0.75) / (neg.length - 1), c] as [number, string]),
    [0, MID[theme]],
    ...pos.map((c, i) => [0.25 + (i * 0.75) / (pos.length - 1), c] as [number, string]),
  ];
}

export function cssGradient(stops: [number, string][], signed: boolean): string {
  const at = (p: number) => (signed ? (p + 1) / 2 : p) * 100;
  return `linear-gradient(to right, ${stops.map(([p, c]) => `${c} ${at(p).toFixed(1)}%`).join(', ')})`;
}
