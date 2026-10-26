// Inline SVG icons (24 × 24 grid, 1.75 px stroke, currentColor): no icon library.
const P = {
  cube: 'M12 3 3.5 7.5v9L12 21l8.5-4.5v-9z M3.5 7.5 12 12l8.5-4.5 M12 12v9',
  database: 'M4 5.5a8 3 0 0 0 16 0 8 3 0 0 0-16 0 M4 5.5v13a8 3 0 0 0 16 0v-13 M4 12a8 3 0 0 0 16 0',
  generate: 'M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z M19 16l.7 2 2 .7-2 .7-.7 2-.7-2-2-.7 2-.7z',
  training: 'M3 3v18h18 M7 15l4-5 3 3 5-7',
  terminal: 'M4 5h16a1 1 0 0 1 1 1v12a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V6a1 1 0 0 1 1-1z M7 10l3 2.5L7 15 M12 15h5',
  left: 'M15 6l-6 6 6 6',
  right: 'M9 6l6 6-6 6',
  down: 'M6 9l6 6 6-6',
  up: 'M6 15l6-6 6 6',
  panelLeft: 'M4 4h16v16H4z M9 4v16',
  panelRight: 'M4 4h16v16H4z M15 4v16',
  panelBottom: 'M4 4h16v16H4z M4 15h16',
  sun: 'M12 16a4 4 0 1 0 0-8 4 4 0 0 0 0 8z M12 2v2 M12 20v2 M2 12h2 M20 12h2 M4.9 4.9l1.4 1.4 M17.7 17.7l1.4 1.4 M4.9 19.1l1.4-1.4 M17.7 6.3l1.4-1.4',
  moon: 'M20 14.5A8.5 8.5 0 1 1 9.5 4a7 7 0 0 0 10.5 10.5z',
  help: 'M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z M9.5 9.5a2.5 2.5 0 1 1 3.5 2.3c-.7.4-1 .9-1 1.7 M12 17h.01',
  warn: 'M12 3 2.5 20h19z M12 10v4 M12 17.2h.01',
  check: 'M5 12.5l4.5 4.5L19 7.5',
  x: 'M6 6l12 12 M18 6 6 18',
  play: 'M8 5v14l11-7z',
  pause: 'M8 5v14 M16 5v14',
  home: 'M4 11 12 4l8 7 M6 10v9h12v-9 M10 19v-5h4v5',
  arrows: 'M5 19 19 5 M9 5h10v10',
  probe: 'M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z M12 3v5 M12 16v5 M3 12h5 M16 12h5',
  split: 'M4 5h16v14H4z M12 5v14',
  download: 'M12 4v11 M7.5 10.5 12 15l4.5-4.5 M5 19h14',
  image: 'M4 5h16v14H4z M4 16l5-5 4 4 3-3 4 4 M15.5 9.5h.01',
  table: 'M4 5h16v14H4z M4 10h16 M4 15h16 M10 5v14',
  lock: 'M6 11h12v9H6z M8.5 11V8a3.5 3.5 0 0 1 7 0v3',
  unlock: 'M6 11h12v9H6z M8.5 11V8a3.5 3.5 0 0 1 6.5-1.8',
  log: 'M4 4v16h16 M6 17c1-8 5-10 13-10',
  layers: 'M12 3 3 8l9 5 9-5z M3 12.5l9 5 9-5 M3 16.5l9 5 9-5',
  slice: 'M3 15l4.5-7H21l-4.5 7z M7.5 8V4 M16.5 15v5',
  upload: 'M12 16V4 M7.5 8.5 12 4l4.5 4.5 M5 20h14',
  clock: 'M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z M12 7v5l3 2',
  plus: 'M12 5v14 M5 12h14',
  link: 'M9.5 14.5l5-5 M8 11l-2 2a3.5 3.5 0 0 0 5 5l2-2 M16 13l2-2a3.5 3.5 0 0 0-5-5l-2 2',
  unlink: 'M8 11l-2 2a3.5 3.5 0 0 0 5 5l1-1 M16 13l2-2a3.5 3.5 0 0 0-5-5l-1 1 M9 4v2 M4 9h2 M15 18v2 M18 15h2',
  info: 'M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18z M12 11v5 M12 7.8h.01',
  keyboard: 'M3 6h18v12H3z M7 10h.01 M11 10h.01 M15 10h.01 M7 14h10',
  sliders: 'M5 7h8 M17 7h2 M5 17h2 M11 17h8 M15 5v4 M9 15v4',
  list: 'M8 6h12 M8 12h12 M8 18h12 M4 6h.01 M4 12h.01 M4 18h.01',
  folder: 'M3 7a1 1 0 0 1 1-1h5l2 2h9a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1z',
  copy: 'M9 9h11v11H9z M5 15H4V4h11v1',
  eye: 'M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z M12 15a3 3 0 1 0 0-6 3 3 0 0 0 0 6z',
  palette: 'M4 8h16v8H4z M8 8v8 M12 8v8 M16 8v8',
  field: 'M4 12c3-6 5-6 8 0s5 6 8 0',
  mesh: 'M4 4h16v16H4z M4 12h16 M12 4v16 M4 4l16 16',
  cpu: 'M7 7h10v10H7z M10 3v4 M14 3v4 M10 17v4 M14 17v4 M3 10h4 M3 14h4 M17 10h4 M17 14h4',
  close: 'M5 5l14 14 M19 5 5 19',
} as const;

export type IconName = keyof typeof P;

interface Props { name: IconName; size?: number; fill?: boolean; className?: string }

export default function Icon({ name, size = 16, fill = false, className }: Props) {
  return (
    <svg className={`icon-svg ${className ?? ''}`} width={size} height={size} viewBox="0 0 24 24" aria-hidden="true"
         focusable="false" fill={fill ? 'currentColor' : 'none'} stroke="currentColor" strokeWidth={1.75}
         strokeLinecap="round" strokeLinejoin="round">
      <path d={P[name]} />
    </svg>
  );
}
