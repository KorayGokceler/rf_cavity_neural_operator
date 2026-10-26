// Small shared controls of the design system: segmented control, icon button, collapsible section, popover menu,
// validated number field. No UI library; styles in styles.css.
import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react';
import { createPortal } from 'react-dom';
import Icon, { type IconName } from './Icon';
import { useOverlay } from '../overlay';
import { t, type Lang } from '../i18n';

export function useMedia(query: string): boolean {
  const [m, setM] = useState(() => window.matchMedia(query).matches);
  useEffect(() => {
    const mq = window.matchMedia(query);
    const f = () => setM(mq.matches);
    mq.addEventListener('change', f);
    f();
    return () => mq.removeEventListener('change', f);
  }, [query]);
  return m;
}

export interface SegOption<T extends string> {
  value: T; label?: ReactNode; icon?: IconName; title?: string; testid?: string; disabled?: boolean;
}

export function Seg<T extends string>({ value, options, onChange, label, className = '' }: {
  value: T; options: SegOption<T>[]; onChange: (v: T) => void; label: string; className?: string;
}) {
  return (
    <div className={`seg ${className}`} role="group" aria-label={label}>
      {options.map((o) => (
        <button key={o.value} type="button" className={value === o.value ? 'on' : ''} aria-pressed={value === o.value}
                title={o.title ?? (typeof o.label === 'string' ? o.label : label)} disabled={o.disabled}
                aria-label={o.title} data-testid={o.testid} onClick={() => onChange(o.value)}>
          {o.icon && <Icon name={o.icon} />}{o.label !== undefined && <span>{o.label}</span>}
        </button>
      ))}
    </div>
  );
}

export function IconButton({ icon, label, onClick, active, disabled, text, className = '', testid, size = 16 }: {
  icon: IconName; label: string; onClick?: () => void; active?: boolean; disabled?: boolean; text?: string;
  className?: string; testid?: string; size?: number;
}) {
  return (
    <button type="button" className={`ibtn ${active ? 'on' : ''} ${text ? 'with-text' : ''} ${className}`} title={label}
            aria-label={label} aria-pressed={active} disabled={disabled} data-testid={testid} onClick={onClick}>
      <Icon name={icon} size={size} />{text && <span>{text}</span>}
    </button>
  );
}

/** A titled section whose body can be folded (Advanced groups start closed). */
export function Collapsible({ title, defaultOpen = true, children, actions, className = '' }: {
  title: ReactNode; defaultOpen?: boolean; children: ReactNode; actions?: ReactNode; className?: string;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className={`collapsible ${open ? 'open' : ''} ${className}`}>
      <div className="collapsible-head">
        <button type="button" className="collapsible-toggle" aria-expanded={open} onClick={() => setOpen(!open)}>
          <Icon name={open ? 'down' : 'right'} size={14} /><span>{title}</span>
        </button>
        {actions && <div className="collapsible-actions">{actions}</div>}
      </div>
      {open && <div className="collapsible-body">{children}</div>}
    </section>
  );
}

/** Button with a popover panel (in a portal, positioned from the button: no parent clips it); closes on an
 *  outside click, on scroll / resize and on Esc (also via the overlay stack). */
export function Menu({ icon, label, text, children, testid, align = 'start', active, disabled, tone }: {
  icon: IconName; label: string; text?: string; children: ReactNode | ((close: () => void) => ReactNode);
  testid?: string; align?: 'start' | 'end'; active?: boolean; disabled?: boolean; tone?: 'warn';
}) {
  const [open, setOpen] = useState(false);
  const [at, setAt] = useState<{ top: number; left: number } | null>(null);
  const box = useRef<HTMLDivElement>(null);
  const panel = useRef<HTMLDivElement>(null);
  const close = () => { setOpen(false); setAt(null); };
  useOverlay(open, close);
  useEffect(() => {
    if (!open) return;
    const down = (e: PointerEvent) => {
      const n = e.target as Node;
      if (box.current?.contains(n) || panel.current?.contains(n)) return;
      close();
    };
    window.addEventListener('pointerdown', down);
    window.addEventListener('resize', close);
    return () => { window.removeEventListener('pointerdown', down); window.removeEventListener('resize', close); };
  }, [open]);
  useLayoutEffect(() => {                               // below the button, kept inside the window
    if (!open || !box.current || !panel.current) return;
    const r = box.current.getBoundingClientRect(), pw = panel.current.offsetWidth, ph = panel.current.offsetHeight;
    const left = align === 'end' ? r.right - pw : r.left;
    const top = r.bottom + 6 + ph > window.innerHeight - 8 ? Math.max(8, r.top - 6 - ph) : r.bottom + 6;
    setAt({ top, left: Math.min(Math.max(8, left), Math.max(8, window.innerWidth - pw - 8)) });
  }, [open, align]);
  return (
    <div className="menu" ref={box}>
      <button type="button" className={`ibtn ${tone ?? ''} ${(active || open) && !tone ? 'on' : ''} ${text ? 'with-text' : ''}`} title={label}
              aria-label={label} aria-haspopup="true" aria-expanded={open} disabled={disabled} data-testid={testid}
              onClick={() => (open ? close() : setOpen(true))}>
        <Icon name={icon} />{text && <span>{text}</span>}<Icon name="down" size={12} className="caret" />
      </button>
      {open && createPortal(
        <div className="menu-panel" ref={panel} role="dialog" aria-label={label} style={at ? { top: at.top, left: at.left } : { top: 0, left: 0, visibility: 'hidden' }}
             onKeyDown={(e) => { if (e.key === 'Escape') { e.stopPropagation(); close(); } }}>
          {typeof children === 'function' ? children(close) : children}
        </div>, document.body)}
    </div>
  );
}

/** Labelled number input with inline range validation (message under the field, aria-invalid). */
export function NumField({ lang, label, value, onChange, min, max, step, integer, disabled, placeholder, allowEmpty,
                           testid, wide }: {
  lang: Lang; label: ReactNode; value: number | null | ''; onChange: (v: number | null) => void; min?: number;
  max?: number; step?: number; integer?: boolean; disabled?: boolean; placeholder?: string; allowEmpty?: boolean;
  testid?: string; wide?: boolean;
}) {
  const empty = value === null || value === '';
  const v = empty ? NaN : Number(value);
  let err: string | null = null;
  if (!disabled) {
    if (empty) err = allowEmpty ? null : t(lang, 'vRequired');
    else if (!Number.isFinite(v)) err = t(lang, 'vNumber');
    else if (integer && !Number.isInteger(v)) err = t(lang, 'vInteger');
    else if ((min !== undefined && v < min) || (max !== undefined && v > max)) {
      err = `${t(lang, 'vRange')} ${min ?? '…'} – ${max ?? '…'}`;
    }
  }
  return (
    <label className={`field ${err ? 'invalid' : ''} ${wide ? 'wide' : ''}`}>
      <span className="field-label">{label}</span>
      <input type="number" value={value ?? ''} min={min} max={max} step={step} disabled={disabled}
             placeholder={placeholder} aria-invalid={err ? true : undefined} data-testid={testid}
             onChange={(e) => onChange(e.target.value === '' ? null : Number(e.target.value))} />
      {err && <span className="field-error" role="alert">{err}</span>}
    </label>
  );
}

/** Whether a NumField with these constraints currently holds a valid value. */
export function numOk(value: number | null | '', o: { min?: number; max?: number; integer?: boolean; allowEmpty?: boolean }): boolean {
  if (value === null || value === '') return !!o.allowEmpty;
  const v = Number(value);
  if (!Number.isFinite(v)) return false;
  if (o.integer && !Number.isInteger(v)) return false;
  return !((o.min !== undefined && v < o.min) || (o.max !== undefined && v > o.max));
}
