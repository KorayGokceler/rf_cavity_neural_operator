// Keyboard shortcut cheat sheet (opened with "?"), a modal dialog closed with Esc or its button.
import { useEffect, useRef } from 'react';
import { t, type Key, type Lang } from '../i18n';
import { useOverlay } from '../overlay';
import Icon from './Icon';

export const SHORTCUTS: { group: Key; rows: { keys: string[]; text: Key }[] }[] = [
  { group: 'scView', rows: [
    { keys: ['1', '…', '7'], text: 'scPresets' }, { keys: ['H'], text: 'scHome' },
    { keys: ['X', 'Y', 'Z'], text: 'scAxis' }, { keys: ['[', ']'], text: 'scCutMove' },
    { keys: ['V'], text: 'scVolume' }, { keys: ['A'], text: 'scArrows' } ] },
  { group: 'scField', rows: [
    { keys: ['←', '→'], text: 'scMode' }, { keys: ['E'], text: 'scE' }, { keys: ['M'], text: 'scH' },
    { keys: ['S'], text: 'scSource' }, { keys: ['Space'], text: 'scPlay' } ] },
  { group: 'scGeneral', rows: [
    { keys: ['Esc'], text: 'scEsc' }, { keys: ['?'], text: 'scHelp' } ] },
];

export default function ShortcutHelp({ lang, open, onClose }: { lang: Lang; open: boolean; onClose: () => void }) {
  useOverlay(open, onClose);
  const btn = useRef<HTMLButtonElement>(null);
  useEffect(() => { if (open) btn.current?.focus(); }, [open]);
  if (!open) return null;
  return (
    <div className="modal-backdrop" onMouseDown={(e) => { if (e.target === e.currentTarget) onClose(); }}>
      <div className="modal" role="dialog" aria-modal="true" aria-label={t(lang, 'shortcuts')} data-testid="shortcut-help">
        <div className="modal-head">
          <Icon name="keyboard" size={16} /><span className="side-title">{t(lang, 'shortcuts')}</span>
          <button type="button" ref={btn} className="ibtn flat" aria-label={t(lang, 'sheetClose')} title={t(lang, 'sheetClose')} onClick={onClose}>
            <Icon name="x" size={14} />
          </button>
        </div>
        <div className="modal-body">
          {SHORTCUTS.map((g) => (
            <section key={g.group}>
              <h3>{t(lang, g.group)}</h3>
              <dl className="keys">
                {g.rows.map((r) => (
                  <div key={r.text}>
                    <dt>{r.keys.map((k, i) => (k === '…' ? <span key={i} className="muted">…</span> : <kbd key={i}>{k}</kbd>))}</dt>
                    <dd>{t(lang, r.text)}</dd>
                  </div>
                ))}
              </dl>
            </section>
          ))}
          <p className="note">{t(lang, 'scNote')}</p>
        </div>
      </div>
    </div>
  );
}
