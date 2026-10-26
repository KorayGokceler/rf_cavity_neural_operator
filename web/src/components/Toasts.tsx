// Toast notifications: bottom-right stack, auto-dismiss, announced through a live region.
import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import Icon from './Icon';

export type ToastKind = 'ok' | 'error' | 'info';
interface Item { id: number; kind: ToastKind; text: string }
type Notify = (kind: ToastKind, text: string) => void;

const Ctx = createContext<Notify>(() => undefined);
export const useToast = () => useContext(Ctx);

const TTL: Record<ToastKind, number> = { ok: 4500, info: 4500, error: 9000 };

export function ToastProvider({ children, closeLabel }: { children: ReactNode; closeLabel: string }) {
  const [items, setItems] = useState<Item[]>([]);
  const seq = useRef(0);
  const drop = useCallback((id: number) => setItems((l) => l.filter((x) => x.id !== id)), []);
  const notify = useCallback<Notify>((kind, text) => {
    const id = ++seq.current;
    setItems((l) => [...l.slice(-3), { id, kind, text }]);
  }, []);
  const ctx = useMemo(() => notify, [notify]);
  return (
    <Ctx.Provider value={ctx}>
      {children}
      <div className="toasts" aria-live="polite" aria-atomic="false" role="status">
        {items.map((it) => <ToastView key={it.id} item={it} onClose={drop} closeLabel={closeLabel} />)}
      </div>
    </Ctx.Provider>
  );
}

function ToastView({ item, onClose, closeLabel }: { item: Item; onClose: (id: number) => void; closeLabel: string }) {
  const [paused, setPaused] = useState(false);
  useEffect(() => {
    if (paused) return;
    const h = setTimeout(() => onClose(item.id), TTL[item.kind]);
    return () => clearTimeout(h);
  }, [paused, item, onClose]);
  return (
    <div className={`toast ${item.kind}`} role={item.kind === 'error' ? 'alert' : undefined}
         onMouseEnter={() => setPaused(true)} onMouseLeave={() => setPaused(false)}>
      <Icon name={item.kind === 'ok' ? 'check' : item.kind === 'error' ? 'warn' : 'info'} size={16} />
      <span className="toast-text">{item.text}</span>
      <button type="button" className="ibtn flat" aria-label={closeLabel} title={closeLabel} onClick={() => onClose(item.id)}>
        <Icon name="x" size={14} />
      </button>
    </div>
  );
}
