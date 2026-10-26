// Stack of open overlays (menus, sheets, the shortcut help): Esc closes the topmost one first.
import { useEffect, useRef } from 'react';

const stack: (() => void)[] = [];

/** Register `close` while `open` is true. */
export function useOverlay(open: boolean, close: () => void): void {
  const ref = useRef(close);
  ref.current = close;
  useEffect(() => {
    if (!open) return;
    const fn = () => ref.current();
    stack.push(fn);
    return () => { const i = stack.indexOf(fn); if (i >= 0) stack.splice(i, 1); };
  }, [open]);
}

/** Close the topmost open overlay; false when none is open. */
export function closeTopOverlay(): boolean {
  const fn = stack[stack.length - 1];
  if (!fn) return false;
  fn();
  return true;
}
