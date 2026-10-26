// localStorage wrapper: every access is guarded (private mode / blocked storage / quota), the app works without it.
import { useCallback, useState } from 'react';

const PREFIX = 'rfcav.';

export function load<T>(key: string, fallback: T): T {
  try {
    const s = window.localStorage.getItem(PREFIX + key);
    return s === null ? fallback : (JSON.parse(s) as T);
  } catch { return fallback; }
}

export function save(key: string, value: unknown): void {
  try { window.localStorage.setItem(PREFIX + key, JSON.stringify(value)); } catch { /* storage unavailable */ }
}

/** useState that survives a reload (when the storage works). */
export function usePersisted<T>(key: string, init: T): [T, (v: T | ((p: T) => T)) => void] {
  const [v, setV] = useState<T>(() => load(key, init));
  const set = useCallback((n: T | ((p: T) => T)) => {
    setV((p) => {
      const next = typeof n === 'function' ? (n as (p: T) => T)(p) : n;
      save(key, next);
      return next;
    });
  }, [key]);
  return [v, set];
}
