import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api, type Info } from './api';
import DatasetPage from './components/DatasetPage';
import GeneratePage from './components/GeneratePage';
import TrainPage from './components/TrainPage';
import TrubaPage from './components/TrubaPage';
import Workspace, { type WorkspaceHandle } from './components/Workspace';
import Icon, { type IconName } from './components/Icon';
import { IconButton, Menu } from './components/ui';
import ShortcutHelp from './components/ShortcutHelp';
import { closeTopOverlay } from './overlay';
import { ToastProvider, useToast } from './components/Toasts';
import type { Theme } from './colors';
import { t, type Key, type Lang } from './i18n';
import { load, save } from './store';

type Page = 'work' | 'data' | 'gen' | 'train' | 'truba';
const NAV: { id: Page; icon: IconName; key: Key; short: Key }[] = [
  { id: 'work', icon: 'cube', key: 'workspace', short: 'navWork' },
  { id: 'data', icon: 'database', key: 'datasetTab', short: 'navData' },
  { id: 'gen', icon: 'generate', key: 'generateTab', short: 'navGen' },
  { id: 'train', icon: 'training', key: 'trainTab', short: 'navTrain' },
  { id: 'truba', icon: 'terminal', key: 'trubaTab', short: 'trubaTab' },
];

function initialTheme(): Theme {
  const saved = load<string | null>('theme', null);
  if (saved === 'light' || saved === 'dark') return saved;
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}
function initialLang(): Lang {
  const saved = load<string | null>('lang', null);
  if (saved === 'tr' || saved === 'en') return saved;
  return navigator.language?.startsWith('tr') ? 'tr' : 'en';
}

export default function App() {
  const [lang, setLang] = useState<Lang>(initialLang);
  return <ToastProvider closeLabel={t(lang, 'sheetClose')}><Shell lang={lang} setLang={setLang} /></ToastProvider>;
}

function Shell({ lang, setLang }: { lang: Lang; setLang: (l: Lang) => void }) {
  const toast = useToast();
  const [theme, setTheme] = useState<Theme>(initialTheme);
  const [page, setPage] = useState<Page>('work');
  const [seen, setSeen] = useState<Set<Page>>(() => new Set<Page>(['work']));
  const [railMin, setRailMin] = useState<boolean>(() => load('rail.min', false));
  const [info, setInfo] = useState<Info | null>(null);
  const [wsWarnings, setWsWarnings] = useState<string[]>([]);
  const [help, setHelp] = useState(false);
  const ws = useRef<WorkspaceHandle>(null);
  const pageRef = useRef<Page>('work');
  pageRef.current = page;

  useEffect(() => { document.documentElement.dataset.theme = theme; }, [theme]);
  // the theme attribute before the re-render: the 3D view reads its colours (CSS variables) in its effects,
  // which run before this component's
  const changeTheme = (th: Theme) => { document.documentElement.dataset.theme = th; setTheme(th); save('theme', th); };
  const changeLang = (l: Lang) => { setLang(l); save('lang', l); };
  useEffect(() => { document.documentElement.lang = lang; }, [lang]);
  useEffect(() => { api.info().then(setInfo).catch((e) => toast('error', String(e.message ?? e))); }, [toast]);

  const go = (p: Page) => { setPage(p); setSeen((s) => (s.has(p) ? s : new Set(s).add(p))); };
  const openItem = async (did: string, sid: number) => { if (await ws.current?.openDataset(did, sid)) go('work'); };
  const toggleRail = () => { setRailMin(!railMin); save('rail.min', !railMin); };

  // keyboard shortcuts: ignored while typing (or when the focused control uses the key itself); Esc closes the
  // topmost overlay first, then the open geometry
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.metaKey || e.altKey || e.isComposing) return;
      const el = e.target as HTMLElement | null;
      const tag = el?.tagName ?? '';
      const type = tag === 'INPUT' ? (el as HTMLInputElement).type : '';
      const typing = tag === 'TEXTAREA' || tag === 'SELECT' || el?.isContentEditable
        || (tag === 'INPUT' && !['range', 'checkbox', 'radio', 'button', 'file'].includes(type));
      if (typing) return;
      const activates = (tag === 'BUTTON' || tag === 'A' || tag === 'SUMMARY' || type === 'checkbox' || type === 'radio'
                         || type === 'file') && (e.key === ' ' || e.key === 'Enter');
      if (e.key === 'Escape') {
        if (closeTopOverlay()) { e.preventDefault(); return; }
        if (pageRef.current === 'work' && ws.current?.closeGeometry()) e.preventDefault();
        return;
      }
      if (e.key === '?') { e.preventDefault(); setHelp((h) => !h); return; }
      if (type === 'range' && (e.key.startsWith('Arrow') || ['Home', 'End', 'PageUp', 'PageDown'].includes(e.key))) return;
      if (activates || pageRef.current !== 'work') return;
      ws.current?.key(e);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  // the browser must not navigate to a file dropped outside the workspace
  useEffect(() => {
    const stop = (e: DragEvent) => { if (e.dataTransfer?.types.includes('Files')) e.preventDefault(); };
    window.addEventListener('dragover', stop);
    window.addEventListener('drop', stop);
    return () => { window.removeEventListener('dragover', stop); window.removeEventListener('drop', stop); };
  }, []);

  const warnings = useMemo(() => [
    ...(info?.model.untrained ? [`${t(lang, 'untrained')}${info.model.load_error ? ` — ${t(lang, 'ckptFailed')}: ${info.model.load_error}` : ''}`] : []),
    ...wsWarnings,
  ], [info, wsWarnings, lang]);
  const onWarnings = useCallback((w: string[]) => setWsWarnings(w), []);

  return (
    <div className={`shell ${railMin ? 'rail-min' : ''}`} data-page={page}>
      <nav className="rail" aria-label={t(lang, 'navMain')}>
        <div className="rail-logo" aria-hidden="true"><Icon name="cube" size={20} /></div>
        {NAV.map((n) => (
          <button key={n.id} type="button" className={`rail-item ${page === n.id ? 'on' : ''}`} data-testid={`nav-${n.id}`}
                  aria-current={page === n.id ? 'page' : undefined} title={t(lang, n.key)} onClick={() => go(n.id)}>
            <Icon name={n.icon} size={20} /><span>{t(lang, n.short)}</span>
          </button>
        ))}
        <span className="grow" />
        <button type="button" className="rail-item" onClick={toggleRail} data-testid="rail-toggle"
                title={t(lang, railMin ? 'navExpand' : 'navCollapse')} aria-label={t(lang, railMin ? 'navExpand' : 'navCollapse')}>
          <Icon name={railMin ? 'right' : 'left'} size={18} /><span>{t(lang, railMin ? 'navExpand' : 'navCollapseShort')}</span>
        </button>
      </nav>

      <header className="topbar">
        <h1>{t(lang, 'title')}</h1>
        <span className="topbar-page">{t(lang, NAV.find((n) => n.id === page)!.key)}</span>
        <span className="grow" />
        {info && (
          <span className={`chip ${info.model.untrained ? 'warn' : ''}`} title={t(lang, 'modelChip')} data-testid="model-chip">
            <Icon name="cpu" size={13} />
            <span>{info.model.model_type ? `${info.model.model_type} · ` : ''}{info.model.run ? `${info.model.run} · ` : ''}{(info.model.n_params / 1e6).toFixed(2)} M · {info.model.device}</span>
            {info.model.untrained && <strong>{t(lang, 'untrainedChip')}</strong>}
          </span>
        )}
        {warnings.length > 0 && (
          <Menu icon="warn" label={t(lang, 'warningsTitle')} text={String(warnings.length)} align="end" tone="warn" testid="warnings-chip">
            <div className="warn-list">
              <div className="warn-title">{t(lang, 'warningsTitle')}</div>
              {warnings.map((w) => <div key={w} className="warn-item"><Icon name="warn" size={14} /><span>{w}</span></div>)}
            </div>
          </Menu>
        )}
        <IconButton icon="keyboard" label={t(lang, 'helpTip')} onClick={() => setHelp(true)} testid="help" />
        <button type="button" className="ibtn with-text" onClick={() => changeLang(lang === 'tr' ? 'en' : 'tr')} title={t(lang, 'langToggle')} data-testid="lang">
          {lang === 'tr' ? 'EN' : 'TR'}
        </button>
        <IconButton icon={theme === 'light' ? 'moon' : 'sun'} label={t(lang, 'themeToggle')} onClick={() => changeTheme(theme === 'light' ? 'dark' : 'light')} testid="theme" />
      </header>

      <main className="content">
        <Workspace ref={ws} lang={lang} theme={theme} info={info} hidden={page !== 'work'} modelKey={`${info?.model.run ?? ''}|${info?.model.checkpoint ?? ''}`} onWarnings={onWarnings} />
        {page === 'data' && <div className="page-host"><DatasetPage lang={lang} onOpen={openItem} /></div>}
        {seen.has('gen') && <div className="page-host" hidden={page !== 'gen'}><GeneratePage lang={lang} onShowDatasets={() => go('data')} /></div>}
        {seen.has('train') && <div className="page-host" hidden={page !== 'train'}>
          <TrainPage lang={lang} model={info?.model ?? null} onModel={() => { api.info().then(setInfo).catch(() => undefined); }} /></div>}
        {seen.has('truba') && <div className="page-host" hidden={page !== 'truba'}><TrubaPage lang={lang} /></div>}
      </main>

      <ShortcutHelp lang={lang} open={help} onClose={() => setHelp(false)} />

      <nav className="bottomnav" aria-label={t(lang, 'navMain')}>
        {NAV.map((n) => (
          <button key={n.id} type="button" className={page === n.id ? 'on' : ''} aria-current={page === n.id ? 'page' : undefined}
                  data-testid={`bnav-${n.id}`} onClick={() => go(n.id)}>
            <Icon name={n.icon} size={20} /><span>{t(lang, n.short)}</span>
          </button>
        ))}
      </nav>
    </div>
  );
}
