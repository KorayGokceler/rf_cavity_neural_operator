import { useState } from 'react';
import type { Features, Geometry } from '../api';
import { int, sci } from '../format';
import { t, type Lang } from '../i18n';
import { Collapsible } from './ui';

interface Props { lang: Lang; geometry: Geometry; features: Features | null }

export default function GeometryInfo({ lang, geometry, features }: Props) {
  const [open, setOpen] = useState(false);
  const src = geometry.source as Record<string, unknown>;
  const params = Object.entries(geometry.params ?? {});
  const facts: [string, string][] = [
    ...(src.dataset ? [[t(lang, 'dataset'), `${src.dataset}`] as [string, string]] : []),
    ...(src.file ? [['file', `${src.file}`] as [string, string]] : []),
    ...(src.name ? [['file', `${src.name}`] as [string, string]] : []),
    ...(src.family ? [[t(lang, 'family'), `${src.family}`] as [string, string]] : []),
    ...(src.id !== undefined ? [['id', `${src.id}`] as [string, string]] : []),
    ...(features ? [['N0 DOF', int(features.n_edges)] as [string, string],
                    ['scale', `${features.scale_mm.toFixed(1)} mm`] as [string, string],
                    ['torsion max', sci(features.torsion_max)] as [string, string]] : []),
    ...(geometry.truth?.f_next ? [[t(lang, 'fNext'), `${geometry.truth.f_next.toFixed(4)} GHz`] as [string, string]] : []),
  ];
  return (
    <Collapsible title={t(lang, 'geomInfo')} className="panel-section">
      <dl className="facts">
        {facts.map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}
      </dl>
      {params.length > 0 && (
        <>
          <button type="button" className="btn ghost small" onClick={() => setOpen(!open)}>{open ? '▾' : '▸'} {t(lang, 'params')} ({params.length})</button>
          {open && (
            <dl className="facts params">
              {params.map(([k, v]) => <div key={k}><dt>{k}</dt><dd className="num">{sci(v, 4)}</dd></div>)}
            </dl>
          )}
        </>
      )}
    </Collapsible>
  );
}
