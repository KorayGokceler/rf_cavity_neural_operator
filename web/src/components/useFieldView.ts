// What one viewer pane shows: the cut / 3D grid, the colour scalars, the arrows and the colour range of a
// (solution, mode, E/H) at the current phase. The workspace calls it once per pane (twice in the split view).
// Cuts and grids are cached per mesh (workspace.ts), so a second pane on the same mesh adds only the sampling.
import { useMemo } from 'react';
import { fillGrid, gather, magnitude, modeField, sampleArrows, sampleGrid, sampleSlice, scaled, sliceArrows, thinGrid,
         gridPoints, type FieldBundle, type Slice } from '../fieldlab';
import { CUT_ARROWS, MAX_ARROWS, arrowBase, cutOf, gridOf, type Display, type Source, type View3D } from '../workspace';
import { phaseFactor, type Arrows, type VolumeScalars } from './Viewer3D';
import type { FeatureView } from '../api';

export interface PaneInput {
  display: Display; bundle: FieldBundle | null; refBundle: FieldBundle | null; mode: number; field: 'E' | 'H';
  view3d: View3D; ai: 0 | 1 | 2; cutPos: number; arrowsOn: boolean; phase: number; source: Source;
  maxOverride: number | null;           // locked / manual colour maximum (null: the mode's own)
  logDecades: number;                   // 0: linear colour scale
  fView?: FeatureView | null; fSurf?: Float32Array | null;     // the input-feature view (primary pane only)
}

/** log10 of the values, floored one decade below the shown range (zero and negative values land there). */
function logOf(x: Float32Array, hi: number, decades: number): Float32Array {
  const floor = hi - decades - 1;
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) out[i] = x[i] > 0 ? Math.max(Math.log10(x[i]), floor) : floor;
  return out;
}

export function useFieldView(p: PaneInput) {
  const { bundle, refBundle, mode, field, view3d, ai, cutPos, arrowsOn, phase, source, display, logDecades } = p;
  const isVol = view3d === 'volume';
  const onField = display === 'field' && !!bundle && mode < bundle.K;
  const F = useMemo(() => (onField && bundle ? modeField(bundle, field, mode) : null), [onField, bundle, field, mode]);
  const cut = useMemo<Slice | null>(() => (onField && bundle && !isVol ? cutOf(bundle, ai, cutPos) : null),
                                     [onField, bundle, isVol, ai, cutPos]);
  const grid = useMemo(() => (onField && bundle && isVol ? gridOf(bundle) : null), [onField, bundle, isVol]);
  const cutV = useMemo(() => (cut && F ? sampleSlice(cut, F) : null), [cut, F]);
  const gridV = useMemo(() => (grid && F ? sampleGrid(grid, F) : null), [grid, F]);
  const planeAmp = useMemo(() => (cutV ? magnitude(cutV) : null), [cutV]);
  const surfAmp = useMemo(() => (bundle && F ? magnitude(gather(F, bundle.vid)) : null), [bundle, F]);
  const volAmp = useMemo(() => (grid && gridV ? fillGrid(grid, magnitude(gridV)) : null), [grid, gridV]);
  const cutSel = useMemo(() => (cut && arrowsOn ? sliceArrows(cut, CUT_ARROWS) : null), [cut, arrowsOn]);
  const gridSel = useMemo(() => {
    if (!grid || !arrowsOn) return null;
    const { sel, step } = thinGrid(grid, MAX_ARROWS);
    return { sel, step, points: gridPoints(grid, sel) };
  }, [grid, arrowsOn]);
  const arrows0 = useMemo(() => {
    if (gridSel && gridV && grid) return arrowBase(gridSel.points, gather(gridV, gridSel.sel), 1.3 * gridSel.step * grid.spacing);
    if (cutSel && cutV) return arrowBase(cutSel.points, sampleArrows(cutV, cutSel.ids, cutSel.w), 1.3 * cutSel.step);
    return null;
  }, [gridSel, gridV, grid, cutSel, cutV]);
  const a = phaseFactor(field, phase);
  const arrows = useMemo<Arrows | null>(() => (arrows0 ? { ...arrows0, vectors: scaled(arrows0.vectors, a) } : null), [arrows0, a]);

  const scaleOf = (b: FieldBundle | null, f: 'E' | 'H') => (b && mode < b.K ? (f === 'E' ? b.Emax : b.Hmax)[mode] : null);
  const lim = scaleOf(refBundle, field) ?? scaleOf(bundle, field) ?? 1;      // differences on the FE scale
  const max = p.maxOverride && p.maxOverride > 0 ? p.maxOverride : lim;
  const hi = Math.log10(max);

  // log colour scale: the scalars are log10 of the amplitude (the phase gain only shifts the colour range)
  const planeS = useMemo(() => (planeAmp && logDecades ? logOf(planeAmp, hi, logDecades) : planeAmp), [planeAmp, logDecades, hi]);
  const surfS = useMemo(() => (surfAmp && logDecades ? logOf(surfAmp, hi, logDecades) : surfAmp), [surfAmp, logDecades, hi]);
  const volume = useMemo<VolumeScalars | null>(() => {
    if (!grid || !volAmp) return null;
    return { dims: grid.dims, origin: grid.origin, spacing: grid.spacing,
             scalars: logDecades ? logOf(volAmp, hi, logDecades) : volAmp };
  }, [grid, volAmp, logDecades, hi]);

  const view = useMemo(() => {
    if (display === 'feature') {
      const fv = p.fView;
      return { plane: fv?.plane ?? null, planeS: fv?.plane?.scalars ?? null, surfS: p.fSurf ?? null,
               volume: null as VolumeScalars | null, arrows: null as Arrows | null,
               range: (fv?.range ?? [0, 1]) as [number, number], phys: (fv?.range ?? [0, 1]) as [number, number],
               gain: 1, signed: fv?.signed ?? false, label: fv?.legend ?? '', unit: '', neutral: fv?.neutral ?? false,
               cells: fv?.cells ?? null, segments: fv?.segments ?? [], log: 0 };
    }
    // colour = |E| or |H| (the scalar field) at the phase: the amplitude times the gain; the arrows carry the direction
    const lg = logDecades;
    return {
      plane: isVol ? null : cut, planeS: isVol ? null : planeS, surfS, volume, arrows,
      range: (lg ? [hi - lg, hi] : [0, max]) as [number, number],
      phys: (lg ? [max * 10 ** -lg, max] : [0, max]) as [number, number],
      gain: Math.abs(a), signed: false, label: `|${source === 'diff' ? 'Δ' : ''}${field}|, U = 1 J`,
      unit: field === 'E' ? 'V/m' : 'A/m', neutral: false, cells: null, segments: [], log: lg,
    };
  }, [display, p.fView, p.fSurf, isVol, cut, planeS, surfS, volume, arrows, max, hi, logDecades, a, source, field]);

  return { onField, view, cut, planeAmp, a, lim, max };
}
