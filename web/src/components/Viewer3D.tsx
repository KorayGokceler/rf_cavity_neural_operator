// vtk.js view: cavity surface (ghost / coloured / hidden) + a coloured cut plane.
// The caller passes per-point scalars (field at a phase, or a model input feature) and the range.
import { useEffect, useRef } from 'react';
import '@kitware/vtk.js/Rendering/Profiles/Geometry';
import vtkGenericRenderWindow from '@kitware/vtk.js/Rendering/Misc/GenericRenderWindow';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkPolyData from '@kitware/vtk.js/Common/DataModel/PolyData';
import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray';
import vtkColorTransferFunction from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction';
import { divergingStops, hexToRgb, sequentialStops, type Theme } from '../colors';
import type { Surface } from '../api';

export type Comp = 'abs' | 'x' | 'y' | 'z';
export type SurfaceMode = 'ghost' | 'field' | 'mesh' | 'hidden';

export interface PlaneGeom { points: Float32Array; triangles: Uint32Array }

interface Props {
  surface: Surface | null;
  surfaceScalars: Float32Array | null;     // per surface point (coloured surface mode)
  plane: PlaneGeom | null;
  planeScalars: Float32Array | null;       // per plane point
  range: [number, number];
  signed: boolean;                          // diverging map around 0
  surfaceMode: SurfaceMode;
  theme: Theme;
  viewAxis: 'x' | 'y' | 'z';               // look along the cut normal
}

/** Scalar per point of a [P,3] vector field at the given phase: E(t) = E cos φ, H(t) = H sin φ. */
export function fieldScalars(F: Float32Array, field: 'E' | 'H', comp: Comp, phaseDeg: number): Float32Array {
  const ph = (phaseDeg * Math.PI) / 180;
  const a = field === 'E' ? Math.cos(ph) : Math.sin(ph);
  const n = F.length / 3;
  const out = new Float32Array(n);
  const c = comp === 'x' ? 0 : comp === 'y' ? 1 : 2;
  for (let i = 0; i < n; i++) {
    const x = F[3 * i], y = F[3 * i + 1], z = F[3 * i + 2];
    out[i] = comp === 'abs' ? Math.abs(a) * Math.hypot(x, y, z) : a * F[3 * i + c];
  }
  return out;
}

function polys(tri: Uint32Array): Uint32Array {
  const n = tri.length / 3;
  const out = new Uint32Array(4 * n);
  for (let i = 0; i < n; i++) {
    out[4 * i] = 3; out[4 * i + 1] = tri[3 * i]; out[4 * i + 2] = tri[3 * i + 1]; out[4 * i + 3] = tri[3 * i + 2];
  }
  return out;
}

function cssVar(name: string, fallback: string) {
  const v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return v || fallback;
}

/* eslint-disable @typescript-eslint/no-explicit-any */
export default function Viewer3D(p: Props) {
  const host = useRef<HTMLDivElement>(null);
  const vtk = useRef<any>(null);

  useEffect(() => {
    const grw = vtkGenericRenderWindow.newInstance({ background: hexToRgb(cssVar('--surface-1', '#fcfcfb')) });
    grw.setContainer(host.current!);
    grw.resize();
    const renderer = grw.getRenderer();
    const mk = () => {
      const pd = vtkPolyData.newInstance();
      const mapper = vtkMapper.newInstance({ interpolateScalarsBeforeMapping: true });
      mapper.setInputData(pd);
      const actor = vtkActor.newInstance();
      actor.setMapper(mapper);
      renderer.addActor(actor);
      return { pd, mapper, actor };
    };
    vtk.current = { grw, renderer, rw: grw.getRenderWindow(), surf: mk(), plane: mk(),
                    ctf: vtkColorTransferFunction.newInstance() };
    const ro = new ResizeObserver(() => { grw.resize(); grw.getRenderWindow().render(); });
    ro.observe(host.current!);
    return () => { ro.disconnect(); grw.delete(); vtk.current = null; };
  }, []);

  // surface geometry + camera along the cut normal (slightly oblique)
  useEffect(() => {
    const v = vtk.current;
    if (!v) return;
    if (p.surface) {
      v.surf.pd.getPoints().setData(p.surface.points, 3);
      v.surf.pd.getPolys().setData(polys(p.surface.triangles));
      v.surf.pd.modified();
    }
    v.surf.actor.setVisibility(!!p.surface);
    const cam = v.renderer.getActiveCamera();
    const b = v.surf.pd.getBounds();
    const c = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2];
    const dir = p.viewAxis === 'x' ? [1, 0.25, 0.2] : p.viewAxis === 'y' ? [0.25, -1, 0.2] : [0.25, 0.2, 1];
    cam.setFocalPoint(c[0], c[1], c[2]);
    cam.setPosition(c[0] + dir[0], c[1] + dir[1], c[2] + dir[2]);
    cam.setViewUp(...(p.viewAxis === 'z' ? [0, 1, 0] : [0, 0, 1]) as [number, number, number]);
    v.renderer.resetCamera();
    v.rw.render();
  }, [p.surface, p.viewAxis]);

  useEffect(() => {
    const v = vtk.current;
    if (!v) return;
    if (p.plane && p.plane.triangles.length) {
      v.plane.pd.getPoints().setData(p.plane.points, 3);
      v.plane.pd.getPolys().setData(polys(p.plane.triangles));
      v.plane.pd.modified();
    }
    v.rw.render();
  }, [p.plane]);

  useEffect(() => {
    const v = vtk.current;
    if (!v) return;
    const [lo, hi] = p.range[1] > p.range[0] ? p.range : [0, 1];
    v.ctf.removeAllPoints();
    if (p.signed) {
      const m = Math.max(Math.abs(lo), Math.abs(hi));
      divergingStops(p.theme).forEach(([x, c]) => v.ctf.addRGBPoint(x * m, ...hexToRgb(c)));
    } else {
      sequentialStops(p.theme).forEach(([x, c]) => v.ctf.addRGBPoint(lo + x * (hi - lo), ...hexToRgb(c)));
    }
    const range: [number, number] = p.signed ? [-Math.max(Math.abs(lo), Math.abs(hi)), Math.max(Math.abs(lo), Math.abs(hi))] : [lo, hi];
    const paint = (o: any, vals: Float32Array | null) => {
      if (!vals) { o.mapper.setScalarVisibility(false); return; }
      o.pd.getPointData().setScalars(vtkDataArray.newInstance({ name: 's', values: vals }));
      o.mapper.setLookupTable(v.ctf);
      o.mapper.setScalarRange(...range);
      o.mapper.setScalarVisibility(true);
      o.pd.modified();
    };
    const planeOn = !!(p.plane && p.plane.triangles.length && p.planeScalars) && p.surfaceMode !== 'mesh';
    if (planeOn) paint(v.plane, p.planeScalars);
    v.plane.actor.setVisibility(planeOn);

    const sp = v.surf.actor.getProperty();
    sp.setRepresentationToSurface();
    sp.setEdgeVisibility(false);
    if (p.surfaceMode === 'mesh') {
      v.surf.mapper.setScalarVisibility(false);
      sp.setColor(...hexToRgb(cssVar('--surface-2', '#f3f2ee')));
      sp.setEdgeColor(...hexToRgb(cssVar('--text-secondary', '#52514e')));
      sp.setEdgeVisibility(true);
      sp.setOpacity(1.0);
      v.surf.actor.setVisibility(!!p.surface);
    } else if (p.surfaceMode === 'field' && p.surfaceScalars) {
      paint(v.surf, p.surfaceScalars);
      sp.setOpacity(1.0);
      v.surf.actor.setVisibility(true);
    } else {
      v.surf.mapper.setScalarVisibility(false);
      sp.setColor(...hexToRgb(cssVar('--muted', '#898781')));
      sp.setOpacity(0.18);
      v.surf.actor.setVisibility(p.surfaceMode !== 'hidden' && !!p.surface);
    }
    v.rw.render();
  }, [p.plane, p.planeScalars, p.surfaceScalars, p.range, p.signed, p.surfaceMode, p.theme, p.surface]);

  return <div className="viewer" ref={host} aria-label="3D view" />;
}
