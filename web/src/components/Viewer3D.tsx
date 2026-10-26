// vtk.js view: cavity surface (ghost / coloured / hidden), a coloured cut plane or the field magnitude in the
// whole volume (volume rendering), and optional arrows for the field direction.
// The caller passes per-point scalars (the amplitude |E| / |H|, or a model input feature), the range of the shown
// values and a gain (the phase factor |cos φ| / |sin φ|): a phase step only rescales the colour maps, no data
// is re-uploaded.
import { useEffect, useImperativeHandle, useRef, useState, type Ref } from 'react';
import '@kitware/vtk.js/Rendering/Profiles/Geometry';
import '@kitware/vtk.js/Rendering/Profiles/Volume';
import '@kitware/vtk.js/Rendering/Profiles/Glyph';
import vtkGenericRenderWindow from '@kitware/vtk.js/Rendering/Misc/GenericRenderWindow';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkGlyph3DMapper from '@kitware/vtk.js/Rendering/Core/Glyph3DMapper';
import vtkVolume from '@kitware/vtk.js/Rendering/Core/Volume';
import vtkVolumeMapper from '@kitware/vtk.js/Rendering/Core/VolumeMapper';
import vtkArrowSource from '@kitware/vtk.js/Filters/Sources/ArrowSource';
import vtkSphereSource from '@kitware/vtk.js/Filters/Sources/SphereSource';
import vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData';
import vtkPolyData from '@kitware/vtk.js/Common/DataModel/PolyData';
import vtkDataArray from '@kitware/vtk.js/Common/Core/DataArray';
import vtkColorTransferFunction from '@kitware/vtk.js/Rendering/Core/ColorTransferFunction';
import vtkPiecewiseFunction from '@kitware/vtk.js/Common/DataModel/PiecewiseFunction';
import { divergingStops, hexToRgb, sequentialStops, type Theme } from '../colors';
import type { ScalarMesh, Segments, Surface } from '../api';
import type { Key } from '../i18n';

export type SurfaceMode = 'ghost' | 'field' | 'mesh' | 'hidden';

export interface PlaneGeom { points: Float32Array; triangles: Uint32Array }
/** Scalars on a regular grid (x fastest), mm. */
export interface VolumeScalars { dims: [number, number, number]; origin: [number, number, number]; spacing: number; scalars: Float32Array }
/** Arrow glyphs: positions [P,3] mm, field vectors [P,3]; the largest |vector| (vmax) is drawn `length` mm long. */
export interface Arrows { points: Float32Array; vectors: Float32Array; vmax: number; length: number }

interface Props {
  ref?: Ref<ViewerHandle>;
  surface: Surface | null;
  surfaceScalars: Float32Array | null;     // per surface point (coloured surface mode)
  plane: PlaneGeom | null;
  planeScalars: Float32Array | null;       // per plane point
  volume?: VolumeScalars | null;            // shown instead of the plane (3D view)
  arrows?: Arrows | null;
  range: [number, number];                  // of the shown values gain · scalars
  gain?: number;                            // shown = gain · scalars (default 1)
  signed: boolean;                          // diverging map around 0
  surfaceMode: SurfaceMode;
  theme: Theme;
  viewAxis: 'x' | 'y' | 'z';               // look along the cut normal
  oblique?: boolean;                        // a more oblique camera (3D volume: see the depth)
  neutralPlane?: boolean;                   // plane drawn in one neutral tone (overlays carry the content)
  cells?: ScalarMesh | null;                // e.g. tets cut by the plane, edges drawn
  segments?: Segments[];                    // line overlays: grid / axis / marker / iso / arrow
  logScale?: boolean;                       // range / scalars are log10 values (the gain shifts the range by log10 g)
  pickMode?: boolean;                       // a click (not a drag) reports a camera ray
  onPick?: (ray: Ray) => void;
  marker?: [number, number, number] | null; // a probe point [mm]
  topInset?: number;                        // CSS px the overlay toolbar covers at the top: the framing leaves them free
}

/** A ray through a screen point: origin + t · dir (mm). */
export interface Ray { origin: [number, number, number]; dir: [number, number, number] }

export const VOL_RAMP_START = 0.35;                  // volume colours: the sequential ramp from 35 % on

const ROLE_STYLE: Record<string, { color: string; width: number; opacity: number }> = {
  grid: { color: '--text-secondary', width: 1, opacity: 0.55 },
  axis: { color: '--text-primary', width: 2, opacity: 0.9 },
  marker: { color: '--critical', width: 3, opacity: 1 },
  iso: { color: '--text-primary', width: 1, opacity: 0.45 },
  arrow: { color: '--text-primary', width: 1, opacity: 0.85 },
};

function lineCells(n: number): Uint32Array {          // n segments: [2, 2i, 2i+1] …
  const out = new Uint32Array(3 * n);
  for (let i = 0; i < n; i++) { out[3 * i] = 2; out[3 * i + 1] = 2 * i; out[3 * i + 2] = 2 * i + 1; }
  return out;
}

/** The sequential stops from VOL_RAMP_START on, rescaled to [0, 1] (the volume's colours and its legend). */
export function volumeStops(theme: Theme): [number, string][] {
  return sequentialStops(theme).filter(([x]) => x >= VOL_RAMP_START)
    .map(([x, c]) => [(x - VOL_RAMP_START) / (1 - VOL_RAMP_START), c] as [number, string]);
}

/** Time factor of a standing wave at the phase: E(t) = E cos φ, H(t) = H sin φ. */
export function phaseFactor(field: 'E' | 'H', phaseDeg: number): number {
  const ph = (phaseDeg * Math.PI) / 180;
  return field === 'E' ? Math.cos(ph) : Math.sin(ph);
}

/** The scalar range the colour maps span: the shown range over the gain (an empty range → [0, 1]). */
function dataRange(range: [number, number], gain = 1, log = false): [number, number] {
  const [lo, hi] = range[1] > range[0] ? range : [0, 1];
  const g = Math.max(Math.abs(gain), 1e-6);              // gain 0 (E at 90°): every value at the low end
  if (log) { const d = Math.log10(g); return [lo - d, hi - d]; }       // log10 of the scalars: a gain is a shift
  return [lo / g, hi / g];
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


// Camera presets: `dir` = from where we look (the camera sits on that side of the geometry), `up` = view-up.
// The side views keep the beam axis z vertical, the ±Z views have y up.
type Vec3 = [number, number, number];
export interface Preset { id: string; label: string; dir: Vec3; up: Vec3; tip: Key }
export const PRESETS: Preset[] = [
  { id: 'px', label: '+X', dir: [1, 0, 0], up: [0, 0, 1], tip: 'viewPX' },
  { id: 'nx', label: '−X', dir: [-1, 0, 0], up: [0, 0, 1], tip: 'viewNX' },
  { id: 'py', label: '+Y', dir: [0, 1, 0], up: [0, 0, 1], tip: 'viewPY' },
  { id: 'ny', label: '−Y', dir: [0, -1, 0], up: [0, 0, 1], tip: 'viewNY' },
  { id: 'pz', label: '+Z', dir: [0, 0, 1], up: [0, 1, 0], tip: 'viewPZ' },
  { id: 'nz', label: '−Z', dir: [0, 0, -1], up: [0, 1, 0], tip: 'viewNZ' },
  { id: 'iso', label: 'Iso', dir: [1, -1, 1], up: [0, 0, 1], tip: 'viewIso' },
];

/* eslint-disable @typescript-eslint/no-explicit-any */
/** Camera at `center + dir`, looking at the centre of the surface, fitted to its bounds. */
function lookFrom(v: any, dir: Vec3, up: Vec3) {
  v.frame = { dir, up };                                  // re-fitted when the canvas is resized (until the user moves the camera)
  v.touched = false;
  const b = v.surf.pd.getBounds();
  const c = [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2];
  const cam = v.renderer.getActiveCamera();
  cam.setFocalPoint(c[0], c[1], c[2]);
  cam.setPosition(c[0] + dir[0], c[1] + dir[1], c[2] + dir[2]);
  cam.setViewUp(...up);
  v.renderer.resetCamera(b);
  cam.orthogonalizeViewUp();
  // a portrait canvas (phone): the vertical view angle fits the bounds, the width would crop them, so back off
  const [w, h] = v.grw.getApiSpecificRenderWindow().getSize();
  if (w > 0 && h > 0 && w < h) { cam.dolly(w / h); v.renderer.resetCameraClippingRange(b); }
  // the toolbar floats over the top of the canvas: shift the image down by half of it (window centre, NDC)
  const hc = v.host?.clientHeight || 0;
  cam.setWindowCenter(0, hc > 0 ? Math.min(0.3, (2 * (v.inset ?? 0)) / hc) : 0);
  v.render();
}

/** The view the viewer opens with: along the cut normal, slightly oblique (more so for the 3D volume). */
function homeView(v: any, axis: 'x' | 'y' | 'z', oblique: boolean) {
  const t = oblique ? 0.7 : 0.25;
  const dir: Vec3 = axis === 'x' ? [1, t, 0.8 * t] : axis === 'y' ? [t, -1, 0.8 * t] : [t, 0.8 * t, 1];
  lookFrom(v, dir, axis === 'z' ? [0, 1, 0] : [0, 0, 1]);
}

const TRIAD: { label: string; e: Vec3; color: string }[] = [
  { label: 'x', e: [1, 0, 0], color: 'var(--critical)' },
  { label: 'y', e: [0, 1, 0], color: 'var(--good)' },
  { label: 'z', e: [0, 0, 1], color: 'var(--accent)' },
];

/** Small x / y / z triad that follows the camera (corner of the viewer). */
function AxesTriad({ camera }: { camera: any }) {
  const [, tick] = useState(0);
  useEffect(() => {
    if (!camera) return;
    const sub = camera.onModified(() => tick((n) => n + 1));
    return () => sub.unsubscribe();
  }, [camera]);
  if (!camera) return null;
  const d: Vec3 = camera.getDirectionOfProjection();
  const u: Vec3 = camera.getViewUp();
  const r: Vec3 = [d[1] * u[2] - d[2] * u[1], d[2] * u[0] - d[0] * u[2], d[0] * u[1] - d[1] * u[0]];   // screen right
  const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const L = 24;
  return (
    <svg className="triad" width="72" height="72" viewBox="-36 -36 72 72" aria-hidden="true">
      {TRIAD.map(({ label, e, color }) => {
        const x = L * dot(e, r), y = -L * dot(e, u), toward = dot(e, d) < 0;   // toward the viewer
        return (
          <g key={label} opacity={toward ? 1 : 0.45}>
            <line x1={0} y1={0} x2={x} y2={y} stroke={color} strokeWidth={2} strokeLinecap="round" />
            <text x={x * 1.3} y={y * 1.3} fill={color} fontSize="11" fontWeight="600" textAnchor="middle"
                  dominantBaseline="central">{label}</text>
          </g>
        );
      })}
    </svg>
  );
}
/* eslint-enable @typescript-eslint/no-explicit-any */

/* eslint-disable @typescript-eslint/no-explicit-any */
// A NEW vtkPolyData per update: vtk.js does not re-upload a polydata whose point / cell arrays were only
// overwritten with setData (the cut plane stayed where it was first drawn).
function setMesh(o: any, points: Float32Array, triangles: Uint32Array) {
  const pd = vtkPolyData.newInstance();
  pd.getPoints().setData(points, 3);
  pd.getPolys().setData(polys(triangles));
  o.mapper.setInputData(pd);
  o.pd = pd;
}

type Vec = [number, number, number];
/** The ray through a client point: a pinhole (or parallel) camera model of vtk's active camera. */
function rayAt(v: any, el: HTMLElement, clientX: number, clientY: number): Ray | null {
  if (!v) return null;
  const r = el.getBoundingClientRect();
  if (!r.width || !r.height) return null;
  const cam = v.renderer.getActiveCamera();
  const wc: [number, number] = cam.getWindowCenter();           // an off-centre window shifts the image plane
  const nx = ((clientX - r.left) / r.width) * 2 - 1 + wc[0], ny = 1 - ((clientY - r.top) / r.height) * 2 + wc[1];
  const pos: Vec = cam.getPosition(), d: Vec = cam.getDirectionOfProjection(), up: Vec = cam.getViewUp();
  const cross = (a: Vec, b: Vec): Vec => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = (a: Vec): Vec => { const n = Math.hypot(...a) || 1; return [a[0] / n, a[1] / n, a[2] / n]; };
  const right = norm(cross(d, up));
  const upn = norm(cross(right, d));
  const aspect = r.width / r.height;
  if (cam.getParallelProjection()) {
    const sc = cam.getParallelScale();
    const o: Vec = [0, 1, 2].map((i) => pos[i] + right[i] * nx * sc * aspect + upn[i] * ny * sc) as Vec;
    return { origin: o, dir: d };
  }
  const th = Math.tan((cam.getViewAngle() * Math.PI) / 360);
  const dir = norm([0, 1, 2].map((i) => d[i] + right[i] * nx * th * aspect + upn[i] * ny * th) as Vec);
  return { origin: pos, dir };
}

/** What the workspace can ask of a viewer: camera presets (the view tools live outside the canvas). */
export interface ViewerHandle {
  home: () => void;
  preset: (id: string) => void;
  camera: () => any;
  syncFrom: (cam: any) => void;               // take another viewer's camera (linked split view)
  render: () => void;
  capture: () => Promise<string>;             // PNG data URL of the canvas, at the device pixel ratio
}

export default function Viewer3D(p: Props) {
  const host = useRef<HTMLDivElement>(null);
  const vtk = useRef<any>(null);
  const [camera, setCamera] = useState<any>(null);
  const pickRef = useRef<{ mode: boolean; fn?: (r: Ray) => void }>({ mode: false });
  pickRef.current = { mode: !!p.pickMode, fn: p.onPick };
  if (vtk.current) vtk.current.inset = p.topInset ?? 0;

  useEffect(() => {
    const grw = vtkGenericRenderWindow.newInstance({ background: hexToRgb(cssVar('--viewer-bg', '#f7f8fa')) });
    grw.setContainer(host.current!);
    if (host.current!.clientWidth > 0) grw.resize();
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
    // volume rendering of |F| on the grid
    const vol = { actor: vtkVolume.newInstance(), mapper: vtkVolumeMapper.newInstance(),
                  ctf: vtkColorTransferFunction.newInstance(), otf: vtkPiecewiseFunction.newInstance() };
    vol.actor.setMapper(vol.mapper);
    vol.actor.getProperty().setRGBTransferFunction(0, vol.ctf);
    vol.actor.getProperty().setScalarOpacity(0, vol.otf);
    vol.actor.getProperty().setInterpolationTypeToLinear();
    vol.actor.getProperty().setShade(false);
    vol.actor.setVisibility(false);
    renderer.addVolume(vol.actor);
    // arrows: one glyph per point, oriented and scaled by the vector
    const arrow = vtkArrowSource.newInstance({ tipResolution: 12, shaftResolution: 8, tipLength: 0.35, tipRadius: 0.14,
                                               shaftRadius: 0.05 });
    const gm = vtkGlyph3DMapper.newInstance({ orientationArray: 'vec', scaleArray: 'vec' });
    gm.setOrientationModeToDirection();
    gm.setScaleModeToScaleByMagnitude();
    gm.setInputConnection(arrow.getOutputPort(), 1);
    gm.setScalarVisibility(false);
    const ga = vtkActor.newInstance();
    ga.setMapper(gm);
    ga.setVisibility(false);
    renderer.addActor(ga);
    // one render per frame: an update runs several of the effects below (data, colour maps, arrows), each asks
    let frame = 0;
    // a hidden page (display: none, e.g. the workspace behind another page) has no size: nothing to draw
    const visible = () => !!host.current && host.current.clientWidth > 0 && host.current.clientHeight > 0;
    const render = () => {
      if (!frame) frame = requestAnimationFrame(() => { frame = 0; if (visible()) grw.getRenderWindow().render(); });
    };
    // probe marker
    const sph = vtkSphereSource.newInstance({ thetaResolution: 24, phiResolution: 16 });
    const sm = vtkMapper.newInstance();
    sm.setInputConnection(sph.getOutputPort());
    const sa = vtkActor.newInstance();
    sa.setMapper(sm);
    sa.getProperty().setColor(...hexToRgb(cssVar('--critical', '#c9302c')));
    sa.getProperty().setAmbient(0.6);
    sa.setVisibility(false);
    renderer.addActor(sa);
    vtk.current = { grw, host: host.current, inset: 0, renderer, render, surf: mk(), plane: mk(), cells: mk(), mk, lines: [],
                    ctf: vtkColorTransferFunction.newInstance(), vol, arrows: { mapper: gm, actor: ga },
                    marker: { source: sph, actor: sa } };
    setCamera(renderer.getActiveCamera());
    const ro = new ResizeObserver(() => {
      if (!visible()) return;
      grw.resize();
      const v = vtk.current;
      if (v?.frame && !v.touched && v.surf.actor.getVisibility()) lookFrom(v, v.frame.dir, v.frame.up);   // keep the fit
      else render();
    });
    ro.observe(host.current!);
    // a click without a drag (the interactor rotates on drags) reports a ray when picking is on
    const el = host.current!;
    let down: { x: number; y: number } | null = null;
    const onDown = (e: PointerEvent) => { down = { x: e.clientX, y: e.clientY }; if (vtk.current) vtk.current.touched = true; };
    const onWheel = () => { if (vtk.current) vtk.current.touched = true; };
    const onUp = (e: PointerEvent) => {
      const d = down; down = null;
      const cb = pickRef.current;
      if (!d || !cb.mode || !cb.fn || Math.hypot(e.clientX - d.x, e.clientY - d.y) > 5) return;
      const ray = rayAt(vtk.current, el, e.clientX, e.clientY);
      if (ray) cb.fn(ray);
    };
    el.addEventListener('pointerdown', onDown);
    el.addEventListener('pointerup', onUp);
    el.addEventListener('wheel', onWheel, { passive: true });
    return () => {
      el.removeEventListener('pointerdown', onDown); el.removeEventListener('pointerup', onUp); el.removeEventListener('wheel', onWheel);
      ro.disconnect(); cancelAnimationFrame(frame); grw.delete(); vtk.current = null;
    };
  }, []);

  useEffect(() => {                                     // background follows the theme
    const v = vtk.current;
    if (!v) return;
    v.renderer.setBackground(...hexToRgb(cssVar('--viewer-bg', '#f7f8fa')));
    v.render();
  }, [p.theme]);

  // surface geometry + camera along the cut normal (slightly oblique)
  useEffect(() => {
    const v = vtk.current;
    if (!v) return;
    if (p.surface) setMesh(v.surf, p.surface.points, p.surface.triangles);
    v.surf.actor.setVisibility(!!p.surface);
    if (p.surface) homeView(v, p.viewAxis, !!p.oblique);
    else v.render();
  }, [p.surface, p.viewAxis, p.oblique]);

  useEffect(() => {
    const v = vtk.current;
    if (!v) return;
    if (p.plane && p.plane.triangles.length) setMesh(v.plane, p.plane.points, p.plane.triangles);
    v.render();
  }, [p.plane]);

  // 3D field magnitude: a new grid per data set; the phase only moves the transfer functions (below)
  useEffect(() => {
    const v = vtk.current;
    if (!v) return;
    const vo = p.volume;
    if (vo) {
      const img = vtkImageData.newInstance();
      img.setDimensions(vo.dims);
      img.setOrigin(vo.origin);
      img.setSpacing([vo.spacing, vo.spacing, vo.spacing]);
      img.getPointData().setScalars(vtkDataArray.newInstance({ name: 'mag', numberOfComponents: 1, values: vo.scalars }));
      v.vol.mapper.setInputData(img);
      v.vol.mapper.setSampleDistance(0.5 * vo.spacing);
      v.vol.actor.getProperty().setScalarOpacityUnitDistance(0, 2 * vo.spacing);
    }
    v.vol.actor.setVisibility(!!vo);
    v.render();
  }, [p.volume]);
  useEffect(() => {
    const v = vtk.current;
    if (!v || !p.volume) return;
    const [lo, hi] = dataRange(p.range, p.gain, p.logScale);
    // the ramp from its mid tone: the near-background light end would make weak fields vanish; the opacity
    // carries the weak end instead
    v.vol.ctf.removeAllPoints();
    volumeStops(p.theme).forEach(([x, c]) => v.vol.ctf.addRGBPoint(lo + x * (hi - lo), ...hexToRgb(c)));
    // weak fields transparent, strong ones opaque: the field's shape inside the cavity
    v.vol.otf.removeAllPoints();
    v.vol.otf.addPoint(lo, 0);
    v.vol.otf.addPoint(lo + 0.1 * (hi - lo), 0);
    v.vol.otf.addPoint(lo + 0.35 * (hi - lo), 0.1);
    v.vol.otf.addPoint(lo + 0.7 * (hi - lo), 0.4);
    v.vol.otf.addPoint(hi, 0.8);
    v.render();
  }, [p.volume, p.range, p.gain, p.theme, p.logScale]);

  // probe marker: a small sphere on the cut plane
  useEffect(() => {
    const v = vtk.current;
    if (!v) return;
    const m = p.marker;
    if (m && p.surface) {
      const b = v.surf.pd.getBounds();
      const diag = Math.hypot(b[1] - b[0], b[3] - b[2], b[5] - b[4]) || 1;
      v.marker.source.setCenter(m);
      v.marker.source.setRadius(0.011 * diag);
      v.marker.actor.getProperty().setColor(...hexToRgb(cssVar('--critical', '#c9302c')));
    }
    v.marker.actor.setVisibility(!!(m && p.surface));
    v.render();
  }, [p.marker, p.surface, p.theme]);

  // arrows (field direction; length ∝ |F|)
  useEffect(() => {
    const v = vtk.current;
    if (!v) return;
    const a = p.arrows;
    if (a && a.points.length) {
      const pd = vtkPolyData.newInstance();
      pd.getPoints().setData(a.points, 3);
      pd.getPointData().addArray(vtkDataArray.newInstance({ name: 'vec', numberOfComponents: 3, values: a.vectors }));
      v.arrows.mapper.setInputData(pd, 0);
      v.arrows.mapper.setScaleFactor(a.vmax > 0 ? a.length / a.vmax : 0);
      const ap = v.arrows.actor.getProperty();
      ap.setColor(...hexToRgb(cssVar('--text-primary', '#0b0b0b')));
      ap.setOpacity(0.9);
    }
    v.arrows.actor.setVisibility(!!(a && a.points.length));
    v.render();
  }, [p.arrows, p.theme]);

  useEffect(() => {
    const v = vtk.current;
    if (!v) return;
    const [lo, hi] = dataRange(p.range, p.gain, p.logScale);
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
      if (o.vals !== vals || o.valsPd !== o.pd) {         // new data (not a phase step): set the scalars
        o.pd.getPointData().setScalars(vtkDataArray.newInstance({ name: 's', values: vals }));
        o.pd.modified();
        o.vals = vals; o.valsPd = o.pd;
      }
      o.mapper.setLookupTable(v.ctf);
      o.mapper.setScalarRange(...range);
      o.mapper.setScalarVisibility(true);
    };
    const planeOn = !p.volume && !!(p.plane && p.plane.triangles.length && (p.planeScalars || p.neutralPlane))
      && p.surfaceMode !== 'mesh';
    const pp = v.plane.actor.getProperty();
    if (planeOn && p.neutralPlane) {
      v.plane.mapper.setScalarVisibility(false);
      pp.setColor(...hexToRgb(cssVar('--surface-2', '#f3f2ee')));
    } else if (planeOn) paint(v.plane, p.planeScalars);
    v.plane.actor.setVisibility(planeOn);
    const cells = p.cells;
    if (cells && cells.triangles.length) {
      setMesh(v.cells, cells.points, cells.triangles);
      paint(v.cells, cells.scalars);
      const cp = v.cells.actor.getProperty();
      cp.setEdgeVisibility(true);
      cp.setEdgeColor(...hexToRgb(cssVar('--text-secondary', '#52514e')));
      v.cells.actor.setVisibility(true);
    } else {
      v.cells.actor.setVisibility(false);
    }

    const sp = v.surf.actor.getProperty();
    sp.setRepresentationToSurface();
    sp.setEdgeVisibility(false);
    // with the volume, only the far walls: the translucent near walls would occlude the volume (depth)
    sp.setFrontfaceCulling(!!p.volume && p.surfaceMode === 'ghost');
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
      sp.setOpacity(p.volume ? 0.25 : 0.18);
      v.surf.actor.setVisibility(p.surfaceMode !== 'hidden' && !!p.surface);
    }
    v.render();
  }, [p.plane, p.planeScalars, p.surfaceScalars, p.range, p.gain, p.signed, p.surfaceMode, p.theme, p.surface, p.cells,
      p.neutralPlane, p.volume, p.logScale]);

  // line overlays (rebuilt when they change)
  useEffect(() => {
    const v = vtk.current;
    if (!v) return;
    v.lines.forEach((o: any) => v.renderer.removeActor(o.actor));
    v.lines = [];
    // lift the lines slightly off the cut plane toward the camera (no z-fighting with the plane)
    const b = v.surf.pd.getBounds();
    const diag = Math.hypot(b[1] - b[0], b[3] - b[2], b[5] - b[4]) || 1;
    const ai = p.viewAxis === 'x' ? 0 : p.viewAxis === 'y' ? 1 : 2;
    const lift = (p.viewAxis === 'y' ? -1 : 1) * 0.004 * diag;
    for (const sg of p.segments ?? []) {
      const n = sg.points.length / 6;
      if (!n) continue;
      const o = v.mk();
      const pts = new Float32Array(sg.points);
      for (let i = ai; i < pts.length; i += 3) pts[i] += lift;
      o.pd.getPoints().setData(pts, 3);
      o.pd.getLines().setData(lineCells(n));
      o.pd.modified();
      const st = ROLE_STYLE[sg.role] ?? ROLE_STYLE.iso;
      const pr = o.actor.getProperty();
      pr.setColor(...hexToRgb(cssVar(st.color, '#0b0b0b')));
      pr.setLineWidth(st.width);
      pr.setOpacity(st.opacity);
      pr.setLighting(false);
      o.mapper.setScalarVisibility(false);
      v.lines.push(o);
    }
    v.render();
  }, [p.segments, p.theme, p.viewAxis]);

  useImperativeHandle(p.ref, () => ({
    home: () => { const v = vtk.current; if (v && p.surface) homeView(v, p.viewAxis, !!p.oblique); },
    preset: (id: string) => {
      const v = vtk.current, pr = PRESETS.find((x) => x.id === id);
      if (v && p.surface && pr) lookFrom(v, pr.dir, pr.up);
    },
    camera: () => vtk.current?.renderer.getActiveCamera(),
    syncFrom: (src: any) => {
      const v = vtk.current;
      if (!v || !src) return;
      const cam = v.renderer.getActiveCamera();
      cam.setPosition(...(src.getPosition() as Vec));
      cam.setFocalPoint(...(src.getFocalPoint() as Vec));
      cam.setViewUp(...(src.getViewUp() as Vec));
      cam.setViewAngle(src.getViewAngle());
      cam.setWindowCenter(...(src.getWindowCenter() as [number, number]));
      if (v.surf.pd.getBounds) v.renderer.resetCameraClippingRange(v.surf.pd.getBounds());
      v.render();
    },
    render: () => vtk.current?.render(),
    capture: () => {
      const v = vtk.current;
      if (!v) return Promise.reject(new Error('no view'));
      const ready = v.grw.getApiSpecificRenderWindow().captureNextImage('image/png') as Promise<string>;
      v.grw.getRenderWindow().render();
      return ready;
    },
  }), [p.surface, p.viewAxis, p.oblique]);
  return (
    <>
      <div className={`viewer ${p.pickMode ? 'pick' : ''}`} ref={host} aria-label="3D view" />
      {p.surface && <AxesTriad camera={camera} />}
    </>
  );
}
