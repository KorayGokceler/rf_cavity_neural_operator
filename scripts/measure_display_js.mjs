// Browser-side cost of the field display (web/src/fieldlab.ts, run by node: V8 as in the browser):
//   node scripts/measure_display_js.mjs payload.json     (payload: GET /api/predictions/{id}/fields)
// decode of the float16 fields, slicePlane + sampleSlice per slider position, rasterize + sampleGrid (3D volume).
import { readFileSync } from 'node:fs';
import { decodeModes, rasterize, sampleGrid, sampleSlice, slicePlane } from '../web/src/fieldlab.ts';

const j = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const bin = (s) => { const b = Buffer.from(s, 'base64'); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const ms = (f) => { const t = performance.now(); const r = f(); return [performance.now() - t, r]; };
const med = (a) => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)];

const K = j.n_modes, nv = j.n_nodes ?? j.n_vertices;
const nodes = new Float32Array(bin(j.nodes_b64)), tets = new Uint32Array(bin(j.tets_b64));
const [tDecode, E] = ms(() => decodeModes(new Uint16Array(bin(j.E_b64)), K, nv, j.E_max));
const b = { K, nv, nodes, tets, vid: new Uint32Array(0), E, H: E, Emax: j.E_max, Hmax: j.E_max, f: j.f_GHz, axis: { z: [], Ez: [] } };
const F = E.subarray(0, 3 * nv);
const lo = [0, 1, 2].map((c) => { let m = Infinity; for (let i = c; i < nodes.length; i += 3) m = Math.min(m, nodes[i]); return m; });
const hi = [0, 1, 2].map((c) => { let m = -Infinity; for (let i = c; i < nodes.length; i += 3) m = Math.max(m, nodes[i]); return m; });
const out = { n_nodes: nv, n_cells: tets.length / 4, decode_E_ms: +tDecode.toFixed(1) };
for (const axis of [0, 1, 2]) {
  const slice = [], samp = [];
  for (let s = 1; s <= 19; s++) {
    const pos = lo[axis] + (hi[axis] - lo[axis]) * s / 20;
    const [t1, sl] = ms(() => slicePlane(b, axis, pos));
    const [t2] = ms(() => sampleSlice(sl, F));
    slice.push(t1); samp.push(t2);
  }
  out[`slice_axis${axis}_ms`] = { median: +med(slice).toFixed(1), max: +Math.max(...slice).toFixed(1), sample_median: +med(samp).toFixed(2) };
}
const [tRaster, g] = ms(() => rasterize(b, 80000));
const [tSample] = ms(() => sampleGrid(g, F));
out.rasterize_ms = +tRaster.toFixed(1); out.sampleGrid_ms = +tSample.toFixed(1); out.grid_inside = g.idx.length;
console.log(JSON.stringify(out, null, 1));
