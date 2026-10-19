// Unit checks for the on-phone quick model (no browser needed): node --test tests/browser/
import test from 'node:test';
import assert from 'node:assert/strict';
import { depthFrameToWorld, VoxelFusion, quickModelPly } from '../../app/static/record/quick-model.js';

// Camera at eye height looking down -Z at a wall 2 m away; fx = fy = 500 px on a 640x480 image.
const frame = (wall = 2) => {
  const depth = new Uint16Array(16 * 12).fill(wall * 1000);
  // readPixels-style snapshot: rows bottom-up, so the first rows are the bottom (blue) of the image.
  const rgb = new Uint8Array(8 * 6 * 3);
  for (let row = 0; row < 6; row++) for (let col = 0; col < 8; col++) rgb.set(row < 3 ? [0, 0, 255] : [255, 0, 0], (row * 8 + col) * 3);
  return {
    bytes: new Uint8Array(depth.buffer), width: 16, height: 12, rawToMeters: 0.001,
    normDepthFromView: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    pose: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0.5, 1.4, 0, 1],
    p0: 500 / 320, p5: 500 / 240, p8: 0, p9: 0,
    color: { width: 8, height: 6, rgb },
  };
};

test('depth back-projects onto the wall, in the world frame', () => {
  const { positions } = depthFrameToWorld(frame(), { stride: 1 });
  assert.equal(positions.length, 16 * 12 * 3);
  for (let i = 0; i < positions.length; i += 3) assert.ok(Math.abs(positions[i + 2] + 2) < 1e-5);
  // Field of view: half-width 2 m * 320 / 500 = 1.28 m around the camera's x = 0.5.
  const xs = positions.filter((_, i) => i % 3 === 0);
  assert.ok(Math.min(...xs) > 0.5 - 1.28 && Math.max(...xs) < 0.5 + 1.28);
});

test('top of the image colours points above eye height', () => {
  const { positions, colors } = depthFrameToWorld(frame(), { stride: 1 });
  for (let i = 0; i < positions.length / 3; i++) {
    const above = positions[i * 3 + 1] > 1.4;
    assert.equal(colors[i * 3] > colors[i * 3 + 2], above);
  }
});

test('out-of-range depth is dropped and fusion averages repeated views', () => {
  assert.equal(depthFrameToWorld(frame(9)).positions.length, 0);
  const fusion = new VoxelFusion(0.05);
  const a = depthFrameToWorld(frame(), { stride: 1 });
  fusion.add(a.positions, a.colors);
  fusion.add(a.positions, a.colors);
  const fused = fusion.result(2);
  assert.ok(fused.positions.length > 0 && fused.positions.length <= a.positions.length);
  const ply = quickModelPly(fused);
  assert.ok(ply.size > 15 * (fused.positions.length / 3));
});
