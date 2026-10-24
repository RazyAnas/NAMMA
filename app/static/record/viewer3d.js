// Phone 3D viewer: the quick model (coloured points), the camera path and the floor, then the laptop's
// photoreal splat on top. three.js + OrbitControls; Spark (Gaussian splats) loads only when a splat
// arrives, because it is 2.7 MB. World frame: WebXR local-floor (metres, y up, floor at y = 0).

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

export class Viewer {
  constructor(container) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.renderer.setClearColor(0x07110f);
    container.appendChild(this.renderer.domElement);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(60, 1, 0.02, 200);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.layers = {};
    this.grid = new THREE.GridHelper(20, 40, 0x2f6f62, 0x1a3a33);
    this.scene.add(this.grid);
    this.resize();
    new ResizeObserver(() => this.resize()).observe(container);
    this.renderer.setAnimationLoop(() => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
    });
  }

  resize() {
    const { clientWidth: w, clientHeight: h } = this.container;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  setLayer(name, object) {
    if (this.layers[name]) {
      this.scene.remove(this.layers[name]);
      this.layers[name].traverse?.((o) => {
        o.geometry?.dispose?.();
        o.material?.dispose?.();
      });
      this.layers[name].dispose?.();
    }
    this.layers[name] = object;
    if (object) this.scene.add(object);
  }

  show(name, visible) {
    if (this.layers[name]) this.layers[name].visible = visible;
  }

  setPoints(name, positions, colors, size = 0.025) {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3, true));
    const material = new THREE.PointsMaterial({ size, vertexColors: true, sizeAttenuation: true });
    this.setLayer(name, new THREE.Points(geometry, material));
    return geometry;
  }

  setPath(positions) {
    if (!positions.length) return;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    this.setLayer('path', new THREE.Line(geometry, new THREE.LineBasicMaterial({ color: 0xffc857 })));
  }

  // Frame the camera on a point set: look at its centre from above and behind the walked path.
  frame(positions) {
    if (!positions.length) return;
    const box = new THREE.Box3().setFromBufferAttribute(new THREE.BufferAttribute(positions, 3));
    const centre = box.getCenter(new THREE.Vector3());
    const size = Math.max(1, box.getSize(new THREE.Vector3()).length());
    this.controls.target.copy(centre);
    this.camera.position.set(centre.x + 0.35 * size, centre.y + 0.55 * size, centre.z + 0.55 * size);
    this.camera.near = size / 500;
    this.camera.far = size * 20;
    this.camera.updateProjectionMatrix();
    this.grid.position.y = 0;
  }

  async setSplat(bytes) {
    const { SparkRenderer, SplatMesh } = await import('@sparkjsdev/spark');
    if (!this.spark) {
      this.spark = new SparkRenderer({ renderer: this.renderer });
      this.scene.add(this.spark);
    }
    const mesh = new SplatMesh({ fileBytes: bytes, fileType: 'splat' });
    await mesh.initialized;
    this.setLayer('splat', mesh);
    return mesh;
  }
}

// Reads the binary PLYs the laptop sends back (float x y z + uchar red green blue), as written by
// app/pipeline/stages/phone.py write_points_ply and by quickModelPly.
export function parsePointsPly(buffer) {
  const bytes = new Uint8Array(buffer);
  const text = new TextDecoder().decode(bytes.subarray(0, Math.min(bytes.length, 2048)));
  const end = text.indexOf('end_header\n');
  if (end < 0) throw new Error('Not a PLY file.');
  const count = Number(/element vertex (\d+)/.exec(text)[1]);
  const offset = new TextEncoder().encode(text.slice(0, end + 'end_header\n'.length)).length;
  const view = new DataView(buffer, offset);
  const positions = new Float32Array(count * 3);
  const colors = new Uint8Array(count * 3);
  for (let i = 0; i < count; i++) {
    const o = i * 15;
    positions[i * 3] = view.getFloat32(o, true);
    positions[i * 3 + 1] = view.getFloat32(o + 4, true);
    positions[i * 3 + 2] = view.getFloat32(o + 8, true);
    colors[i * 3] = view.getUint8(o + 12);
    colors[i * 3 + 1] = view.getUint8(o + 13);
    colors[i * 3 + 2] = view.getUint8(o + 14);
  }
  return { positions, colors };
}
