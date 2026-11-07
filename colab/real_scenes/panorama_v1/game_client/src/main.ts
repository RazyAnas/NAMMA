import GUI from 'lil-gui';
import * as THREE from 'three';
import RAPIER from '@dimforge/rapier3d-compat';
import {
  BloomEffect,
  BrightnessContrastEffect,
  EffectComposer,
  EffectPass,
  RenderPass,
  VignetteEffect,
} from 'postprocessing';
import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
// NAMMA: world, sky, panorama lighting, car, HUD (src/namma.ts)
import { NAMMA, addSkyDome, addSurface, ProbeLighting, Car, Hud } from './namma';
await RAPIER.init();

/**
 * Environment GLB `ShadowMaterial` draws in the transparent pass after splats (`renderOrder`).
 * `depthWrite: false` keeps splats from being punched out; `depthTest: true` (on the material)
 * keeps shadows from drawing on top of opaque geometry like the character (opaque pass runs first).
 */
const SHADOW_CATCHER_RENDER_ORDER = 1000;

/**
 * **Swap the world here:** Gaussian splat (`.spz`) + invisible collider mesh (`.glb`).
 * Vite serves everything in `public/` at the root path `/`.
 */
const WORLD_ASSETS = {
  splatSpz: NAMMA.splatUrl, // NAMMA: from game.json (.rad LoD stream or .ply)
  colliderGlb: NAMMA.colliderUrl,
} as const;

/**
 * **Swap the character here:** any animated GLB with idle, walk, and air-idle clips.
 *   1. Drop your `.glb` under `public/` and update `glb` below.
 *   2. If your clips have different names, update the `CLIP_*` constants below as well.
 */
const CHARACTER_ASSETS = {
  /** Path served from `public/` — replace with your own animated character GLB. */
  glb: NAMMA.character.glb, // NAMMA: CC0 avatar by default
} as const;

/**
 * Animation clip names expected inside `CHARACTER_ASSETS.glb`.
 * Update these if your model uses different clip names.
 */
const CLIP_HAPPY_IDLE = NAMMA.character.clips.idle;
const CLIP_AIR_IDLE = NAMMA.character.clips.air;
const CLIP_WALK = NAMMA.character.clips.walk;
const CLIP_TRICK = NAMMA.character.clips.trick;
const TRICK_DURATION_MS = 2000;

/** Total KCC capsule height (meters): cylinder section + two hemispheres = height + 2 * radius */
const CHARACTER_CAPSULE_HEIGHT = NAMMA.capsuleHeight; // NAMMA: metric human
/** Orbit `controls.target` Y = character feet Y + this (focus upper body / head). */
const CAMERA_ORBIT_TARGET_Y_OFFSET = CHARACTER_CAPSULE_HEIGHT * 0.74;
const characterRadiusStanding = NAMMA.metric ? 0.3 : 0.55 * 1.5;
const characterHeightStanding = CHARACTER_CAPSULE_HEIGHT - 2 * characterRadiusStanding;
const characterMass = 1000;
const maxSlopeAngle = (45 * Math.PI) / 180;
const characterPadding = 0.02;

const tuning = {
  moveSpeed: 5,
  /** Move speed multiplier while Shift is held */
  sprintMoveMultiplier: 1.85,
  /** Extra walk-clip speed multiplier while Shift is held (stacks with walk anim speed) */
  sprintWalkAnimMultiplier: 1.4,
  jumpSpeed: 12,
  gravityY: -25,
  enableWalkStairs: true,
  enableStickToFloor: true,
  controlMovementDuringJump: true,
  enableCharacterInertia: true,
  /** World-space height of the character mesh (should match capsule height). */
  characterHeight: CHARACTER_CAPSULE_HEIGHT,
  characterYawDeg: 0,
  characterOffsetX: 0,
  /** Extra vertical offset in rig space (positive lifts the mesh along the capsule axis). */
  characterOffsetY: 0,
  characterOffsetZ: 0,
  /** Higher = character snaps to move direction faster (~rad/s style exponential). */
  characterTurnSpeed: 6,
  /** Horizontal speed above this threshold triggers the walk clip (m/s). */
  characterWalkSpeedThreshold: 0.08,
  /**
   * Seconds Rapier can report "not grounded" before the character anim switches to airborne
   * (reduces flicker on small bumps; jump / input still uses instant physics grounding).
   */
  characterAnimGroundReleaseHold: 0.11,
  /** Larger = walk-speed smoothing reacts more slowly to spikes (reduces walk/idle flicker). */
  characterAnimWalkSpeedSmoothing: 10,
  /** Crossfade duration in seconds between idle and walk clips. */
  characterAnimCrossfade: 0.28,
  /** Walk-clip playback rate (1 = normal, 2 = double speed). */
  characterWalkAnimTimeScale: 1.35,
  showPhysicsDebug: false,
  debugBodies: true,
  /** Uniform scale for the Spark splat mesh (see `WORLD_ASSETS.splatSpz`). */
  splatUniformScale: 5,
  /** Uniform scale applied to the collider GLB root before Rapier bake (match splat authoring units). Changing this rebakes physics. */
  colliderGlbUniformScale: 5,
  ambientIntensity: 0.38,
  sunIntensity: 1.62,
  /** When `sunShadowFollowCharacter` is true, these are world-space offsets from character feet → sun position. */
  sunPosX: -15.5,
  sunPosY: 102,
  sunPosZ: 12,
  /** Move the directional light + shadow frustum with the player (keeps shadow map high-res near you). */
  sunShadowFollowCharacter: true,
  sunColor: new THREE.Color(0xffe8c9),
  shadowMapSize: 2048,
  shadowBias: 0,
  shadowNormalBias: 0,
  shadowRadius: 8,
  shadowCameraNear: 1,
  shadowCameraFar: 120,
  shadowCameraHalfExtent: 25,
  shadowMapType: 'PCFShadowMap' as
    | 'BasicShadowMap'
    | 'PCFShadowMap'
    | 'PCFSoftShadowMap'
    | 'VSMShadowMap',
  /** How strongly shadows darken surfaces (`mix(1, shadow, intensity)`); lower = softer. */
  shadowIntensity: 1,
  /** Number of blur passes for VSM soft shadows (more = smoother, slower). */
  shadowBlurSamples: 25,
  /** Shared `ShadowMaterial` opacity for collider GLB meshes (see `WORLD_ASSETS.colliderGlb`). */
  colliderGlbShadowOpacity: 0.3,
  colliderGlbShadowColor: new THREE.Color(0x000000),

  /** [pmndrs/postprocessing](https://github.com/pmndrs/postprocessing) */
  ppEnabled: true,
  ppBloomIntensity: 0.75,
  ppBloomThreshold: 0.29,
  ppBloomSmoothing: 0.5,
  ppBrightness: -0.1,
  ppContrast: 0.1,
  ppVignetteDarkness: 0.57,
  ppVignetteOffset: 0.5,
};

Object.assign(tuning, NAMMA.tuning); // NAMMA: metric speeds, no colour grading on real photos
/** Capsule center Y in world space = feet Y + this (matches `THREE.CapsuleGeometry` layout). */
const capsuleCenterY = 0.5 * characterHeightStanding + characterRadiusStanding;

const rapierWorld = new RAPIER.World({ x: 0, y: tuning.gravityY, z: 0 });

const characterController = rapierWorld.createCharacterController(characterPadding);
characterController.setUp({ x: 0, y: 1, z: 0 });
characterController.setSlideEnabled(true);
characterController.setMaxSlopeClimbAngle(maxSlopeAngle);
characterController.setMinSlopeSlideAngle((30 * Math.PI) / 180);
characterController.setApplyImpulsesToDynamicBodies(true);
characterController.setCharacterMass(characterMass);
characterController.enableAutostep(0.4, 0.2, true);
characterController.enableSnapToGround(0.5);

type CharacterState = {
  desiredVelocity: THREE.Vector3;
  linearVelocity: THREE.Vector3;
  allowSliding: boolean;
  /** Grounding from the last `computeColliderMovement` (Rapier [character controller](https://rapier.rs/docs/user_guides/bevy_plugin/character_controller/)). */
  grounded: boolean;
};

const characterState: CharacterState = {
  desiredVelocity: new THREE.Vector3(),
  linearVelocity: new THREE.Vector3(),
  allowSliding: false,
  grounded: false,
};

const spawnFeetY = NAMMA.spawn.pos[1] - 1.7; // NAMMA: spawn on the route (eye height 1.7 m)
const spawnX = NAMMA.spawn.pos[0];
const spawnZ = NAMMA.spawn.pos[2];
const playerBodyDesc = RAPIER.RigidBodyDesc.kinematicPositionBased().setTranslation(
  spawnX,
  spawnFeetY + capsuleCenterY,
  spawnZ,
);
const playerBody = rapierWorld.createRigidBody(playerBodyDesc);
const playerColliderDesc = RAPIER.ColliderDesc.capsule(
  characterHeightStanding / 2,
  characterRadiusStanding,
);
const playerCollider = rapierWorld.createCollider(playerColliderDesc, playerBody);

/** Character feet (world space) for sun + shadow follow; reused in `syncLightingAndShadowsFromTuning`. */
const _sunShadowFollowFeet = new THREE.Vector3();

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xd6d6d6);

const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.05, 4000);
await addSkyDome(scene, camera); // NAMMA: learned sky (splats never model the sky)
void addSurface(scene); // NAMMA: textured street mesh under the splats (no smears from any angle)

const renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'high-performance' }); // NAMMA: Spark perf guide
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, Number(new URLSearchParams(location.search).get('dpr') || 1.5)));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.NoToneMapping;
document.body.appendChild(renderer.domElement);

/** PMREM env from `public/pano.jpg` (character + reflector balls). */
let characterReflectionEnvMap: THREE.Texture | null = null;

const reflectorBallsGroup = new THREE.Group();
reflectorBallsGroup.name = 'PanoReflectorBalls';

type ReflectorBallRig = { mesh: THREE.Mesh; body: RAPIER.RigidBody };
const reflectorBallRigs: ReflectorBallRig[] = [];

{
  const ballCount = 0; // NAMMA: no demo balls
  const ballRadius = 1.2;
  const ballGeo = new THREE.SphereGeometry(ballRadius, 32, 24);
  const pinkHues = [0xf8bbd9, 0xf06292, 0xad1457, 0xf48fb1, 0xce93d8] as const;
  /** Center spacing so dynamic balls do not start inside each other (diameter ≈ 2.4). */
  const gridStep = 2.75;
  const cols = 5;
  const rows = Math.ceil(ballCount / cols);
  /** −Z toward default camera (+Z); +X shifts the whole row to the right of the character. */
  const spawnZInFrontOfCharacter = -12;
  const spawnXOffsetRight = 4;
  for (let i = 0; i < ballCount; i++) {
    const col = i % cols;
    const row = Math.floor(i / cols);
    const x = (col - (cols - 1) / 2) * gridStep + spawnXOffsetRight;
    const z = (row - (rows - 1) / 2) * gridStep + spawnZInFrontOfCharacter;
    const y = ballRadius;

    const mat = new THREE.MeshStandardMaterial({
      color: pinkHues[i % pinkHues.length],
      metalness: 0.22,
      roughness: 0.4,
      envMapIntensity: 0.5,
    });
    const ball = new THREE.Mesh(ballGeo, mat);
    ball.castShadow = true;
    ball.receiveShadow = true;
    ball.position.set(x, y, z);
    reflectorBallsGroup.add(ball);

    const bodyDesc = RAPIER.RigidBodyDesc.dynamic()
      .setTranslation(x, y, z)
      .setLinearDamping(0.02)
      .setAngularDamping(0.05);
    const body = rapierWorld.createRigidBody(bodyDesc);
    const colliderDesc = RAPIER.ColliderDesc.ball(ballRadius)
      .setMass(0.85)
      .setFriction(0.38)
      .setRestitution(0.12);
    rapierWorld.createCollider(colliderDesc, body);
    reflectorBallRigs.push({ mesh: ball, body });
  }
  scene.add(reflectorBallsGroup);
}

rapierWorld.propagateModifiedBodyPositionsToColliders(); // NAMMA: Rapier 0.20 removed updateSceneQueries

function syncReflectorBallsFromPhysics() {
  for (const { mesh, body } of reflectorBallRigs) {
    const t = body.translation();
    const r = body.rotation();
    mesh.position.set(t.x, t.y, t.z);
    mesh.quaternion.set(r.x, r.y, r.z, r.w);
  }
}

function tryApplyPanoReflectionEnv() {
  const env = characterReflectionEnvMap;
  if (!env) return;

  if (characterModel) {
    characterModel.traverse((obj) => {
      const mesh = obj as THREE.Mesh;
      if (!mesh.isMesh || !mesh.material) return;
      const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
      for (const mat of mats) {
        if (mat instanceof THREE.MeshStandardMaterial || mat instanceof THREE.MeshPhysicalMaterial) {
          mat.envMap = env;
          mat.envMapIntensity = 1;
        }
      }
    });
  }

  reflectorBallsGroup.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    if (!mesh.isMesh || !mesh.material) return;
    const m = mesh.material;
    if (m instanceof THREE.MeshStandardMaterial || m instanceof THREE.MeshPhysicalMaterial) {
      m.envMap = env;
      m.envMapIntensity = 0.5;
    }
  });
}

// NAMMA: the nearest real panorama lights the avatar and the car
const probeLighting = new ProbeLighting(renderer, (env) => {
  characterReflectionEnvMap = env;
  tryApplyPanoReflectionEnv();
  for (const m of nammaCar.materials) { m.envMap = env; m.envMapIntensity = 1; m.needsUpdate = true; }
});
if (!NAMMA.probes.length) new THREE.TextureLoader().load(
  '/pano.jpg',
  (tex) => {
    tex.mapping = THREE.EquirectangularReflectionMapping;
    tex.colorSpace = THREE.SRGBColorSpace;
    const pmrem = new THREE.PMREMGenerator(renderer);
    pmrem.compileEquirectangularShader();
    const { texture } = pmrem.fromEquirectangular(tex);
    tex.dispose();
    pmrem.dispose();
    characterReflectionEnvMap = texture;
    tryApplyPanoReflectionEnv();
  },
  undefined,
  (err) => {
    console.warn('Could not load /pano.jpg — add it under public/pano.jpg for character reflections', err);
  },
);

/** Gaussian splat background ([Spark docs](https://sparkjs.dev/docs/)); URL from `WORLD_ASSETS`. */
const sparkRenderer = new SparkRenderer({
  renderer,
  enableLod: true,
  /** Higher skips tinier screen-space splats; ~2 is often hard to notice ([performance tuning](https://sparkjs.dev/docs/)). */
  lodRenderScale: 2,
});
scene.add(sparkRenderer);
const splatMesh = new SplatMesh({
  url: WORLD_ASSETS.splatSpz,
  /** Required so the asset gets a LoD tree (`lodSplats`); otherwise Spark renders the full splat count every frame. */
  lod: !NAMMA.splatIsRad, // NAMMA: a pre-built .rad already carries its LoD tree
  paged: NAMMA.splatIsRad, // NAMMA: stream the .rad instead of downloading it whole
});
splatMesh.quaternion.identity();
splatMesh.position.set(0, 0, 0);
splatMesh.scale.setScalar(tuning.splatUniformScale);
scene.add(splatMesh);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(spawnX, spawnFeetY + CAMERA_ORBIT_TARGET_Y_OFFSET, spawnZ);
{ // NAMMA: start behind the avatar, looking down the route
  const look = new THREE.Vector3(NAMMA.spawn.look_at[0] - spawnX, 0, NAMMA.spawn.look_at[2] - spawnZ).normalize();
  camera.position.set(spawnX - look.x * 4, spawnFeetY + 2.2, spawnZ - look.z * 4);
}
const nammaCar = new Car(rapierWorld, scene);
const nammaHud = new Hud();
document.addEventListener('keydown', (e) => { // NAMMA: C = get in / out of the car
  if (e.code !== 'KeyC') return;
  if (nammaCar.active) {
    const out = nammaCar.despawn();
    if (out) playerBody.setTranslation({ x: out.x, y: out.y + capsuleCenterY, z: out.z }, true);
    playerCollider.setEnabled(true);
    if (characterModel) characterModel.visible = true;
  } else {
    const t = playerBody.translation();
    const fwd = new THREE.Vector3(); camera.getWorldDirection(fwd);
    nammaCar.spawn({ x: t.x, y: t.y - capsuleCenterY, z: t.z }, Math.atan2(-fwd.x, -fwd.z));
    playerCollider.setEnabled(false);
    if (characterModel) characterModel.visible = false;
  }
});

const brightnessContrastEffect = new BrightnessContrastEffect({
  brightness: tuning.ppBrightness,
  contrast: tuning.ppContrast,
});
const bloomEffect = new BloomEffect({
  mipmapBlur: true,
  luminanceThreshold: tuning.ppBloomThreshold,
  luminanceSmoothing: tuning.ppBloomSmoothing,
  intensity: tuning.ppBloomIntensity,
  radius: 0.55,
});
const vignetteEffect = new VignetteEffect({
  darkness: tuning.ppVignetteDarkness,
  offset: tuning.ppVignetteOffset,
});
const effectPass = new EffectPass(camera, brightnessContrastEffect, bloomEffect, vignetteEffect);
const composer = new EffectComposer(renderer, {
  depthBuffer: true,
  stencilBuffer: false,
});
composer.addPass(new RenderPass(scene, camera));
composer.addPass(effectPass);
composer.setSize(window.innerWidth, window.innerHeight);

function syncPostProcessingFromTuning() {
  if (!tuning.ppEnabled) {
    bloomEffect.intensity = 0;
    brightnessContrastEffect.brightness = 0;
    brightnessContrastEffect.contrast = 0;
    vignetteEffect.darkness = 0;
    return;
  }
  bloomEffect.intensity = tuning.ppBloomIntensity;
  bloomEffect.luminanceMaterial.threshold = tuning.ppBloomThreshold;
  bloomEffect.luminanceMaterial.smoothing = tuning.ppBloomSmoothing;
  brightnessContrastEffect.brightness = tuning.ppBrightness;
  brightnessContrastEffect.contrast = tuning.ppContrast;
  vignetteEffect.darkness = tuning.ppVignetteDarkness;
  vignetteEffect.offset = tuning.ppVignetteOffset;
}

const ambientLight = new THREE.AmbientLight(0xffffff, tuning.ambientIntensity);
scene.add(ambientLight);

const sun = new THREE.DirectionalLight(0xfff5e6, tuning.sunIntensity);
sun.castShadow = true;
scene.add(sun);
sun.target.position.set(0, 0, 0);
scene.add(sun.target);


/** One shadow-catcher material for every mesh in the collider GLB; created when the asset finishes loading. */
let colliderGlbShadowMaterial: THREE.ShadowMaterial | null = null;

function syncLightingAndShadowsFromTuning() {
  ambientLight.intensity = tuning.ambientIntensity;
  sun.color.copy(tuning.sunColor);
  sun.intensity = tuning.sunIntensity;
  if (tuning.sunShadowFollowCharacter) {
    const t = playerBody.translation();
    _sunShadowFollowFeet.set(t.x, t.y - capsuleCenterY, t.z);
    sun.target.position.copy(_sunShadowFollowFeet);
    sun.position.set(
      _sunShadowFollowFeet.x + tuning.sunPosX,
      _sunShadowFollowFeet.y + tuning.sunPosY,
      _sunShadowFollowFeet.z + tuning.sunPosZ,
    );
  } else {
    sun.target.position.set(0, 0, 0);
    sun.position.set(tuning.sunPosX, tuning.sunPosY, tuning.sunPosZ);
  }

  const shadowCam = sun.shadow.camera as THREE.OrthographicCamera;
  shadowCam.near = tuning.shadowCameraNear;
  shadowCam.far = tuning.shadowCameraFar;
  const half = tuning.shadowCameraHalfExtent;
  shadowCam.left = -half;
  shadowCam.right = half;
  shadowCam.top = half;
  shadowCam.bottom = -half;
  shadowCam.updateProjectionMatrix();

  sun.shadow.bias = tuning.shadowBias;
  sun.shadow.normalBias = tuning.shadowNormalBias;
  sun.shadow.radius = tuning.shadowRadius;
  sun.shadow.intensity = tuning.shadowIntensity;
  sun.shadow.blurSamples = Math.round(
    Math.min(32, Math.max(4, tuning.shadowBlurSamples)),
  );

  const ms = Math.min(4096, Math.max(256, Math.round(tuning.shadowMapSize / 128) * 128));
  if (sun.shadow.mapSize.width !== ms) {
    sun.shadow.map?.dispose();
    sun.shadow.map = null;
    sun.shadow.mapSize.set(ms, ms);
  }

  renderer.shadowMap.type = THREE[tuning.shadowMapType];


  if (colliderGlbShadowMaterial) {
    colliderGlbShadowMaterial.color.copy(tuning.colliderGlbShadowColor);
    colliderGlbShadowMaterial.opacity = tuning.colliderGlbShadowOpacity;
    colliderGlbShadowMaterial.transparent = tuning.colliderGlbShadowOpacity < 0.999;
    colliderGlbShadowMaterial.depthWrite = false;
    colliderGlbShadowMaterial.depthTest = true;
  }
}

syncLightingAndShadowsFromTuning();
syncPostProcessingFromTuning();

const rapierDebugGeom = new THREE.BufferGeometry();
const rapierDebugPos = new THREE.BufferAttribute(new Float32Array(0), 3);
const rapierDebugCol = new THREE.BufferAttribute(new Float32Array(0), 3);
rapierDebugGeom.setAttribute('position', rapierDebugPos);
rapierDebugGeom.setAttribute('color', rapierDebugCol);
const rapierDebugLines = new THREE.LineSegments(
  rapierDebugGeom,
  new THREE.LineBasicMaterial({ vertexColors: true, toneMapped: false }),
);
rapierDebugLines.visible = false;
scene.add(rapierDebugLines);

const capsuleCylinderLength = characterHeightStanding;
const capsuleMat = new THREE.MeshStandardMaterial({
  color: 0xc45cff,
  roughness: 0.4,
  metalness: 0.15,
  transparent: true,
  opacity: 0,
  depthWrite: false,
  wireframe: false,
});
/** Visual capsule (mirrors Rapier capsule collider on the kinematic body — this mesh has no physics). */
const capsuleMesh = new THREE.Mesh(new THREE.CapsuleGeometry(characterRadiusStanding, capsuleCylinderLength, 6, 12), capsuleMat);
capsuleMesh.castShadow = false;

/** Feet at origin; follows character capsule center each frame */
const playerRoot = new THREE.Group();
scene.add(playerRoot);

/** Rig that moves with the character: capsule mesh + character visual. */
const capsuleRig = new THREE.Group();
capsuleRig.name = 'CapsuleRig';
capsuleRig.position.set(0, capsuleCenterY, 0);
playerRoot.add(capsuleRig);
capsuleMesh.position.set(0, 0, 0);
capsuleRig.add(capsuleMesh);

const characterHolder = new THREE.Group();
characterHolder.name = 'CharacterVisual';
capsuleRig.add(characterHolder);

/** Yaw toward movement; character mesh stays child here so layout yaw stays separate. */
const characterFacingPivot = new THREE.Group();
characterFacingPivot.name = 'CharacterFacing';
characterHolder.add(characterFacingPivot);

const _charFwd = new THREE.Vector3();
const _charInvQuat = new THREE.Quaternion();
let characterSmoothedFacingYaw = 0;

function shortestAngleDelta(from: number, to: number) {
  return Math.atan2(Math.sin(to - from), Math.cos(to - from));
}

function lerpAngleRad(current: number, target: number, t: number) {
  return current + shortestAngleDelta(current, target) * t;
}

function syncCharacterHolderPosition() {
  characterHolder.position.set(tuning.characterOffsetX, -capsuleCenterY + tuning.characterOffsetY, tuning.characterOffsetZ);
}

syncCharacterHolderPosition();

let characterModel: THREE.Object3D | null = null;

let characterMixer: THREE.AnimationMixer | null = null;
let characterHappyIdleAction: THREE.AnimationAction | null = null;
let characterAirIdleAction: THREE.AnimationAction | null = null;
let characterWalkAction: THREE.AnimationAction | null = null;
let characterTrickAction: THREE.AnimationAction | null = null;
/** Wall-clock end time for trick override; 0 = not playing */
let characterTrickEndTime = 0;
let characterTrickRestoreTimer: ReturnType<typeof setTimeout> | undefined;
let characterAnimPlaying: 'happy' | 'walk' | 'airIdle' = 'happy';

/** Animation-only: stays true briefly after physics reports not grounded (see `characterAnimGroundReleaseHold`). */
let characterAnimStableGrounded = false;
let characterAnimAirAccum = 0;
/** Low-pass horizontal speed for walk vs idle (m/s). */
let characterSmoothedHSpeed = 0;

function updateCharacterAnimStability(deltaTime: number, physicsGrounded: boolean) {
  if (physicsGrounded) {
    characterAnimStableGrounded = true;
    characterAnimAirAccum = 0;
  } else {
    characterAnimAirAccum += deltaTime;
    if (characterAnimAirAccum >= tuning.characterAnimGroundReleaseHold) {
      characterAnimStableGrounded = false;
    }
  }

  const lv = characterState.linearVelocity;
  const h = Math.hypot(lv.x, lv.z);
  const k = tuning.characterAnimWalkSpeedSmoothing;
  const t = 1 - Math.exp(-k * deltaTime);
  characterSmoothedHSpeed += (h - characterSmoothedHSpeed) * t;
}

function characterClipActionFor(anim: typeof characterAnimPlaying): THREE.AnimationAction | null {
  switch (anim) {
    case 'happy':
      return characterHappyIdleAction;
    case 'walk':
      return characterWalkAction;
    case 'airIdle':
      return characterAirIdleAction;
    default:
      return null;
  }
}

function layoutCharacterModel() {
  if (!characterModel) return;
  const previousParent = characterModel.parent;
  if (previousParent) previousParent.remove(characterModel);
  scene.add(characterModel);

  characterModel.position.set(0, 0, 0);
  characterModel.quaternion.identity();
  characterModel.scale.set(1, 1, 1);
  characterModel.updateMatrixWorld(true);

  const box = new THREE.Box3().setFromObject(characterModel);
  const h = Math.max(box.max.y - box.min.y, 1e-4);
  const s = tuning.characterHeight / h;
  characterModel.scale.setScalar(s);
  characterModel.updateMatrixWorld(true);
  box.setFromObject(characterModel);
  characterModel.position.set(-(box.min.x + box.max.x) / 2, -box.min.y, -(box.min.z + box.max.z) / 2);
  characterModel.rotation.y = THREE.MathUtils.degToRad(tuning.characterYawDeg);

  scene.remove(characterModel);
  characterFacingPivot.add(characterModel);
  syncCharacterHolderPosition();
}

function initCharacterAnimations(gltf: { animations: THREE.AnimationClip[] }) {
  if (!characterModel || gltf.animations.length === 0) return;

  characterMixer = new THREE.AnimationMixer(characterModel);

  const byName = (name: string) => gltf.animations.find((c) => c.name === name);
  const happyClip =
    byName(CLIP_HAPPY_IDLE) ?? gltf.animations.find((c) => /happy\s*idle/i.test(c.name));
  const airIdleClip = byName(CLIP_AIR_IDLE);
  const walkClip =
    byName(CLIP_WALK) ?? gltf.animations.find((c) => /brutal/i.test(c.name));

  if (!happyClip) {
    console.warn('[character] Missing idle clip (CLIP_HAPPY_IDLE). Available:', gltf.animations.map((a) => a.name).join(', '));
  } else {
    characterHappyIdleAction = characterMixer.clipAction(happyClip);
    characterHappyIdleAction.setLoop(THREE.LoopRepeat, Infinity);
  }
  if (!airIdleClip) {
    console.warn('[character] Missing airborne idle clip (CLIP_AIR_IDLE). Available:', gltf.animations.map((a) => a.name).join(', '));
  } else {
    characterAirIdleAction = characterMixer.clipAction(airIdleClip);
    characterAirIdleAction.setLoop(THREE.LoopRepeat, Infinity);
  }
  if (!walkClip) {
    console.warn('[character] Missing walk clip (CLIP_WALK). Available:', gltf.animations.map((a) => a.name).join(', '));
  } else {
    characterWalkAction = characterMixer.clipAction(walkClip);
    characterWalkAction.setLoop(THREE.LoopRepeat, Infinity);
  }

  const trickClip =
    byName(CLIP_TRICK) ?? gltf.animations.find((c) => /step\s*hip\s*hop/i.test(c.name));
  if (!trickClip) {
    console.warn('[character] Missing trick clip (CLIP_TRICK / press 1). Available:', gltf.animations.map((a) => a.name).join(', '));
  } else {
    characterTrickAction = characterMixer.clipAction(trickClip);
    characterTrickAction.setLoop(THREE.LoopRepeat, Infinity);
  }

  characterHappyIdleAction?.stop();
  characterAirIdleAction?.stop();
  characterWalkAction?.stop();
  characterTrickAction?.stop();

  if (characterHappyIdleAction) {
    characterHappyIdleAction.reset().setEffectiveWeight(1).play();
    characterAnimPlaying = 'happy';
  } else if (characterAirIdleAction) {
    characterAirIdleAction.reset().setEffectiveWeight(1).play();
    characterAnimPlaying = 'airIdle';
  } else if (characterWalkAction) {
    characterWalkAction.reset().setEffectiveWeight(1).play();
    characterAnimPlaying = 'walk';
  }
}

function resolveCharacterAnimWant(inAir: boolean, movingOnGround: boolean): typeof characterAnimPlaying {
  if (inAir) {
    if (characterAirIdleAction) return 'airIdle';
    if (characterHappyIdleAction) return 'happy';
    return 'walk';
  }
  if (movingOnGround) {
    if (characterWalkAction) return 'walk';
    if (characterHappyIdleAction) return 'happy';
    return 'airIdle';
  }
  if (characterHappyIdleAction) return 'happy';
  if (characterAirIdleAction) return 'airIdle';
  return 'walk';
}

function finishCharacterTrick() {
  if (!characterMixer || !characterTrickAction) return;
  characterTrickEndTime = 0;
  const inAir = !characterAnimStableGrounded;
  const movingOnGround = characterSmoothedHSpeed > tuning.characterWalkSpeedThreshold;
  const want = resolveCharacterAnimWant(inAir, movingOnGround);
  const next = characterClipActionFor(want);
  const d = tuning.characterAnimCrossfade;
  if (next) {
    next.reset().setEffectiveWeight(1).play();
    characterTrickAction.crossFadeTo(next, d, false);
    characterAnimPlaying = want;
  }
}

function startCharacterTrick() {
  if (!characterTrickAction || !characterMixer) return;
  if (characterTrickRestoreTimer !== undefined) {
    clearTimeout(characterTrickRestoreTimer);
  }
  characterTrickEndTime = performance.now() + TRICK_DURATION_MS;
  const prev = characterClipActionFor(characterAnimPlaying);
  characterTrickAction.reset().setEffectiveWeight(1).play();
  if (prev && prev !== characterTrickAction) {
    prev.crossFadeTo(characterTrickAction, tuning.characterAnimCrossfade, false);
  } else {
    characterTrickAction.fadeIn(tuning.characterAnimCrossfade).play();
  }
  characterTrickRestoreTimer = setTimeout(() => {
    characterTrickRestoreTimer = undefined;
    finishCharacterTrick();
  }, TRICK_DURATION_MS);
}

function updateCharacterAnimations(deltaTime: number) {
  if (!characterMixer) return;

  const trickPlaying = characterTrickAction !== null && characterTrickEndTime > 0 && performance.now() < characterTrickEndTime;

  if (characterWalkAction && !trickPlaying) {
    let ts = tuning.characterWalkAnimTimeScale;
    if (input.sprintPressed && characterAnimPlaying === 'walk') {
      ts *= tuning.sprintWalkAnimMultiplier;
    }
    characterWalkAction.setEffectiveTimeScale(ts);
  }

  if (trickPlaying) {
    characterMixer.update(deltaTime);
    return;
  }

  const inAir = !characterAnimStableGrounded;
  const movingOnGround = characterSmoothedHSpeed > tuning.characterWalkSpeedThreshold;
  const want = resolveCharacterAnimWant(inAir, movingOnGround);

  if (want !== characterAnimPlaying) {
    const next = characterClipActionFor(want);
    const prev = characterClipActionFor(characterAnimPlaying);
    if (next) {
      const d = tuning.characterAnimCrossfade;
      next.reset().setEffectiveWeight(1).play();
      if (prev && prev !== next) {
        prev.crossFadeTo(next, d, false);
      } else {
        next.fadeIn(d);
      }
      characterAnimPlaying = want;
    }
  }

  characterMixer.update(deltaTime);
}

const _envTriWorld = new THREE.Vector3();

/** Merge all `Mesh` geometry under `root` into one triangle soup in world space (for `trimesh`). */
function mergeWorldSpaceTrianglesForPhysics(root: THREE.Object3D): { positions: number[]; indices: number[] } {
  const positions: number[] = [];
  const indices: number[] = [];
  let vertexBase = 0;
  const worldMat = new THREE.Matrix4();
  root.updateMatrixWorld(true);
  root.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    if (!mesh.isMesh) return;
    const geom = mesh.geometry;
    const posAttr = geom.getAttribute('position') as THREE.BufferAttribute | undefined;
    if (!posAttr) return;
    worldMat.copy(mesh.matrixWorld);
    for (let i = 0; i < posAttr.count; i++) {
      _envTriWorld.fromBufferAttribute(posAttr, i).applyMatrix4(worldMat);
      positions.push(_envTriWorld.x, _envTriWorld.y, _envTriWorld.z);
    }
    const indexAttr = geom.getIndex();
    if (indexAttr) {
      for (let i = 0; i < indexAttr.count; i++) {
        indices.push(vertexBase + indexAttr.getX(i));
      }
    } else {
      for (let i = 0; i + 2 < posAttr.count; i += 3) {
        indices.push(vertexBase + i, vertexBase + i + 1, vertexBase + i + 2);
      }
    }
    vertexBase += posAttr.count;
  });
  return { positions, indices };
}

new GLTFLoader().load(
  CHARACTER_ASSETS.glb,
  (gltf) => {
    characterModel = gltf.scene;
    characterModel.traverse((obj) => {
      const m = obj as THREE.Mesh;
      if (m.isMesh) {
        m.castShadow = true;
        m.receiveShadow = false;
      }
    });
    characterFacingPivot.add(characterModel);
    layoutCharacterModel();
    initCharacterAnimations(gltf);
    tryApplyPanoReflectionEnv();
  },
  undefined,
  (err) => {
    console.warn(`Could not load ${CHARACTER_ASSETS.glb} — add the file under public${CHARACTER_ASSETS.glb}`, err);
  },
);

const colliderGlbRoot = new THREE.Group();
colliderGlbRoot.name = 'ColliderEnvironmentGLB';
scene.add(colliderGlbRoot);

/** Static Rapier body for the trimesh collider; stored so it can be removed and rebaked on scale change. */
let staticColliderBody: RAPIER.RigidBody | null = null;

function bakeColliderPhysics() {
  if (staticColliderBody) {
    rapierWorld.removeRigidBody(staticColliderBody);
    staticColliderBody = null;
  }
  colliderGlbRoot.scale.setScalar(tuning.colliderGlbUniformScale);
  colliderGlbRoot.updateMatrixWorld(true);
  const { positions, indices } = mergeWorldSpaceTrianglesForPhysics(colliderGlbRoot);
  if (indices.length < 3) {
    console.warn(`${WORLD_ASSETS.colliderGlb}: no mesh triangles found for static collider`);
    return;
  }
  try {
    const verts = new Float32Array(positions);
    const idx = new Uint32Array(indices);
    staticColliderBody = rapierWorld.createRigidBody(RAPIER.RigidBodyDesc.fixed());
    rapierWorld.createCollider(RAPIER.ColliderDesc.trimesh(verts, idx), staticColliderBody);
    rapierWorld.propagateModifiedBodyPositionsToColliders(); // NAMMA: Rapier 0.20 removed updateSceneQueries
  } catch (e) {
    console.error(`${WORLD_ASSETS.colliderGlb}: failed to build triangle-mesh collider`, e);
  }
}

const colliderGlbPublicPath = `public${WORLD_ASSETS.colliderGlb}`;

new GLTFLoader().load(
  WORLD_ASSETS.colliderGlb,
  (gltf) => {
    const model = gltf.scene;
    colliderGlbRoot.add(model);

    if (!colliderGlbShadowMaterial) {
      colliderGlbShadowMaterial = new THREE.ShadowMaterial({
        opacity: tuning.colliderGlbShadowOpacity,
        color: tuning.colliderGlbShadowColor,
      });
      colliderGlbShadowMaterial.transparent = tuning.colliderGlbShadowOpacity < 0.999;
      colliderGlbShadowMaterial.depthWrite = false;
      colliderGlbShadowMaterial.depthTest = true;
    }

    model.traverse((child) => {
      const mesh = child as THREE.Mesh;
      if (!mesh.isMesh) return;

      const prev = mesh.material;
      const list = Array.isArray(prev) ? prev : [prev];
      for (const mat of list) {
        mat?.dispose();
      }

      mesh.material = colliderGlbShadowMaterial!;
      mesh.castShadow = false;
      mesh.receiveShadow = true;
      mesh.renderOrder = SHADOW_CATCHER_RENDER_ORDER;
    });

    bakeColliderPhysics();
  },
  undefined,
  (err) => {
    console.warn(
      `Could not load ${WORLD_ASSETS.colliderGlb} — add the file under ${colliderGlbPublicPath}`,
      err,
    );
  },
);

function syncRapierDebugFromTuning() {
  rapierDebugLines.visible = tuning.showPhysicsDebug && tuning.debugBodies;
}

function updateRapierDebugLines() {
  if (!rapierDebugLines.visible) return;
  const buffers = rapierWorld.debugRender();
  const v = buffers.vertices;
  const c = buffers.colors;
  const nVert = v.length / 3;
  const colors3 = new Float32Array(nVert * 3);
  for (let i = 0; i < nVert; i++) {
    colors3[i * 3 + 0] = c[i * 4 + 0];
    colors3[i * 3 + 1] = c[i * 4 + 1];
    colors3[i * 3 + 2] = c[i * 4 + 2];
  }
  rapierDebugGeom.setAttribute('position', new THREE.BufferAttribute(v.slice(), 3));
  rapierDebugGeom.setAttribute('color', new THREE.BufferAttribute(colors3, 3));
  rapierDebugGeom.getAttribute('position').needsUpdate = true;
  rapierDebugGeom.getAttribute('color').needsUpdate = true;
}

function syncPlayerRoot() {
  const t = playerBody.translation();
  playerRoot.position.set(t.x, t.y - capsuleCenterY, t.z);
  playerRoot.quaternion.identity();
  syncCharacterHolderPosition();
  playerRoot.updateMatrixWorld(true);
}

/**
 * Smooth yaw toward where you are steering (camera-relative input). Using physics velocity
 * only when there is no steer input avoids snapping when `hSpeed` crosses a threshold: while
 * circling, intent and velocity are not always the same vector.
 */
function updateCharacterMovementFacing(deltaTime: number, inputDirWorld: THREE.Vector3) {
  if (!characterModel) return;
  const inLen = inputDirWorld.length();
  let dx: number;
  let dz: number;
  if (inLen > 1e-5) {
    dx = inputDirWorld.x / inLen;
    dz = inputDirWorld.z / inLen;
  } else {
    const lv = characterState.linearVelocity;
    const hSpeed = Math.hypot(lv.x, lv.z);
    if (hSpeed < 1e-5) return;
    dx = lv.x / hSpeed;
    dz = lv.z / hSpeed;
  }
  _charFwd.set(dx, 0, dz);
  _charInvQuat.copy(playerRoot.quaternion).invert();
  _charFwd.applyQuaternion(_charInvQuat);
  const targetYaw = Math.atan2(_charFwd.x, _charFwd.z);
  const t = 1 - Math.exp(-tuning.characterTurnSpeed * deltaTime);
  characterSmoothedFacingYaw = lerpAngleRad(characterSmoothedFacingYaw, targetYaw, t);
  characterFacingPivot.rotation.y = characterSmoothedFacingYaw;
}

const input = {
  forwardPressed: false,
  backwardPressed: false,
  leftPressed: false,
  rightPressed: false,
  jump: false,
  sprintPressed: false,
};

document.addEventListener('keydown', (event) => {
  switch (event.code) {
    case 'KeyW':
      input.forwardPressed = true;
      break;
    case 'KeyS':
      input.backwardPressed = true;
      break;
    case 'KeyA':
      input.leftPressed = true;
      break;
    case 'KeyD':
      input.rightPressed = true;
      break;
    case 'Space':
      input.jump = true;
      event.preventDefault();
      break;
    case 'ShiftLeft':
    case 'ShiftRight':
      input.sprintPressed = true;
      break;
    case 'Digit1':
      if (!event.repeat) {
        startCharacterTrick();
      }
      break;
  }
});

document.addEventListener('keyup', (event) => {
  switch (event.code) {
    case 'KeyW':
      input.forwardPressed = false;
      break;
    case 'KeyS':
      input.backwardPressed = false;
      break;
    case 'KeyA':
      input.leftPressed = false;
      break;
    case 'KeyD':
      input.rightPressed = false;
      break;
    case 'Space':
      input.jump = false;
      break;
    case 'ShiftLeft':
    case 'ShiftRight':
      input.sprintPressed = false;
      break;
  }
});

const _mdTmp = new THREE.Vector3();

const handleCharacterInput = (state: CharacterState, deltaTime: number, movementDir: THREE.Vector3) => {
  const playerControlsHorizontalVelocity = tuning.controlMovementDuringJump || state.grounded;

  const moveSpeed =
    tuning.moveSpeed * (input.sprintPressed ? tuning.sprintMoveMultiplier : 1);

  _mdTmp.copy(movementDir);
  const movementLength = _mdTmp.length();
  if (movementLength > 1e-6) {
    _mdTmp.multiplyScalar(1 / movementLength);
  }

  if (playerControlsHorizontalVelocity) {
    state.allowSliding = movementLength > 1e-6;
    if (tuning.enableCharacterInertia) {
      state.desiredVelocity.multiplyScalar(0.75);
      state.desiredVelocity.addScaledVector(_mdTmp, 0.25 * moveSpeed);
    } else {
      state.desiredVelocity.copy(_mdTmp).multiplyScalar(moveSpeed);
    }
  } else {
    state.allowSliding = true;
  }

  const characterUp = new THREE.Vector3(0, 1, 0);
  const linearVelocity = state.linearVelocity.clone();
  const currentVerticalVelocity = characterUp.clone().multiplyScalar(linearVelocity.dot(characterUp));
  const groundVelocity = new THREE.Vector3(0, 0, 0);
  const gravity = new THREE.Vector3(0, tuning.gravityY, 0);

  const newVelocity = new THREE.Vector3();
  const verticalRelativeVel = currentVerticalVelocity.clone().sub(groundVelocity).dot(characterUp);
  const movingTowardsGround = verticalRelativeVel < 0.1;

  if (state.grounded) {
    const shouldStickToGround = tuning.enableCharacterInertia ? movingTowardsGround : true;

    if (shouldStickToGround) {
      newVelocity.copy(groundVelocity);
      if (input.jump && movingTowardsGround) {
        newVelocity.addScaledVector(characterUp, tuning.jumpSpeed);
      }
    } else {
      newVelocity.copy(currentVerticalVelocity);
    }
  } else {
    newVelocity.copy(currentVerticalVelocity);
  }

  newVelocity.addScaledVector(gravity, deltaTime);

  if (playerControlsHorizontalVelocity) {
    newVelocity.add(state.desiredVelocity);
  } else {
    const currentHorizontalVelocity = linearVelocity.clone().sub(currentVerticalVelocity);
    newVelocity.add(currentHorizontalVelocity);
  }

  state.linearVelocity.copy(newVelocity);
};

function syncCharacterControllerFromTuning() {
  if (tuning.enableWalkStairs) {
    characterController.enableAutostep(0.4, 0.2, true);
  } else {
    characterController.disableAutostep();
  }
  if (tuning.enableStickToFloor) {
    characterController.enableSnapToGround(0.5);
  } else {
    characterController.disableSnapToGround();
  }
}

const maxDelta = 1 / 30;
let lastTime = performance.now();

// ─── GUI ────────────────────────────────────────────────────────────────────
const gui = new GUI({ title: 'NAMMA Street World' });
gui.close(); // NAMMA: tuning panel available but closed

const moveFolder = gui.addFolder('Movement');
moveFolder.add(tuning, 'moveSpeed', 1, 40, 0.5).name('Move speed');
moveFolder.add(tuning, 'sprintMoveMultiplier', 1, 3, 0.05).name('Sprint multiplier (Shift)');
moveFolder.add(tuning, 'sprintWalkAnimMultiplier', 1, 2.5, 0.05).name('Sprint anim multiplier');
moveFolder.add(tuning, 'jumpSpeed', 2, 35, 0.5).name('Jump impulse');
moveFolder.add(tuning, 'controlMovementDuringJump').name('Air control');
moveFolder.add(tuning, 'enableCharacterInertia').name('Move inertia');
moveFolder.close();

const physFolder = gui.addFolder('Physics');
physFolder.add(tuning, 'gravityY', -60, -5, 0.5).name('Gravity Y');
physFolder.add(tuning, 'enableWalkStairs').name('Walk stairs');
physFolder.add(tuning, 'enableStickToFloor').name('Stick to floor');
physFolder.close();

const characterFolder = gui.addFolder('Character');
characterFolder.add(tuning, 'characterHeight', 0.5, 8, 0.05).name('Height (units)').onChange(() => layoutCharacterModel());
characterFolder.add(tuning, 'characterYawDeg', -180, 180, 1).name('Mesh yaw (°)').onChange(() => layoutCharacterModel());
characterFolder.add(tuning, 'characterTurnSpeed', 0.5, 24, 0.25).name('Turn speed');
characterFolder.add(tuning, 'characterWalkSpeedThreshold', 0.02, 0.35, 0.01).name('Walk speed threshold');
characterFolder.add(tuning, 'characterAnimGroundReleaseHold', 0, 0.35, 0.01).name('Air anim delay (s)');
characterFolder.add(tuning, 'characterAnimWalkSpeedSmoothing', 2, 40, 0.5).name('Walk speed smoothing');
characterFolder.add(tuning, 'characterWalkAnimTimeScale', 0.25, 2.5, 0.05).name('Walk anim speed');
characterFolder.add(tuning, 'characterAnimCrossfade', 0.05, 0.8, 0.01).name('Anim crossfade (s)');

characterFolder.close();
const characterOffsetFolder = characterFolder.addFolder('Character offset');
characterOffsetFolder.add(tuning, 'characterOffsetX', -2, 2, 0.01).name('X');
characterOffsetFolder.add(tuning, 'characterOffsetY', -2, 6, 0.01).name('Y (up)');
characterOffsetFolder.add(tuning, 'characterOffsetZ', -2, 2, 0.01).name('Z');
characterOffsetFolder.close();

const lightingFolder = gui.addFolder('Lighting');
lightingFolder.add(tuning, 'ambientIntensity', 0, 2, 0.01).name('Ambient');
lightingFolder.add(tuning, 'sunIntensity', 0, 3, 0.02).name('Sun intensity');
lightingFolder.addColor(tuning, 'sunColor').name('Sun color');
lightingFolder.add(tuning, 'sunShadowFollowCharacter').name('Sun follows character');
lightingFolder.add(tuning, 'sunPosX', -120, 120, 0.5).name('Sun X (offset if follow)');
lightingFolder.add(tuning, 'sunPosY', -20, 120, 0.5).name('Sun Y (offset if follow)');
lightingFolder.add(tuning, 'sunPosZ', -120, 120, 0.5).name('Sun Z (offset if follow)');
lightingFolder.close();

const shadowFolder = gui.addFolder('Shadows');
shadowFolder.add(tuning, 'shadowIntensity', 0, 1, 0.01).name('Intensity');
shadowFolder.add(tuning, 'shadowRadius', 0, 10, 0.1).name('Radius (softness)');
shadowFolder.add(tuning, 'shadowBias', -0.005, 0.005, 0.0001).name('Bias');
shadowFolder.add(tuning, 'shadowMapSize', 256, 4096, 128).name('Map size');
shadowFolder
  .add(tuning, 'shadowMapType', ['BasicShadowMap', 'PCFShadowMap', 'PCFSoftShadowMap', 'VSMShadowMap'])
  .name('Map filter');
shadowFolder.add(tuning, 'shadowCameraHalfExtent', 5, 100, 1).name('Frustum size');
shadowFolder.add(tuning, 'shadowCameraNear', 0.1, 50, 0.1).name('Near');
shadowFolder.add(tuning, 'shadowCameraFar', 10, 400, 1).name('Far');
shadowFolder.add(tuning, 'colliderGlbShadowOpacity', 0, 1, 0.01).name('Floor shadow opacity');
shadowFolder.addColor(tuning, 'colliderGlbShadowColor').name('Floor shadow color');
shadowFolder.close();

const ppFolder = gui.addFolder('Post-processing');
ppFolder.add(tuning, 'ppEnabled').name('Enabled');
ppFolder.add(tuning, 'ppBloomIntensity', 0, 2, 0.01).name('Bloom intensity');
ppFolder.add(tuning, 'ppBloomThreshold', 0, 1, 0.01).name('Bloom threshold');
ppFolder.add(tuning, 'ppBloomSmoothing', 0, 1, 0.01).name('Bloom smoothing');
ppFolder.add(tuning, 'ppBrightness', -1, 1, 0.02).name('Brightness');
ppFolder.add(tuning, 'ppContrast', -1, 1, 0.02).name('Contrast');
ppFolder.add(tuning, 'ppVignetteDarkness', 0, 1, 0.01).name('Vignette darkness');
ppFolder.add(tuning, 'ppVignetteOffset', 0, 1, 0.01).name('Vignette offset');
ppFolder.close();

const worldFolder = gui.addFolder('World');
worldFolder
  .add(tuning, 'splatUniformScale', 0.05, 10, 0.01)
  .name('Splat scale')
  .onChange(() => {
    splatMesh.scale.setScalar(tuning.splatUniformScale);
  });
worldFolder
  .add(tuning, 'colliderGlbUniformScale', 0.05, 10, 0.01)
  .name('Collider scale')
  .onChange(() => {
    bakeColliderPhysics();
  });
worldFolder.close();


const dbgFolder = gui.addFolder('Physics debug (Rapier)');
dbgFolder.add(tuning, 'showPhysicsDebug').name('Enabled');
dbgFolder.add(tuning, 'debugBodies').name('Collider wireframe');
dbgFolder.close();

gui.add(
  {
    resetCharacter() {
      playerBody.setTranslation({ x: spawnX, y: spawnFeetY + capsuleCenterY, z: spawnZ }, true);
      playerBody.setLinvel({ x: 0, y: 0, z: 0 }, true);
      characterState.linearVelocity.set(0, 0, 0);
      characterState.desiredVelocity.set(0, 0, 0);
      characterState.grounded = false;
      characterAnimStableGrounded = false;
      characterAnimAirAccum = 0;
      characterSmoothedHSpeed = 0;
    },
  },
  'resetCharacter',
).name('Reset position');

// ─── Main loop ──────────────────────────────────────────────────────────────
function animate() {
  requestAnimationFrame(animate);

  const currentTime = performance.now();
  const deltaTime = Math.min((currentTime - lastTime) / 1000, maxDelta);
  lastTime = currentTime;

  rapierWorld.gravity = { x: 0, y: tuning.gravityY, z: 0 };

  // NAMMA: benchmark fly-through, then driving, otherwise the template's walking code below
  if (nammaHud.frame(deltaTime, camera, nammaCar.active ? `driving ${Math.round(nammaCar.kmh)} km/h` : 'walking')) {
    composer.render(deltaTime);
    return;
  }
  if (nammaCar.active) {
    nammaCar.step(deltaTime);
    nammaCar.followCamera(camera);
    probeLighting.update(nammaCar.group.position);
    syncLightingAndShadowsFromTuning();
    composer.render(deltaTime);
    return;
  }

  syncCharacterControllerFromTuning();

  const cameraRotation = new THREE.Quaternion();
  camera.getWorldQuaternion(cameraRotation);
  const forward = input.forwardPressed ? 1 : input.backwardPressed ? -1 : 0;
  const right = input.rightPressed ? 1 : input.leftPressed ? -1 : 0;
  const cameraDirection = new THREE.Vector3(right, 0, -forward).applyQuaternion(cameraRotation);
  cameraDirection.y = 0;
  if (cameraDirection.lengthSq() > 1e-8) {
    cameraDirection.normalize();
  }

  handleCharacterInput(characterState, deltaTime, cameraDirection);

  characterController.setSlideEnabled(characterState.allowSliding);

  const oldFeet = new THREE.Vector3();
  {
    const t = playerBody.translation();
    oldFeet.set(t.x, t.y - capsuleCenterY, t.z);
  }

  const desired = new THREE.Vector3(
    characterState.linearVelocity.x * deltaTime,
    characterState.linearVelocity.y * deltaTime,
    characterState.linearVelocity.z * deltaTime,
  );
  characterController.computeColliderMovement(playerCollider, desired);
  const movement = characterController.computedMovement();

  const t0 = playerBody.translation();
  playerBody.setNextKinematicTranslation({
    x: t0.x + movement.x,
    y: t0.y + movement.y,
    z: t0.z + movement.z,
  });

  rapierWorld.timestep = deltaTime;
  rapierWorld.step();

  characterState.grounded = characterController.computedGrounded();
  if (deltaTime > 1e-8) {
    characterState.linearVelocity.set(
      movement.x / deltaTime,
      movement.y / deltaTime,
      movement.z / deltaTime,
    );
  }

  updateCharacterAnimStability(deltaTime, characterState.grounded);

  syncReflectorBallsFromPhysics();

  const newFeet = new THREE.Vector3(
    playerBody.translation().x,
    playerBody.translation().y - capsuleCenterY,
    playerBody.translation().z,
  );
  const deltaPos = new THREE.Vector3().subVectors(newFeet, oldFeet);
  camera.position.add(deltaPos);
  controls.target.set(
    newFeet.x,
    newFeet.y + CAMERA_ORBIT_TARGET_Y_OFFSET,
    newFeet.z,
  );

  syncLightingAndShadowsFromTuning();
  syncPostProcessingFromTuning();
  syncRapierDebugFromTuning();
  updateRapierDebugLines();

  syncPlayerRoot();
  probeLighting.update(newFeet); // NAMMA
  updateCharacterMovementFacing(deltaTime, cameraDirection);
  updateCharacterAnimations(deltaTime);
  controls.update();
  composer.render(deltaTime);
}

// ─── HUD ────────────────────────────────────────────────────────────────────
const hintsEl = document.createElement('div');
hintsEl.style.cssText = [
  'position:fixed', 'bottom:16px', 'left:16px',
  'color:rgba(255,255,255,0.55)', 'font-size:13px',
  'font-family:system-ui,sans-serif', 'line-height:1.7',
  'pointer-events:none', 'user-select:none',
  'text-shadow:0 1px 3px rgba(0,0,0,0.6)',
].join(';');
hintsEl.innerHTML =
  'WASD &mdash; Move &nbsp;&nbsp; Space &mdash; Jump &nbsp;&nbsp; Shift &mdash; Sprint &nbsp;&nbsp; C &mdash; Car &nbsp;&nbsp; B &mdash; Benchmark<br>' +
  'Click + drag &mdash; Orbit camera &nbsp;&nbsp; Scroll &mdash; Zoom';
document.body.appendChild(hintsEl);

const loadingEl = document.createElement('div');
loadingEl.style.cssText = [
  'position:fixed', 'bottom:16px', 'left:50%', 'transform:translateX(-50%)',
  'color:rgba(255,255,255,0.65)', 'font-size:13px',
  'font-family:system-ui,sans-serif',
  'pointer-events:none', 'user-select:none',
  'text-shadow:0 1px 3px rgba(0,0,0,0.6)',
].join(';');
loadingEl.textContent = 'Loading splat\u2026';
document.body.appendChild(loadingEl);
splatMesh.initialized.then(() => {
  loadingEl.remove();
});

syncPlayerRoot();
animate();

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
  composer.setSize(window.innerWidth, window.innerHeight);
});
