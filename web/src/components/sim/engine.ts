// Renders the MuJoCo room and the walking G1 with three.js.
//
// Everything comes from web/public/sim, written by scripts/sim/export.py from the scene XML:
// the room's geoms and materials as MuJoCo compiled them, the robot's body tree and meshes,
// and one lap of the walk as qpos per frame. Poses are applied exactly the way MuJoCo does
// forward kinematics (body offset, then hinge rotation), so what you see is the sim's pose.

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { mergeGeometries, toCreasedNormals } from "three/examples/jsm/utils/BufferGeometryUtils.js";
import { CAMERA as TRIPOD, ZONE } from "@/lib/capture";

type V3 = [number, number, number];
type V4 = [number, number, number, number];

interface MjMaterial {
  name: string;
  rgba: V4;
  emission: number;
  specular: number;
  reflectance: number;
  tex: string | null;
  repeat: [number, number];
}
interface RoomGeom {
  t: "box" | "cylinder" | "capsule" | "sphere" | "plane" | "ellipsoid";
  m: number;
  c: V4 | null;
  s: V3;
  p: V3;
  q: V4;
  n: string | null;
}
interface Body {
  name: string;
  parent: number;
  pos: V3;
  quat: V4;
  joint: { q: number; axis: V3 } | null;
  free: boolean;
}
interface Footstep {
  x: number;
  y: number;
  yaw: number;
  side: number;
  t: number;
}
export interface SceneFile {
  source: string;
  materials: MjMaterial[];
  room: RoomGeom[];
  lights: { name: string; pos: V3; dir: V3; diffuse: V3; cutoff: number; directional: boolean }[];
  robot: {
    bodies: Body[];
    geoms: { body: number; mesh: number; pos: V3; quat: V4; rgba: V4 }[];
    meshes: { name: string; verts: number; tris: number; pos: number; idx: number; wide: boolean }[];
    nq: number;
  };
  walk: { fps: number; frames: number; duration: number; speed: number; steps_per_lap: number; footsteps: Footstep[] };
}

export type Mode = "follow" | "film";
export interface Anchor {
  id: "zone" | "tripod";
  x: number;
  y: number;
  visible: boolean;
}

const BASE = "/sim/";
const BG = new THREE.Color("#0e0d0c");
const AMBER = new THREE.Color("#ffa22e");
const quat = (q: V4) => new THREE.Quaternion(q[1], q[2], q[3], q[0]); // MuJoCo is w,x,y,z

// room extents (inside the walls), used to keep the camera indoors
const ROOM = { x0: -5.75, x1: 5.75, y0: -6.85, y1: 5.55, z0: 0.5, z1: 3.3 };

const FILM_VIEW = {
  pos: new THREE.Vector3(TRIPOD.x + 1.2, TRIPOD.y - 2.4, 3.25),
  target: new THREE.Vector3((ZONE.x + TRIPOD.x) / 2 - 0.2, (ZONE.y + TRIPOD.y) / 2 + 0.4, 0.2),
};
// where the camera sits relative to the robot when nobody is steering: up the aisle from
// the entrance side, high enough to see over the tables, with the stage screen behind
const FOLLOW_DIR = new THREE.Vector3(0.32, -0.86, 0.52).normalize();
const FOLLOW_DIST = 5.0;

const easeInOut = (u: number) => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2);

function primitive(g: RoomGeom): THREE.BufferGeometry {
  const [a, b, c] = g.s;
  switch (g.t) {
    case "box":
      return new THREE.BoxGeometry(2 * a, 2 * b, 2 * c);
    case "cylinder":
      return new THREE.CylinderGeometry(a, a, 2 * b, 20).rotateX(Math.PI / 2);
    case "capsule":
      return new THREE.CapsuleGeometry(a, 2 * b, 4, 12).rotateX(Math.PI / 2);
    case "sphere":
      return new THREE.SphereGeometry(a, 16, 12);
    case "ellipsoid":
      return new THREE.SphereGeometry(1, 16, 12).scale(a, b, c);
    case "plane":
      return new THREE.PlaneGeometry(a > 0 ? 2 * a : 60, b > 0 ? 2 * b : 60);
  }
}

// MuJoCo materials carry no "emissive" flag for these, so screens and lamps are picked by name
const GLOW: Record<string, number> = {
  room_screen_textured: 1.1,
  v2_screen_0: 1.2,
  v2_screen_1: 1.2,
  v2_screen_2: 1.2,
  v2_screen_3: 1.2,
  v2_screen_4: 1.2,
  room_blue: 2.4,
  v2_blue_glow: 2.4,
  v2_teal_glow: 2,
  room_glow: 1.6,
  v2_warm_glass: 1.4,
  room_marker: 0.35,
};

export class SimEngine {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera = new THREE.PerspectiveCamera(42, 1, 0.05, 60);
  private controls: OrbitControls;
  private data!: SceneFile;
  private walk!: Float32Array;
  private pelvis!: THREE.Object3D;
  private jointed: { node: THREE.Object3D; base: THREE.Quaternion; axis: THREE.Vector3; q: number }[] = [];
  private follow!: THREE.SpotLight;
  private steps!: THREE.InstancedMesh;
  private zone!: THREE.Group;
  private t = 0;
  private raf = 0;
  private last = 0;
  private running = false;
  private mode: Mode = "follow";
  private tween: { from: [THREE.Vector3, THREE.Vector3]; to: () => [THREE.Vector3, THREE.Vector3]; u: number } | null = null;
  private interacting = false;
  private size = { w: 1, h: 1 };
  private tmpQ = new THREE.Quaternion();
  private tmpQ2 = new THREE.Quaternion();
  private robotAt = new THREE.Vector3();
  private followDir = FOLLOW_DIR.clone();
  private clock = 0;

  constructor(
    canvas: HTMLCanvasElement,
    private opts: { reducedMotion: boolean; coarse: boolean; onAnchors?: (a: Anchor[]) => void },
  ) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, opts.coarse ? 1.5 : 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.05;
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.scene.background = BG;
    this.scene.fog = new THREE.FogExp2(BG, 0.055);

    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.enableZoom = false;
    this.controls.enablePan = false;
    this.controls.rotateSpeed = 0.5;
    this.controls.minPolarAngle = 0.75;
    this.controls.maxPolarAngle = 1.45;
    this.controls.enableRotate = !opts.coarse; // on phones a drag should scroll the page
    this.controls.addEventListener("start", () => (this.interacting = true));
    this.controls.addEventListener("end", () => (this.interacting = false));
  }

  async load(onProgress: (p: number) => void) {
    let done = 0;
    let total = 3;
    const tick = () => onProgress(++done / total);
    const get = async (f: string) => {
      const r = await fetch(BASE + f);
      if (!r.ok) throw new Error(`${f}: HTTP ${r.status}`);
      return r;
    };
    const [data, robotBuf, walkBuf] = await Promise.all([
      get("scene.json").then((r) => r.json() as Promise<SceneFile>).finally(tick),
      get("robot.bin").then((r) => r.arrayBuffer()).finally(tick),
      get("walk.bin").then((r) => r.arrayBuffer()).finally(tick),
    ]);
    this.data = data;
    this.walk = new Float32Array(walkBuf);

    const files = [...new Set(data.materials.map((m) => m.tex).filter((t): t is string => !!t))];
    total += files.length;
    const loader = new THREE.TextureLoader();
    const tex = new Map<string, THREE.Texture>();
    await Promise.all(
      files.map((f) =>
        loader
          .loadAsync(BASE + "tex/" + f)
          .then((t) => tex.set(f, t))
          .catch(() => {})
          .finally(tick),
      ),
    );

    this.buildRoom(tex);
    this.buildLights();
    this.buildRobot(robotBuf);
    this.buildFootsteps();
    this.buildZone();
    this.applyPose(0);
    this.placeCamera();
  }

  // ------------------------------------------------------------------ build

  private material(mm: MjMaterial, rgba: V4, tex: Map<string, THREE.Texture>) {
    const color = new THREE.Color().setRGB(rgba[0], rgba[1], rgba[2], THREE.SRGBColorSpace);
    const mat = new THREE.MeshStandardMaterial({
      color,
      roughness: THREE.MathUtils.clamp(0.95 - mm.specular * 0.5 - mm.reflectance * 1.6, 0.3, 0.95),
      metalness: 0,
      transparent: rgba[3] < 0.99,
      opacity: rgba[3],
    });
    if (mm.tex && tex.has(mm.tex)) {
      const t = tex.get(mm.tex)!.clone();
      t.wrapS = t.wrapT = THREE.RepeatWrapping;
      t.repeat.set(mm.repeat[0], mm.repeat[1]);
      t.colorSpace = THREE.SRGBColorSpace;
      t.anisotropy = 4;
      t.needsUpdate = true;
      mat.map = t;
    }
    const glow = GLOW[mm.name];
    if (glow) {
      mat.emissive = mat.map ? new THREE.Color(1, 1, 1) : color.clone();
      if (mat.map) mat.emissiveMap = mat.map;
      mat.emissiveIntensity = glow;
    }
    return mat;
  }

  private buildRoom(tex: Map<string, THREE.Texture>) {
    const groups = new Map<string, { mat: THREE.Material; geos: THREE.BufferGeometry[]; floor: boolean }>();
    const m4 = new THREE.Matrix4();
    const one = new THREE.Vector3(1, 1, 1);
    for (const g of this.data.room) {
      const mm = this.data.materials[g.m];
      const rgba = g.c ?? mm?.rgba ?? [0.5, 0.5, 0.5, 1];
      const key = `${g.m}|${g.c?.join(",") ?? ""}`;
      if (!groups.has(key)) {
        const mat = mm ? this.material(mm, rgba, tex) : new THREE.MeshStandardMaterial({ color: new THREE.Color().setRGB(rgba[0], rgba[1], rgba[2], THREE.SRGBColorSpace) });
        groups.set(key, { mat, geos: [], floor: !!mm && /floor/.test(mm.name) });
      }
      const geo = primitive(g);
      geo.applyMatrix4(m4.compose(new THREE.Vector3(...g.p), quat(g.q), one));
      groups.get(key)!.geos.push(geo);
    }
    for (const { mat, geos, floor } of groups.values()) {
      const mesh = new THREE.Mesh(mergeGeometries(geos, false), mat);
      mesh.receiveShadow = true;
      mesh.castShadow = !floor;
      this.scene.add(mesh);
      geos.forEach((g) => g.dispose());
    }
  }

  private buildLights() {
    this.scene.add(new THREE.HemisphereLight(0xb9c2d6, 0x1c1610, 0.55));
    for (const l of this.data.lights) {
      if (l.directional) continue; // the sun: it's night in here
      const warm = l.diffuse[0] >= l.diffuse[2];
      const s = new THREE.SpotLight(warm ? 0xffd2a1 : 0x6f8cff, warm ? 26 : 40, 0, THREE.MathUtils.degToRad(Math.min(l.cutoff, 62)), 0.85, 2);
      s.position.set(...l.pos);
      s.target.position.set(l.pos[0] + l.dir[0], l.pos[1] + l.dir[1], l.pos[2] + l.dir[2] * 3);
      this.scene.add(s, s.target);
    }
    // a follow-spot on the robot, like a performer on stage; it's also what casts its shadow
    this.follow = new THREE.SpotLight(0xfff1dc, 70, 0, THREE.MathUtils.degToRad(17), 0.65, 2);
    this.follow.castShadow = true;
    this.follow.shadow.mapSize.set(this.opts.coarse ? 1024 : 2048, this.opts.coarse ? 1024 : 2048);
    this.follow.shadow.bias = -0.0004;
    this.follow.shadow.camera.near = 1;
    this.follow.shadow.camera.far = 12;
    this.scene.add(this.follow, this.follow.target);
  }

  private buildRobot(buf: ArrayBuffer) {
    const { bodies, geoms, meshes } = this.data.robot;
    const geos = meshes.map((m) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(new Float32Array(buf, m.pos, m.verts * 3), 3));
      g.setIndex(new THREE.BufferAttribute(m.wide ? new Uint32Array(buf, m.idx, m.tris * 3) : new Uint16Array(buf, m.idx, m.tris * 3), 1));
      return toCreasedNormals(g, 0.55);
    });
    const mats = new Map<string, THREE.MeshStandardMaterial>();
    const matFor = (rgba: V4) => {
      const k = rgba.join(",");
      if (!mats.has(k)) {
        const light = rgba[0] > 0.5;
        mats.set(
          k,
          new THREE.MeshStandardMaterial({
            color: new THREE.Color().setRGB(rgba[0], rgba[1], rgba[2], THREE.SRGBColorSpace),
            roughness: light ? 0.38 : 0.55,
            metalness: light ? 0.25 : 0.1,
          }),
        );
      }
      return mats.get(k)!;
    };
    const nodes = bodies.map((b) => {
      const o = new THREE.Object3D();
      o.name = b.name;
      o.position.set(...b.pos);
      o.quaternion.copy(quat(b.quat));
      return o;
    });
    bodies.forEach((b, i) => {
      if (b.parent >= 0) nodes[b.parent].add(nodes[i]);
      else this.scene.add(nodes[i]);
      if (b.free) this.pelvis = nodes[i];
      if (b.joint) this.jointed.push({ node: nodes[i], base: quat(b.quat), axis: new THREE.Vector3(...b.joint.axis), q: b.joint.q });
    });
    for (const g of geoms) {
      const mesh = new THREE.Mesh(geos[g.mesh], matFor(g.rgba));
      mesh.position.set(...g.pos);
      mesh.quaternion.copy(quat(g.quat));
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      nodes[g.body].add(mesh);
    }
  }

  private buildFootsteps() {
    const c = document.createElement("canvas");
    c.width = 64;
    c.height = 32;
    const x = c.getContext("2d")!;
    x.fillStyle = "#fff";
    x.beginPath();
    x.roundRect(4, 6, 56, 20, 10);
    x.fill();
    const map = new THREE.CanvasTexture(c);
    const steps = this.data.walk.footsteps;
    this.steps = new THREE.InstancedMesh(
      new THREE.PlaneGeometry(0.2, 0.085),
      new THREE.MeshBasicMaterial({ map, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, fog: false }),
      steps.length,
    );
    const m4 = new THREE.Matrix4();
    steps.forEach((s, i) => {
      const cx = s.x + 0.035 * Math.cos(s.yaw);
      const cy = s.y + 0.035 * Math.sin(s.yaw);
      m4.compose(new THREE.Vector3(cx, cy, 0.004), new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), s.yaw), new THREE.Vector3(1, 1, 1));
      this.steps.setMatrixAt(i, m4);
      this.steps.setColorAt(i, new THREE.Color(0, 0, 0));
    });
    this.steps.renderOrder = 2;
    this.scene.add(this.steps);
  }

  private buildZone() {
    const g = new THREE.Group();
    const line = new THREE.MeshBasicMaterial({ color: AMBER, transparent: true, opacity: 0.9, depthWrite: false, fog: false });
    const fill = new THREE.Mesh(
      new THREE.PlaneGeometry(ZONE.w, ZONE.d),
      new THREE.MeshBasicMaterial({ color: AMBER, transparent: true, opacity: 0.06, depthWrite: false, blending: THREE.AdditiveBlending }),
    );
    fill.position.set(ZONE.x, ZONE.y, 0.003);
    g.add(fill);
    // dashed outline
    const dash = 0.2;
    const gap = 0.12;
    const edges: [number, number, number, number][] = [
      [-1, -1, 1, -1],
      [1, -1, 1, 1],
      [1, 1, -1, 1],
      [-1, 1, -1, -1],
    ];
    for (const [ax, ay, bx, by] of edges) {
      const x0 = ZONE.x + (ax * ZONE.w) / 2;
      const y0 = ZONE.y + (ay * ZONE.d) / 2;
      const x1 = ZONE.x + (bx * ZONE.w) / 2;
      const y1 = ZONE.y + (by * ZONE.d) / 2;
      const len = Math.hypot(x1 - x0, y1 - y0);
      const ang = Math.atan2(y1 - y0, x1 - x0);
      for (let s = 0; s < len; s += dash + gap) {
        const l = Math.min(dash, len - s);
        const seg = new THREE.Mesh(new THREE.PlaneGeometry(l, 0.03), line);
        const mid = s + l / 2;
        seg.position.set(x0 + Math.cos(ang) * mid, y0 + Math.sin(ang) * mid, 0.005);
        seg.rotation.z = ang;
        g.add(seg);
      }
    }
    // tripod with a phone on it, at hip height, aimed at the middle of the zone
    const metal = new THREE.MeshStandardMaterial({ color: 0x2a2a2c, roughness: 0.5, metalness: 0.4 });
    const head = new THREE.Vector3(TRIPOD.x, TRIPOD.y, 0.95);
    for (let i = 0; i < 3; i++) {
      const a = (i / 3) * Math.PI * 2;
      const foot = new THREE.Vector3(TRIPOD.x + 0.32 * Math.cos(a), TRIPOD.y + 0.32 * Math.sin(a), 0);
      const dir = new THREE.Vector3().subVectors(head, foot);
      const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.011, 0.011, dir.length(), 8), metal);
      leg.position.copy(foot).addScaledVector(dir, 0.5);
      leg.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.normalize());
      leg.castShadow = true;
      g.add(leg);
    }
    const aim = Math.atan2(ZONE.y - TRIPOD.y, ZONE.x - TRIPOD.x);
    const phone = new THREE.Mesh(new THREE.BoxGeometry(0.012, 0.16, 0.078), new THREE.MeshStandardMaterial({ color: 0x111114, roughness: 0.3 }));
    phone.position.copy(head).add(new THREE.Vector3(0, 0, 0.05));
    phone.rotation.z = aim;
    phone.castShadow = true;
    const lens = new THREE.Mesh(new THREE.SphereGeometry(0.012, 12, 8), new THREE.MeshBasicMaterial({ color: AMBER, fog: false }));
    lens.position.set(0.008, 0.05, 0.015);
    phone.add(lens);
    g.add(phone);
    this.zone = g;
    this.scene.add(g);
  }

  // ------------------------------------------------------------------ pose

  private applyPose(t: number) {
    const { frames, fps } = this.data.walk;
    const nq = this.data.robot.nq;
    const f = (t * fps) % frames;
    const i0 = Math.floor(f);
    const i1 = (i0 + 1) % frames;
    const a = f - i0;
    const q0 = i0 * nq;
    const q1 = i1 * nq;
    const w = this.walk;
    const lerp = (k: number) => w[q0 + k] + (w[q1 + k] - w[q0 + k]) * a;
    this.pelvis.position.set(lerp(0), lerp(1), lerp(2));
    this.tmpQ.set(w[q0 + 4], w[q0 + 5], w[q0 + 6], w[q0 + 3]);
    this.tmpQ2.set(w[q1 + 4], w[q1 + 5], w[q1 + 6], w[q1 + 3]);
    this.pelvis.quaternion.slerpQuaternions(this.tmpQ, this.tmpQ2, a);
    for (const j of this.jointed) {
      j.node.quaternion.copy(j.base).multiply(this.tmpQ.setFromAxisAngle(j.axis, lerp(j.q)));
    }
    this.robotAt.copy(this.pelvis.position);
  }

  private updateFootsteps(t: number) {
    const { footsteps, duration } = this.data.walk;
    const FADE = 7;
    const c = new THREE.Color();
    footsteps.forEach((s, i) => {
      const age = (t - s.t + duration * 2) % duration;
      const k = age < FADE ? Math.pow(1 - age / FADE, 1.6) * 0.5 : 0;
      this.steps.setColorAt(i, c.copy(AMBER).multiplyScalar(k));
    });
    this.steps.instanceColor!.needsUpdate = true;
  }

  // ------------------------------------------------------------------ camera

  private followPose(): [THREE.Vector3, THREE.Vector3] {
    const target = this.robotAt.clone().setZ(0.72);
    const pos = target.clone().addScaledVector(this.followDir, this.followDist());
    return [this.clampRoom(pos), target];
  }

  // a portrait screen sees a narrow slice of the room, so stand back a bit further
  private followDist() {
    return FOLLOW_DIST * (this.size.w < this.size.h ? 1.3 : 1);
  }

  private clampRoom(p: THREE.Vector3) {
    p.x = THREE.MathUtils.clamp(p.x, ROOM.x0, ROOM.x1);
    p.y = THREE.MathUtils.clamp(p.y, ROOM.y0, ROOM.y1);
    p.z = THREE.MathUtils.clamp(p.z, ROOM.z0, ROOM.z1);
    return p;
  }

  private placeCamera() {
    const [pos, target] = this.followPose();
    this.camera.position.copy(pos);
    this.controls.target.copy(target);
    this.controls.update();
  }

  info() {
    const w = this.data.walk;
    return { source: this.data.source, lap: w.duration, steps: w.steps_per_lap, speed: w.speed };
  }

  setMode(mode: Mode) {
    if (mode === this.mode) return;
    this.mode = mode;
    if (mode === "follow") this.followDir.copy(FOLLOW_DIR);
    this.tween = {
      from: [this.camera.position.clone(), this.controls.target.clone()],
      to: mode === "film" ? () => [FILM_VIEW.pos.clone(), FILM_VIEW.target.clone()] : () => this.followPose(),
      u: 0,
    };
    if (!this.running) this.renderOnce();
  }

  private updateCamera(dt: number) {
    if (this.tween) {
      this.tween.u = Math.min(1, this.tween.u + dt / 1.6);
      const e = easeInOut(this.tween.u);
      const [p, tg] = this.tween.to();
      this.camera.position.lerpVectors(this.tween.from[0], p, e);
      this.controls.target.lerpVectors(this.tween.from[1], tg, e);
      if (this.tween.u >= 1) this.tween = null;
      this.camera.lookAt(this.controls.target);
      return;
    }
    if (this.mode === "follow") {
      // keep the user's angle, but carry the camera along with the robot
      const before = this.controls.target.clone();
      const goal = this.robotAt.clone().setZ(0.72);
      this.controls.target.lerp(goal, 1 - Math.exp(-dt * 2.5));
      this.camera.position.add(this.controls.target.clone().sub(before));
      if (!this.interacting && !this.opts.reducedMotion) {
        // ease back to the framing above after a drag, with a slow sway so it never sits dead still
        this.clock += dt;
        const home = FOLLOW_DIR.clone().applyAxisAngle(new THREE.Vector3(0, 0, 1), 0.22 * Math.sin(this.clock * 0.11));
        const off = this.camera.position.clone().sub(this.controls.target);
        const dist = THREE.MathUtils.lerp(off.length(), this.followDist(), 1 - Math.exp(-dt * 0.6));
        off.normalize().lerp(home, 1 - Math.exp(-dt * 0.35)).normalize().multiplyScalar(dist);
        this.camera.position.copy(this.controls.target).add(off);
      }
    }
    this.controls.update();
    this.clampRoom(this.camera.position);
    this.followDir.copy(this.camera.position).sub(this.controls.target).normalize();
  }

  private anchors() {
    if (!this.opts.onAnchors) return;
    const show = this.mode === "film" && (!this.tween || this.tween.u > 0.85);
    const pts: [Anchor["id"], THREE.Vector3][] = [
      ["zone", new THREE.Vector3(ZONE.x, ZONE.y, 0.02)],
      ["tripod", new THREE.Vector3(TRIPOD.x, TRIPOD.y, 1.08)],
    ];
    this.opts.onAnchors(
      pts.map(([id, p]) => {
        const v = p.project(this.camera);
        return { id, x: ((v.x + 1) / 2) * this.size.w, y: ((1 - v.y) / 2) * this.size.h, visible: show && v.z < 1 };
      }),
    );
  }

  // ------------------------------------------------------------------ loop

  resize(w: number, h: number) {
    this.size = { w, h };
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    // put the robot off-centre: right of the copy on wide screens, above it on tall ones
    if (w / h > 1.2) this.camera.setViewOffset(w, h, -w * 0.17, 0, w, h);
    else this.camera.setViewOffset(w, h, 0, h * 0.14, w, h);
    this.camera.updateProjectionMatrix();
    if (!this.running) this.renderOnce();
  }

  private frame = (now: number) => {
    // a rAF timestamp can be earlier than the performance.now() taken in start()
    const dt = Math.max(0, Math.min(0.05, (now - this.last) / 1000 || 0));
    this.last = Math.max(this.last, now);
    const lap = this.data.walk.duration;
    if (!this.opts.reducedMotion) this.t = (((this.t + dt) % lap) + lap) % lap;
    this.applyPose(this.t);
    this.updateFootsteps(this.t);
    this.follow.position.set(this.robotAt.x + 0.4, this.robotAt.y - 0.6, 3.3);
    this.follow.target.position.set(this.robotAt.x, this.robotAt.y, 0.4);
    this.updateCamera(dt);
    this.renderer.render(this.scene, this.camera);
    this.anchors();
    if (this.running) this.raf = requestAnimationFrame(this.frame);
  };

  renderOnce() {
    if (!this.data) return;
    this.last = performance.now();
    const was = this.running;
    this.running = false;
    this.frame(this.last);
    this.running = was;
  }

  start() {
    if (this.running || !this.data) return;
    this.running = true;
    this.last = performance.now();
    this.raf = requestAnimationFrame(this.frame);
  }

  stop() {
    this.running = false;
    cancelAnimationFrame(this.raf);
  }

  dispose() {
    this.stop();
    this.controls.dispose();
    this.scene.traverse((o) => {
      const m = o as THREE.Mesh;
      m.geometry?.dispose();
      const mat = m.material as THREE.Material | THREE.Material[] | undefined;
      (Array.isArray(mat) ? mat : mat ? [mat] : []).forEach((x) => x.dispose());
    });
    this.renderer.dispose();
  }
}
