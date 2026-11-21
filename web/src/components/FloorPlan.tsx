import room from "@/data/room.json";
import { CAMERA, CAMERA_DIST, FACING_DEG, ZONE } from "@/lib/capture";

type Box = number[]; // [x, y, hx, hy]

// Main room only; the refreshments annex is off to the left.
const X0 = -6.4;
const X1 = 6.4;
const Y0 = -7.75;
const Y1 = 7.6;
const W = X1 - X0;
const H = Y1 - Y0;

// world (z-up, +y to the stage) -> svg (y down, stage at the top)
const sx = (x: number) => x - X0;
const sy = (y: number) => Y1 - y;

function Rects({ boxes, className }: { boxes: Box[]; className: string }) {
  return (
    <g className={className}>
      {boxes.map(([x, y, hx, hy], i) => (
        <rect key={i} x={sx(x - hx)} y={sy(y + hy)} width={hx * 2} height={hy * 2} />
      ))}
    </g>
  );
}

const deg = (a: number) => (a * 180) / Math.PI;

export function FloorPlan() {
  const zx = sx(ZONE.x - ZONE.w / 2);
  const zy = sy(ZONE.y + ZONE.d / 2);
  const px = sx(ZONE.x);
  const py = sy(ZONE.y);
  const cx = sx(CAMERA.x);
  const cy = sy(CAMERA.y);

  // camera looks at the judge
  const look = Math.atan2(py - cy, px - cx);
  const half = (31 * Math.PI) / 180; // ~62° horizontal field of view, a phone in landscape
  const reach = CAMERA_DIST + 1.6;
  const cone = [
    [cx, cy],
    [cx + reach * Math.cos(look - half), cy + reach * Math.sin(look - half)],
    [cx + reach * Math.cos(look + half), cy + reach * Math.sin(look + half)],
  ];

  // the judge's front, and the 45° arc to the camera, in svg angles
  const front = (-FACING_DEG * Math.PI) / 180;
  const toCam = Math.atan2(cy - py, cx - px);
  const arcR = 1.05;
  const arc = (a: number) => [px + arcR * Math.cos(a), py + arcR * Math.sin(a)];
  const [a0x, a0y] = arc(front);
  const [a1x, a1y] = arc(toCam);
  const mid = (front + toCam) / 2;

  // dimension line between judge and camera, pulled back from both ends
  const dx = Math.cos(toCam);
  const dy = Math.sin(toCam);
  const d0 = [px + dx * 0.45, py + dy * 0.45];
  const d1 = [cx - dx * 0.5, cy - dy * 0.5];
  const dimMid = [(d0[0] + d1[0]) / 2, (d0[1] + d1[1]) / 2];
  const dimAngle = deg(toCam) + (Math.abs(deg(toCam)) > 90 ? 180 : 0);

  const [g1x, g1y] = room.spawn;

  return (
    <svg
      className="plan"
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-labelledby="plan-title plan-desc"
      preserveAspectRatio="xMidYMid meet"
    >
      <title id="plan-title">Floor plan of the event room</title>
      <desc id="plan-desc">
        The capture area is a 4 by 3 metre rectangle in the open aisle in front of the stage. The camera stands about
        3.5 metres away, 45 degrees off the judge&apos;s front.
      </desc>
      <defs>
        <pattern id="hatch" width="0.22" height="0.22" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
          <line x1="0" y1="0" x2="0" y2="0.22" className="plan-hatch" />
        </pattern>
      </defs>

      <Rects boxes={room.floors} className="plan-floor" />
      <Rects boxes={room.stage} className="plan-stage" />
      <Rects boxes={room.tables} className="plan-table" />
      <Rects boxes={room.chairs} className="plan-chair" />
      <Rects boxes={room.columns} className="plan-column" />
      <Rects boxes={room.screen} className="plan-screen" />
      <Rects boxes={room.walls} className="plan-wall" />

      <text className="plan-label" x={sx(0)} y={sy(6.38)}>
        Stage
      </text>
      <text className="plan-label" x={sx(0)} y={sy(-6.95)}>
        Entrance
      </text>
      <text className="plan-label plan-label--start" x={sx(-5.75)} y={sy(-2.16)}>
        ← Refreshments
      </text>

      {/* where the sim's G1 starts */}
      <g className="plan-g1">
        <circle cx={sx(g1x)} cy={sy(g1y)} r={0.32} />
        <circle cx={sx(g1x)} cy={sy(g1y)} r={0.07} className="plan-g1-dot" />
        <text x={sx(g1x) + 0.52} y={sy(g1y) + 0.12}>
          G1 start
        </text>
      </g>

      {/* capture area */}
      <polygon className="plan-cone" points={cone.map((p) => p.join(",")).join(" ")} />
      <rect className="plan-zone" x={zx} y={zy} width={ZONE.w} height={ZONE.d} />
      <text className="plan-zone-label" x={zx + 0.18} y={zy + 0.46}>
        4 × 3 m clear
      </text>

      <line className="plan-front" x1={px} y1={py} x2={px + 1.35 * Math.cos(front)} y2={py + 1.35 * Math.sin(front)} />
      <path className="plan-arc" d={`M ${a0x} ${a0y} A ${arcR} ${arcR} 0 0 0 ${a1x} ${a1y}`} />
      <text className="plan-small" x={px + (arcR + 0.42) * Math.cos(mid)} y={py + (arcR + 0.42) * Math.sin(mid) + 0.12}>
        45°
      </text>

      <g className="plan-dim">
        <line x1={d0[0]} y1={d0[1]} x2={d1[0]} y2={d1[1]} />
        <text transform={`translate(${dimMid[0]} ${dimMid[1]}) rotate(${dimAngle}) translate(0 -0.18)`}>3–4 m</text>
      </g>

      <g className="plan-you">
        <circle cx={px} cy={py} r={0.3} />
        <text x={px - 0.42} y={py + 0.12}>
          You
        </text>
      </g>

      <g className="plan-camera" transform={`translate(${cx} ${cy}) rotate(${deg(look)})`}>
        <rect x={-0.42} y={-0.24} width={0.5} height={0.48} rx={0.05} />
        <path d="M 0.08 -0.11 L 0.3 -0.22 L 0.3 0.22 L 0.08 0.11 Z" />
      </g>
      <text className="plan-small" x={cx + 0.5} y={cy - 0.05}>
        Camera
      </text>

      {/* scale */}
      <g className="plan-scale" transform={`translate(${sx(3.6)} ${sy(-6.75)})`}>
        <line x1={0} y1={0} x2={2} y2={0} />
        <line x1={0} y1={-0.12} x2={0} y2={0.12} />
        <line x1={1} y1={-0.08} x2={1} y2={0.08} />
        <line x1={2} y1={-0.12} x2={2} y2={0.12} />
        <text x={1} y={-0.24}>
          2 m
        </text>
      </g>
    </svg>
  );
}
