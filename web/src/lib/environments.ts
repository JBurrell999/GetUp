// The places people can say they filmed in. Each one is a MuJoCo scene in g1_room_mujoco/,
// and a take records which one so the robot copies it in a matching room.
// Screenshots: web/scripts/sim/screenshots.py renders public/envs/<id>-<n>[-sm].webp.

import { asset } from "./site";

export const ENV_IDS = ["hackathon", "kitchen", "warehouse"] as const;
export type EnvId = (typeof ENV_IDS)[number];

export interface Shot {
  /** 1280 px wide; the "-sm" file next to it is 640 px. */
  src: string;
  small: string;
  alt: string;
}

export interface Environment {
  id: EnvId;
  name: string;
  /** A few words on where this is, shown above the name. */
  where: string;
  blurb: string;
  facts: string[];
  /** The scene the sim uses, relative to the repo root (written into each take's details). */
  scene: string;
  shots: Shot[];
}

const shots = (id: EnvId, alts: string[]): Shot[] =>
  alts.map((alt, i) => ({ src: asset(`/envs/${id}-${i + 1}.webp`), small: asset(`/envs/${id}-${i + 1}-sm.webp`), alt }));

export const ENVIRONMENTS: Environment[] = [
  {
    id: "hackathon",
    name: "Hackathon room",
    where: "Here at the event",
    blurb: "This room: worktables with laptops, red chairs, the stage and screen, and the refreshment bar.",
    facts: ["About 12 × 14 m", "Tables and chairs everywhere", "Clear aisle in front of the stage"],
    scene: "g1_room_mujoco/Hackathon_G1_Ultra_Detail_V2/generated/G1_HACKATHON_V2.xml",
    shots: shots("hackathon", [
      "The simulated G1 standing in the hackathon room, among black worktables and red chairs, facing the stage screen",
      "The hackathon room's centre aisle between worktables, looking toward the stage",
      "The refreshment bar along the wood-panelled wall, with drinks, fruit and a worktable",
    ]),
  },
  {
    id: "kitchen",
    name: "Kitchen",
    where: "At home",
    blurb: "An open-plan home kitchen: island with bar stools, counters, sink, range, fridge, pantry and a dining table for six.",
    facts: ["About 8 × 7 m", "Stools, dining chairs, island at 0.93 m", "Clear floor by the door"],
    scene: "g1_room_mujoco/Kitchen_G1_Detail/generated/G1_KITCHEN.xml",
    shots: shots("kitchen", [
      "The simulated G1 standing in an open-plan kitchen with an island, pendant lights and a set dining table",
      "Kitchen counters with a sink under a window, a gas range with a kettle and pot, and a stand mixer",
      "A stocked open pantry, wall oven, microwave and a fridge with a calendar and drawings on it",
    ]),
  },
  {
    id: "warehouse",
    name: "Warehouse",
    where: "At work",
    blurb: "A distribution warehouse: pallet racking, loading docks, a forklift, pack stations, and cartons and totes to lift.",
    facts: ["About 30 × 28 m, 9 m ceiling", "Cartons, totes, benches and stools", "Painted test zone to work in"],
    scene: "g1_room_mujoco/Warehouse_G1_Detail/generated/G1_WAREHOUSE.xml",
    shots: shots("warehouse", [
      "The simulated G1 standing in a blue robot test zone at the head of a warehouse aisle, with racking and pack stations",
      "The whole warehouse from above a corner: pallet racks, a forklift, pack stations and loading docks",
      "A warehouse aisle between tall blue and orange pallet racks loaded with cartons",
    ]),
  },
];

export const ENV_BY_ID = new Map(ENVIRONMENTS.map((e) => [e.id, e]));

export const isEnvId = (s: unknown): s is EnvId => typeof s === "string" && (ENV_IDS as readonly string[]).includes(s);
