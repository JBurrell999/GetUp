import raw from "@/data/tasks.json";

export type Tier = "core" | "extended" | "edge";

export interface Task {
  id: string;
  /** Position in tasks.json, 1-based. Shown as the scene number. */
  no: number;
  category: string;
  tier: Tier;
  name: string;
  instruction: string;
  stop: "still" | "time";
  max_s: number;
  still_s: number;
  props: string[];
  note?: string;
}

interface RawTask {
  id: string;
  category: string;
  tier: string;
  name: string;
  instruction: string;
  stop?: string;
  max_s?: number;
  still_s?: number;
  props?: string[];
  note?: string;
}

const file = raw as { setup: string; tasks: RawTask[] };

export const TASKS: Task[] = file.tasks.map((t, i) => ({
  id: t.id,
  no: i + 1,
  category: t.category,
  tier: t.tier as Tier,
  name: t.name,
  instruction: t.instruction,
  stop: t.stop === "time" ? "time" : "still",
  max_s: t.max_s ?? 15,
  still_s: t.still_s ?? 1.5,
  props: t.props ?? [],
  note: t.note,
}));

export const TASK_BY_ID = new Map(TASKS.map((t) => [t.id, t]));

/** Categories in the order they first appear in tasks.json. */
export const CATEGORIES = [...new Set(TASKS.map((t) => t.category))];

export const TIERS: { id: Tier; label: string; short: string; blurb: string }[] = [
  { id: "core", label: "Core", short: "Core", blurb: "The essential set. Start here." },
  { id: "extended", label: "Extended", short: "More", blurb: "Everything else people do in a day." },
  { id: "edge", label: "Edge cases", short: "Edge", blurb: "Hard on the camera on purpose." },
];

export const countByTier = (tier: Tier) => TASKS.filter((t) => t.tier === tier).length;

export const sceneNo = (t: Task) => String(t.no).padStart(3, "0");

const secs = (s: number) => `${Number.isInteger(s) ? s : s.toFixed(1)} s`;

/** Short form for the list column. */
export function lengthLabel(t: Task) {
  return t.stop === "time" ? `${secs(t.max_s)} fixed` : `up to ${secs(t.max_s)}`;
}

/** How a take of this task should end, written for the person filming. */
export function endHint(t: Task) {
  if (t.stop === "time") {
    return `Runs for ${secs(t.max_s)}. Keep going until then; the take is cut at that length.`;
  }
  return `Finish, then hold still for ${secs(t.still_s)} before you stop recording. Most takes are done well inside ${secs(t.max_s)}.`;
}

/** Turns "chair (measure seat height)" into a name plus an instruction. */
export function splitProp(p: string): { name: string; extra?: string } {
  const m = p.match(/^(.*?)\s*\((.*)\)\s*$/);
  return m ? { name: m[1], extra: m[2] } : { name: p };
}
