// A few ideas for what to film. They're only suggestions: people can upload anything, and
// they name each video themselves.

export interface Example {
  id: string;
  name: string;
  hint: string;
}

export const EXAMPLES: Example[] = [
  { id: "move_water_bottle", name: "Move the water bottle", hint: "Pick it up off a table and set it down somewhere else." },
  { id: "move_chair", name: "Move a chair", hint: "Slide it, lift it, or turn it to face a new way." },
  { id: "back_flip", name: "Back flip", hint: "Only if you can already land one. On a mat, with a spotter." },
  { id: "carry_box", name: "Carry a box", hint: "Lift it off the floor, walk it across the room, put it down." },
  { id: "get_up_off_floor", name: "Get up off the floor", hint: "Lie down, then stand back up any way you like." },
];

export const exampleFor = (what: string) => EXAMPLES.find((e) => e.name.toLowerCase() === what.trim().toLowerCase());
