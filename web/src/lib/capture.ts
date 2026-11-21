// Where people film, in the room's own coordinates (metres, +y toward the stage).
// Checked against the scene's collision boxes: the zone and the camera spot are clear.
// Move them here if the furniture on the day is different.

export const ZONE = { x: 0, y: 3.9, w: 4, d: 3 }; // centre, width (x), depth (y)

/** The judge faces the back of the room (-y); the camera sits 45° round to their left. */
export const FACING_DEG = -90; // world angle of the judge's front, 0 = +x
export const CAMERA_DIST = 3.5;
export const CAMERA_OFFSET_DEG = 45;

const rad = ((FACING_DEG + CAMERA_OFFSET_DEG) * Math.PI) / 180;
export const CAMERA = {
  x: ZONE.x + CAMERA_DIST * Math.cos(rad),
  y: ZONE.y + CAMERA_DIST * Math.sin(rad),
};
