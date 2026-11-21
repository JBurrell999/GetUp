# Gemini model combinations (stage 1, no facts, in the room)

Wall clock for all 12 runs in parallel: 1005 s. Audit = mean of Hydrogen and Carbon scoring the final
side-by-side with one fixed prompt (same for every config).

| config | mean audit | mean sustained collisions | mean time per video |
|---|---|---|---|
| H | 7.06 | 2.8 | 218 s |
| C>H | 5.00 | 5.0 | 319 s |
| H+C | 7.50 | 0.8 | 584 s |

| config | clip | audit | own judge score | sustained coll. | grasp err | time | audit: main issue |
|---|---|---|---|---|---|---|---|
| H | bottle | **7.25** (H 7.5, C 7.0) | 9.0 | 0 | 1.2 cm | 99 s | Robot uses the opposite (right) hand instead of the left hand to pick up and drink from th |
| H | bottle | **8.75** (H 9.5, C 8.0) | 8.0 | 0 | 5.8 cm | 136 s | Minor differences in head orientation and grip |
| H | chair | **5.75** (H 8.0, C 3.5) | 5.5 | 9 | 14.0 cm | 350 s | Robot raises its arms too high and lifts the chair unnaturally during the initial grab (00 |
| H | chair | **6.50** (H 7.5, C 5.5) | 7.0 | 2 | 0.3 cm | 286 s | Around 2.1s, the robot lifts the chair much higher than the human does, and at the end its |
| C>H | bottle | **5.50** (H 7.5, C 3.5) | 6.5 | 7 | 17.4 cm | 342 s | The robot does not actually grasp the bottle (it floats near the hand) and its body postur |
| C>H | bottle | **7.50** (H 8.5, C 6.5) | 6.5 | 4 | 15.6 cm | 312 s | The bottle floats slightly detached from the robot's hand while being lifted and held near |
| C>H | chair | **3.75** (H 4.5, C 3.0) | 6.5 | 3 | 3.2 cm | 271 s | The chair flies onto the table and away from the robot's hands, so the robot mimes holding |
| C>H | chair | **3.25** (H 5.5, C 1.0) | 5.0 | 6 | 3.0 cm | 352 s | The robot's posture and timing match the human well, but the chair disappears at around 2. |
| H+C | bottle | **7.75** (H 8.5, C 7.0) | 6.0 | 3 | 8.7 cm | 442 s | Robot briefly raises both arms at the start (0.3s) before resting, and has slightly stiffe |
| H+C | bottle | **4.75** (H 4.5, C 5.0) | 5.75 | 0 | 31.8 cm | 285 s | The robot uses the opposite (right) hand instead of the left hand, and the bottle floats r |
| H+C | chair | **9.25** (H 9.5, C 9.0) | 8.75 | 0 | 0.1 cm | 1000 s | None; the robot faithfully reproduces the human's posture, timing, and chair manipulation. |
| H+C | chair | **8.25** (H 9.0, C 7.5) | 8.4 | 0 | 0.2 cm | 609 s | Minor arm and chair glitch around 1.8s when initially grabbing the chair |