# Stage 1 benchmark (Gemini mapping, kinematic replay)

Raw phone videos (`.MOV`, auto-converted), both phases from scratch, judge = gemini-4-hydrogen, 6 runs in parallel, wall clock 263 s.

| run | facts | env score | motion scores by round | best | collisions all / sustained | grasp err | place err | judge s/round | total |
|---|---|---|---|---|---|---|---|---|---|
| bench_bottle_facts_r1 | yes | 7.0 (3 rd) | 6.5 → 6.5 → 7.5 → 8.5 → 8.0 → 7.5 | **8.5** | 1 / 0 | 6.6 cm | 2.4 cm | 9 s | 171 s |
| bench_bottle_facts_r2 | yes | 6.5 (3 rd) | 4.5 → 6.5 → 6.5 → 7.0 → 6.5 → 7.5 | **7.0** | 5 / 4 | 14.6 cm | 0.2 cm | 13 s | 207 s |
| bench_bottle_nofacts | no | 7.5 (3 rd) | 6.5 → 6.5 → 7.5 → 7.5 → 8.0 → 7.5 → 8.5 → 8.5 → 8.5 → 8.5 | **8.5** | 1 / 0 | 1.6 cm | 0.2 cm | 10 s | 259 s |
| bench_chair_facts_r1 | yes | 8.0 (1 rd) | 5.5 → 6.5 → 7.5 → 6.5 → 6.0 | **7.5** | 14 / 11 | 0.3 cm | 12.5 cm | 14 s | 186 s |
| bench_chair_facts_r2 | yes | 8.5 (2 rd) | 6.5 → 5.5 → 5.5 | **6.5** | 31 / 9 | 5.5 cm | 0.1 cm | 14 s | 166 s |
| bench_chair_nofacts | no | 8.5 (2 rd) | 7.5 → 7.5 → 8.5 → 7.5 → 6.5 | **8.5** | 5 / 1 | 0.3 cm | 0.2 cm | 15 s | 201 s |

*sustained* = longer than 0.1 s and deeper than 2 cm. Grasp / place error = worst hand / object distance from target.
