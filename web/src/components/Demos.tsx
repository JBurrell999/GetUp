// Two finished runs, shown so people can see what their upload turns into: the person on the left,
// the simulated G1 copying them on the right. The clips are static files in public/demos/.

const DEMOS = [
  {
    src: "/demos/lift-a-chair.mp4",
    title: "Lifting a chair",
    note: "Both hands on the backrest, lift, set it down.",
    label: "Side by side: a person lifts a chair by its backrest with both hands, and the simulated G1 robot does the same",
  },
  {
    src: "/demos/walk.mp4",
    title: "Walking",
    note: "Walking along a counter, step for step.",
    label: "Side by side: a person walks along a counter, and the simulated G1 robot walks the same way",
  },
];

export function Demos() {
  return (
    <section className="demos" id="demos" aria-labelledby="demos-title">
      <div className="wrap">
        <div className="envs-head">
          <h2 className="envs-title" id="demos-title">
            See it work
          </h2>
          <p className="envs-lede">
            Each video goes to a simulated Unitree G1. Gemini works out what the person is doing, the robot copies it,
            and Gemini compares the two until they match. Left: the upload. Right: the robot.
          </p>
        </div>
        <ul className="demos-grid">
          {DEMOS.map((d) => (
            <li key={d.src} className="demo">
              <video src={d.src} aria-label={d.label} autoPlay muted loop playsInline controls preload="metadata" />
              <p className="demo-title">{d.title}</p>
              <p className="demo-note">{d.note}</p>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
