import { useCallback, useEffect, useMemo, useRef, useState } from "react";

const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;

function fmtProb(p) {
  if (p >= 0.1) return pct(p, 1);
  if (p >= 0.001) return pct(p, 2);
  if (p >= 0.00001) return pct(p, 4);
  return "< 0.001%";
}

// log-scale position 1e-5 .. 1 -> 0 .. 100 (the threshold 0.005 sits near the middle)
const logPos = (p) => Math.min(100, Math.max(0, ((Math.log10(Math.max(p, 1e-5)) + 5) / 5) * 100));

/* ---------------- icons ---------------- */
const Icon = {
  upload: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 16V4m0 0l-4 4m4-4l4 4" /><path d="M4 16v2a2 2 0 002 2h12a2 2 0 002-2v-2" />
    </svg>
  ),
  bone: (
    <svg viewBox="0 0 64 64" fill="none" strokeWidth="4" strokeLinejoin="round">
      <defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stopColor="#22d3ee" /><stop offset="1" stopColor="#818cf8" /></linearGradient></defs>
      <path stroke="url(#lg)" d="M20 14a6 6 0 0 1 9 5l-1 7 4 4 7-1a6 6 0 1 1 5 9l-1 7a6 6 0 1 1-9-5l1-7-4-4-7 1a6 6 0 1 1-5-9z" />
      <path stroke="#f43f5e" strokeWidth="3" strokeLinecap="round" d="M30 27l5 4-3 3 4 3" />
    </svg>
  ),
  alert: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M10.3 3.9L1.8 18a2 2 0 001.7 3h17a2 2 0 001.7-3L13.7 3.9a2 2 0 00-3.4 0z" /><path d="M12 9v4M12 17h.01" />
    </svg>
  ),
  check: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="9" /><path d="M8 12.5l2.7 2.7L16 9.8" />
    </svg>
  ),
  info: (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
      <circle cx="12" cy="12" r="9" /><path d="M12 11v5M12 8h.01" />
    </svg>
  ),
};

/* ---------------- header ---------------- */
function Header({ info }) {
  return (
    <header className="header">
      <div className="brand">
        <div className="logo">{Icon.bone}</div>
        <div>
          <h1>FracAtlas <span className="grad">AI</span></h1>
          <p>Bone fracture detection and localization in X-rays</p>
        </div>
      </div>
      <div className="pills">
        {info && (
          <>
            <span className="pill">Classifier · {info.classifier.model}</span>
            <span className="pill">Detector · {info.detector.model}</span>
            <span className={`pill ${info.device.startsWith("GPU") ? "pill-ok" : ""}`}>
              <span className="dot" /> {info.device}
            </span>
          </>
        )}
      </div>
    </header>
  );
}

/* ---------------- upload ---------------- */
function Dropzone({ onFile, busy }) {
  const [drag, setDrag] = useState(false);
  const input = useRef(null);
  const pick = (f) => f && onFile(f);
  return (
    <div
      className={`dropzone ${drag ? "drag" : ""} ${busy ? "busy" : ""}`}
      onClick={() => !busy && input.current.click()}
      onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => { e.preventDefault(); setDrag(false); pick(e.dataTransfer.files[0]); }}
    >
      <input ref={input} type="file" accept="image/*" hidden onChange={(e) => { pick(e.target.files[0]); e.target.value = ""; }} />
      <div className="dz-icon">{Icon.upload}</div>
      <div className="dz-title">Drop an X-ray here</div>
      <div className="dz-sub">or <span className="link">browse</span> · JPG or PNG</div>
    </div>
  );
}

function Examples({ items, onPick, busy, active }) {
  if (!items.length) return null;
  return (
    <div className="card">
      <div className="card-title">Try a test image</div>
      <div className="examples">
        {items.map((e) => (
          <button key={e.name} className={`ex ${active === e.name ? "ex-active" : ""}`} disabled={busy} onClick={() => onPick(e)}>
            <img src={e.url} alt={e.name} loading="lazy" />
            <span className={`ex-tag ${e.true_label === "fractured" ? "t-red" : "t-green"}`}>{e.true_label}</span>
          </button>
        ))}
      </div>
      <p className="muted small">Images from the held-out test set, with their true label. Prediction only, nothing is tuned.</p>
    </div>
  );
}

/* ---------------- viewer ---------------- */
function Viewer({ src, result, boxes, showBoxes, busy }) {
  if (!src) {
    return (
      <div className="viewer empty">
        <div className="empty-art">{Icon.bone}</div>
        <p>Upload an X-ray or pick a test image to see the analysis.</p>
      </div>
    );
  }
  const w = result?.width, h = result?.height;
  const sw = w ? Math.max(2, w / 220) : 2;
  return (
    <div className="viewer">
      <div className="img-wrap">
        <img src={src} alt="X-ray" />
        {result && showBoxes && (
          <svg className="overlay" viewBox={`0 0 ${w} ${h}`} preserveAspectRatio="none">
            {boxes.map((b, i) => (
              <g key={i} className="box">
                <rect x={b.x1} y={b.y1} width={b.x2 - b.x1} height={b.y2 - b.y1} rx={sw * 2} strokeWidth={sw} />
                <rect x={b.x1} y={Math.max(0, b.y1 - sw * 11)} width={sw * 30} height={sw * 10} rx={sw * 2} className="tag-bg" />
                <text x={b.x1 + sw * 3} y={Math.max(0, b.y1 - sw * 11) + sw * 7.4} fontSize={sw * 7}>
                  {(b.conf * 100).toFixed(0)}%
                </text>
              </g>
            ))}
          </svg>
        )}
        {busy && <div className="scan"><div className="scan-line" /></div>}
      </div>
    </div>
  );
}

/* ---------------- result ---------------- */
function Verdict({ result, boxes, conf }) {
  const fr = result.fractured;
  const p = result.probability;
  let note = null;
  if (fr && boxes.length === 0)
    note = "The classifier flags a fracture, but the detector found no region above this box confidence. Try lowering the slider.";
  if (!fr && boxes.length > 0)
    note = "The classifier rates this image normal, but the detector marked a suspicious region. Worth a closer look.";
  return (
    <div className={`card verdict ${fr ? "v-red" : "v-green"}`}>
      <div className="v-head">
        <div className="v-icon">{fr ? Icon.alert : Icon.check}</div>
        <div>
          <div className="v-label">{fr ? "Fracture detected" : "No fracture detected"}</div>
          <div className="muted small">{result.filename} · {result.width}×{result.height}px · {result.ms} ms</div>
        </div>
      </div>

      <div className="meter">
        <div className="meter-top">
          <span>Fracture score</span>
          <b>{fmtProb(p)}</b>
        </div>
        <div className="bar">
          <div className={`fill ${fr ? "f-red" : "f-green"}`} style={{ width: `${logPos(p)}%` }} />
          <div className="thr" style={{ left: `${logPos(result.threshold)}%` }}>
            <span>threshold {fmtProb(result.threshold)}</span>
          </div>
        </div>
        <div className="scale"><span>0.001%</span><span>0.1%</span><span>10%</span><span>100%</span></div>
        <p className="muted small">
          Log scale. The threshold was chosen on the validation set so the model finds about 80% of fractures; anything
          above it is flagged.
        </p>
      </div>

      <div className="boxes-list">
        <div className="card-title">Fracture regions ({boxes.length} at ≥ {Math.round(conf * 100)}% confidence)</div>
        {boxes.length === 0 ? (
          <p className="muted small">No regions above the current box confidence.</p>
        ) : (
          boxes.map((b, i) => (
            <div className="box-row" key={i}>
              <span className="box-n">#{i + 1}</span>
              <div className="box-bar"><div style={{ width: `${b.conf * 100}%` }} /></div>
              <span className="box-c">{(b.conf * 100).toFixed(1)}%</span>
            </div>
          ))
        )}
      </div>
      {note && <div className="note">{Icon.info}<span>{note}</span></div>}
    </div>
  );
}

function Controls({ conf, setConf, showBoxes, setShowBoxes, disabled }) {
  return (
    <div className="controls">
      <label className="slider">
        <span>Box confidence <b>{Math.round(conf * 100)}%</b></span>
        <input type="range" min="0.05" max="0.9" step="0.05" value={conf} disabled={disabled}
          onChange={(e) => setConf(parseFloat(e.target.value))} />
      </label>
      <label className="toggle">
        <input type="checkbox" checked={showBoxes} onChange={(e) => setShowBoxes(e.target.checked)} />
        <span className="track"><span className="knob" /></span>
        Show boxes
      </label>
    </div>
  );
}

/* ---------------- performance ---------------- */
function Performance({ info }) {
  const c = info?.test_metrics?.classification;
  const d = info?.test_metrics?.detection;
  if (!c) return null;
  const tiles = [
    ["Accuracy", pct(c.accuracy), "all test images"],
    ["Recall", pct(c.recall), "fractures found"],
    ["Precision", pct(c.precision), "flags that were right"],
    ["F1 score", c.f1.toFixed(3), "recall/precision balance"],
  ];
  return (
    <section className="perf">
      <div className="section-head">
        <h2>Model performance</h2>
        <p className="muted">Held-out test set (345 images: 63 fractured, 282 normal), evaluated once with the threshold fixed on validation.</p>
      </div>
      <div className="perf-grid">
        <div className="card">
          <div className="card-title">Classification · {info.classifier.model}</div>
          <div className="tiles">
            {tiles.map(([k, v, s]) => (
              <div className="tile" key={k}><div className="tile-v grad">{v}</div><div className="tile-k">{k}</div><div className="tile-s">{s}</div></div>
            ))}
          </div>
        </div>
        <div className="card">
          <div className="card-title">Confusion matrix (test)</div>
          <div className="cm">
            <span /><span className="cm-h">pred fractured</span><span className="cm-h">pred normal</span>
            <span className="cm-r">true fractured</span><span className="cm-c good">{c.tp}</span><span className="cm-c bad">{c.fn}</span>
            <span className="cm-r">true normal</span><span className="cm-c bad">{c.fp}</span><span className="cm-c good">{c.tn}</span>
          </div>
        </div>
        {d && (
          <div className="card">
            <div className="card-title">Localization · {info.detector.model}</div>
            <div className="tiles tiles-3">
              <div className="tile"><div className="tile-v grad">{d.map50.toFixed(3)}</div><div className="tile-k">mAP50</div></div>
              <div className="tile"><div className="tile-v grad">{d.precision.toFixed(2)}</div><div className="tile-k">Precision</div></div>
              <div className="tile"><div className="tile-v grad">{d.recall.toFixed(2)}</div><div className="tile-k">Recall</div></div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}

/* ---------------- app ---------------- */
export default function App() {
  const [info, setInfo] = useState(null);
  const [examples, setExamples] = useState([]);
  const [src, setSrc] = useState(null);
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [conf, setConf] = useState(0.25);
  const [showBoxes, setShowBoxes] = useState(true);
  const [active, setActive] = useState(null);

  useEffect(() => {
    fetch("/api/info").then((r) => r.json()).then((i) => { setInfo(i); setConf(i.detector.default_conf); })
      .catch(() => setError("Cannot reach the backend. Start it with: .venv\\Scripts\\python.exe ui_server.py"));
    fetch("/api/examples").then((r) => r.json()).then(setExamples).catch(() => {});
  }, []);

  const analyse = useCallback(async (file, name) => {
    setError(null); setBusy(true); setResult(null); setActive(name || null);
    setSrc((old) => { if (old?.startsWith("blob:")) URL.revokeObjectURL(old); return URL.createObjectURL(file); });
    try {
      const fd = new FormData();
      fd.append("file", file, file.name || name || "image.jpg");
      const r = await fetch("/api/predict", { method: "POST", body: fd });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || "Prediction failed");
      setResult(j);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }, []);

  const pickExample = async (e) => {
    const blob = await (await fetch(e.url)).blob();
    analyse(new File([blob], e.name, { type: "image/jpeg" }), e.name);
  };

  const boxes = useMemo(() => (result ? result.boxes.filter((b) => b.conf >= conf - 1e-9) : []), [result, conf]);

  return (
    <div className="app">
      <div className="bg-glow" />
      <Header info={info} />

      <main className="main">
        <aside className="side">
          <Dropzone onFile={(f) => analyse(f)} busy={busy} />
          <Examples items={examples} onPick={pickExample} busy={busy} active={active} />
        </aside>

        <section className="center">
          <Viewer src={src} result={result} boxes={boxes} showBoxes={showBoxes} busy={busy} />
          <Controls conf={conf} setConf={setConf} showBoxes={showBoxes} setShowBoxes={setShowBoxes} disabled={!result} />
        </section>

        <aside className="side">
          {error && <div className="card error">{Icon.alert}<span>{error}</span></div>}
          {busy && <div className="card skeleton"><div /><div /><div /></div>}
          {result && !busy && <Verdict result={result} boxes={boxes} conf={conf} />}
          {!result && !busy && !error && (
            <div className="card hint">
              <div className="card-title">How it works</div>
              <ol>
                <li><b>Classifier</b> (YOLOv8m-cls) decides whether the X-ray shows a fracture.</li>
                <li><b>Detector</b> (YOLOv8n) draws boxes around likely fracture regions.</li>
                <li>Use the slider to show more or fewer boxes.</li>
              </ol>
            </div>
          )}
        </aside>
      </main>

      <Performance info={info} />

      <footer className="footer">
        {Icon.info}
        <span>Research demo trained on the FracAtlas dataset. Not a medical device and not a diagnosis. The model misses
          about 1 in 6 fractures and flags about 1 in 8 normal images.</span>
      </footer>
    </div>
  );
}
