import { useState, useRef, useEffect, useCallback } from "react";

const API_URL = "/api";

const SUSPICIOUS = ["SHOCKING","WAKE UP","SHEEPLE","EXPOSED","BOMBSHELL","DEEP STATE",
  "WHISTLEBLOWER","DELETED","CONTROLLED","SECRET","HIDDEN","BANNED","CENSORED"];

function markSuspicious(text) {
  let r = text;
  SUSPICIOUS.forEach(w => {
    r = r.replace(new RegExp(`(${w.replace(/[.*+?^${}()|[\]\\]/g,"\\$&")})`, "gi"),
      "<mark>$1</mark>");
  });
  return r;
}

const EXAMPLES = [
  { tag: "Credible", title: "Federal Reserve raises interest rates by 0.25 percent",
    text: "WASHINGTON (Reuters) - The Federal Reserve raised its benchmark interest rate by a quarter of a percentage point on Wednesday. Fed Chair Jerome Powell said the committee remains committed to returning inflation to its 2 percent target. The decision was unanimous among voting members of the Federal Open Market Committee." },
  { tag: "Fake", title: "SHOCKING!!! Deep State EXPOSED: Government puts CHIPS in vaccines!!!",
    text: "WAKE UP SHEEPLE!!! A whistleblower has come forward with BOMBSHELL evidence that the deep state globalists have been secretly injecting microchips into vaccines since 2020!!! The mainstream media REFUSES to cover this because they are CONTROLLED!!! Share EVERYWHERE before it gets DELETED!!!" },
];

function AnimatedNumber({ value, decimals = 1, suffix = "%" }) {
  const [display, setDisplay] = useState(0);
  useEffect(() => {
    const end = value;
    const duration = 1000;
    const startTime = performance.now();
    const tick = (now) => {
      const p = Math.min((now - startTime) / duration, 1);
      const ease = 1 - Math.pow(1 - p, 4);
      setDisplay(end * ease);
      if (p < 1) requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);
  }, [value]);
  return <>{display.toFixed(decimals)}{suffix}</>;
}

function LimeChart({ features }) {
  const filtered = features.filter(f =>
    f.word !== "NUM" &&
    !f.word.match(/^\d+$/) &&
    Math.abs(f.weight) > 0.001
  );
  if (!filtered || filtered.length === 0) return (
    <p style={{ fontFamily: "'Figtree', sans-serif", fontSize: "0.75rem", color: "#bbb", fontStyle: "italic" }}>
      Model is very confident — individual words have minimal impact on this prediction.
    </p>
  );
  const max = Math.max(...filtered.map(f => Math.abs(f.weight)));
  return (
    <div>
      {filtered.map((f, i) => {
        const pct = Math.abs(f.weight) / max * 100;
        const positive = f.weight > 0;
        const color = positive ? "#16a34a" : "#dc2626";
        return (
          <div key={i} style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "7px" }}>
            <span style={{ fontFamily: "'Figtree', sans-serif", fontSize: "0.75rem",
              color: "#555", width: "110px", textAlign: "right",
              flexShrink: 0, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {f.word}
            </span>
            <div style={{ flex: 1, height: "14px", background: "#f1f5f9", borderRadius: "3px", overflow: "hidden", position: "relative" }}>
              <div style={{ position: "absolute",
                left: positive ? "0" : "auto", right: positive ? "auto" : "0",
                width: `${pct}%`, height: "100%", background: color,
                borderRadius: "3px", transition: "width 0.8s ease", opacity: 0.85 }}/>
            </div>
            <span style={{ fontFamily: "'Figtree', sans-serif", fontSize: "0.68rem",
              color, fontWeight: 600, width: "48px", flexShrink: 0 }}>
              {f.weight > 0 ? "+" : ""}{f.weight.toFixed(3)}
            </span>
          </div>
        );
      })}
      <div style={{ display: "flex", gap: "16px", marginTop: "10px",
        fontFamily: "'Figtree', sans-serif", fontSize: "0.68rem", color: "#999" }}>
        <span style={{ display: "flex", alignItems: "center", gap: "4px" }}>
          <span style={{ width: "10px", height: "10px", borderRadius: "2px", background: "#16a34a", display: "inline-block" }}/>
          → REAL
        </span>
        <span style={{ display: "flex", alignItems: "center", gap: "4px" }}>
          <span style={{ width: "10px", height: "10px", borderRadius: "2px", background: "#dc2626", display: "inline-block" }}/>
          → FAKE
        </span>
      </div>
    </div>
  );
}

export default function App() {
  const [title, setTitle] = useState("");
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingExplain, setLoadingExplain] = useState(false);
  const [error, setError] = useState("");
  const [history, setHistory] = useState([]);
  const [focused, setFocused] = useState(null);
  const [apiStatus, setApiStatus] = useState(null);
  const resultRef = useRef(null);

  const wc = text.trim() ? text.trim().split(/\s+/).length : 0;
  const excl = (text.match(/!/g) || []).length;
  const caps = text.length ? ((text.match(/[A-Z]/g) || []).length / text.length * 100) : 0;
  const suspCount = SUSPICIOUS.filter(w => text.toUpperCase().includes(w)).length;
  const isReal = result?.label === "REAL";
  const canSubmit = title.trim().length > 0 && wc >= 20 && !loading;

  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then(r => r.ok ? setApiStatus(true) : setApiStatus(false))
      .catch(() => setApiStatus(false));
  }, []);

  useEffect(() => { if (result) { setResult(null); setExplanation(null); } }, [title, text]);

  const submit = useCallback(async () => {
    setError(""); setResult(null); setExplanation(null);
    if (!title.trim()) { setError("Please enter a headline."); return; }
    if (wc < 20) { setError(`Article too short — ${wc} of 20 words minimum.`); return; }
    setLoading(true);
    try {
      const r = await fetch(`${API_URL}/predict`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, text })
      });
      if (!r.ok) throw new Error((await r.json()).detail || "Prediction failed");
      const d = await r.json();
      setResult(d);
      setHistory(prev => [{ id: Date.now(), title, label: d.label,
        confidence: d.confidence, ts: new Date().toLocaleTimeString() }, ...prev.slice(0, 9)]);
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" }), 100);

      setLoadingExplain(true);
      fetch(`${API_URL}/explain`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title, text })
      }).then(r => r.json()).then(d => {
        setExplanation(d.explanation);
        setLoadingExplain(false);
      }).catch(() => setLoadingExplain(false));

    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  }, [title, text, wc]);

  useEffect(() => {
    const handler = (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter" && canSubmit) submit();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [canSubmit, submit]);

  const reset = () => { setTitle(""); setText(""); setResult(null); setExplanation(null); setError(""); };

  const C = {
    navy: "#1a3a5c", green: "#1a5c3a", greenLight: "#f4fbf7",
    red: "#8b0000", redLight: "#fff8f8", amber: "#b8860b",
    ink: "#0a0a0a", mid: "#555", muted: "#999",
    border: "#e0e0e0", bg: "#f7f6f3",
  };

  const inputLine = (name) => ({
    width: "100%", background: "transparent", border: "none",
    borderBottom: `1.5px solid ${focused === name ? C.navy : C.border}`,
    color: C.ink, fontFamily: "'Fraunces', Georgia, serif",
    fontSize: name === "title" ? "1.05rem" : "0.92rem",
    padding: "10px 0", outline: "none", transition: "border-color 0.25s",
    lineHeight: 1.75, resize: name === "text" ? "vertical" : undefined,
  });

  return (
    <div style={{ minHeight: "100vh", background: C.bg, fontFamily: "'Fraunces', Georgia, serif", color: C.ink }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,600;1,9..144,300;1,9..144,400;1,9..144,600&family=Figtree:wght@300;400;500;600&display=swap');
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #f7f6f3; }
        ::selection { background: #d6e4f0; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-thumb { background: #ccc; border-radius: 2px; }
        ::placeholder { color: #bbb; font-style: italic; font-family: 'Fraunces', serif; }
        mark { background: #fff176; color: #7a5c00; border-radius: 2px; padding: 0 2px; font-style: normal; font-weight: 600; }
        @keyframes fadeUp { from { opacity:0; transform:translateY(16px); } to { opacity:1; transform:none; } }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes growBar { from { width: 0%; } }
        @keyframes reveal { from { opacity:0; transform:translateX(10px); } to { opacity:1; transform:none; } }
        @keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0.25; } }
        @keyframes pulse { 0%,100% { opacity:1; } 50% { opacity:0.4; } }
        .ex-btn { background: white; border: 1.5px solid #e5e2dc; color: #888; padding: 7px 16px; border-radius: 20px; font-family: 'Figtree', sans-serif; font-size: 0.72rem; font-weight: 500; cursor: pointer; transition: all 0.18s; letter-spacing: 0.03em; }
        .ex-btn:hover { border-color: #1a3a5c; color: #1a3a5c; background: #f0f4f8; }
        .submit-btn { flex: 1; padding: 16px; background: #1a3a5c; color: #fff; border: none; font-family: 'Figtree', sans-serif; font-size: 0.8rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase; cursor: pointer; transition: all 0.2s; border-radius: 2px; }
        .submit-btn:hover:not(:disabled) { background: #0a2540; transform: translateY(-1px); box-shadow: 0 6px 20px rgba(26,58,92,0.18); }
        .submit-btn:disabled { background: #e0ddd8; color: #bbb; cursor: not-allowed; }
        .reset-btn { padding: 16px 22px; background: transparent; border: 1.5px solid #e0e0e0; color: #999; font-family: 'Figtree', sans-serif; font-size: 0.78rem; font-weight: 500; cursor: pointer; transition: all 0.2s; border-radius: 2px; letter-spacing: 0.06em; text-transform: uppercase; }
        .reset-btn:hover { border-color: #1a3a5c; color: #1a3a5c; }
        .label-tag { font-family: 'Figtree', sans-serif; font-size: 0.62rem; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; color: #999; margin-bottom: 8px; display: block; }
        .stat-row { display: flex; justify-content: space-between; align-items: center; padding: 11px 0; border-bottom: 1px solid #ece9e3; }
        .stat-row:last-child { border-bottom: none; }
        .hist-row { transition: background 0.15s; border-radius: 4px; }
        .hist-row:hover { background: #ece9e3; }
        @media (max-width: 900px) { .main-grid { grid-template-columns: 1fr !important; gap: 48px !important; } .result-col { position: static !important; } .page-pad { padding-left: 24px !important; padding-right: 24px !important; } }
      `}</style>

      <div style={{ height: "3px", background: `linear-gradient(90deg, ${C.navy}, ${C.green})` }} />

      {/* HEADER */}
      <header style={{ background: "white", borderBottom: `1px solid ${C.border}`, position: "sticky", top: 0, zIndex: 50, boxShadow: "0 1px 12px rgba(0,0,0,0.05)" }}>
        <div className="page-pad" style={{ maxWidth: "1160px", margin: "0 auto", padding: "0 56px", display: "flex", alignItems: "center", justifyContent: "space-between", height: "66px" }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: "3px" }}>
            <span style={{ fontSize: "1.55rem", fontWeight: 600, letterSpacing: "-0.5px", color: C.ink }}>Truth</span>
            <span style={{ fontSize: "1.55rem", fontWeight: 300, fontStyle: "italic", letterSpacing: "-0.5px", color: C.navy }}>Lens</span>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: "20px" }}>
            {history.length > 0 && (
              <span style={{ fontFamily: "'Figtree', sans-serif", fontSize: "0.72rem", color: "#aaa" }}>
                {history.length} verified
              </span>
            )}
            <div style={{ display: "flex", alignItems: "center", gap: "7px" }}>
              <div style={{ width: "7px", height: "7px", borderRadius: "50%",
                background: apiStatus === null ? "#ccc" : apiStatus ? C.green : C.red,
                animation: apiStatus === null ? "pulse 1.2s ease infinite" : "none" }} />
              <span style={{ fontFamily: "'Figtree', sans-serif", fontSize: "0.7rem", color: "#bbb" }}>
                {apiStatus === null ? "Connecting…" : apiStatus ? "API online" : "API offline"}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* MAIN */}
      <main style={{ maxWidth: "1160px", margin: "0 auto" }}>
        <div className="main-grid page-pad" style={{ display: "grid", gridTemplateColumns: "1.1fr 0.9fr", gap: "72px", alignItems: "start", padding: "56px 56px 100px" }}>

          {/* LEFT */}
          <div>
            <div style={{ display: "flex", gap: "8px", marginBottom: "40px", flexWrap: "wrap", alignItems: "center" }}>
              <span style={{ fontFamily: "'Figtree', sans-serif", fontSize: "0.7rem", color: "#bbb", marginRight: "4px" }}>Try:</span>
              {EXAMPLES.map(ex => (
                <button key={ex.tag} className="ex-btn"
                  onClick={() => { setTitle(ex.title); setText(ex.text); setResult(null); setExplanation(null); setError(""); }}>
                  {ex.tag}
                </button>
              ))}
            </div>

            <div style={{ display: "flex", flexDirection: "column", gap: "32px" }}>
              <div>
                <span className="label-tag">Headline</span>
                <input value={title} onChange={e => setTitle(e.target.value)}
                  onFocus={() => setFocused("title")} onBlur={() => setFocused(null)}
                  placeholder="Enter the news headline..."
                  style={inputLine("title")} />
              </div>

              <div>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
                  <span className="label-tag" style={{ marginBottom: 0 }}>Article body</span>
                  <span style={{ fontFamily: "'Figtree', sans-serif", fontSize: "0.68rem", fontWeight: 500,
                    color: wc >= 20 ? C.green : wc > 0 ? C.amber : "#ccc" }}>
                    {wc > 0 ? `${wc} words${wc < 20 ? ` (min 20)` : " ✓"}` : "min 20 words"}
                  </span>
                </div>
                <textarea value={text} onChange={e => setText(e.target.value)}
                  onFocus={() => setFocused("text")} onBlur={() => setFocused(null)}
                  placeholder="Paste the full article text here..." rows={9}
                  style={inputLine("text")} />
              </div>

              {text.length > 0 && (
                <div style={{ display: "flex", gap: "7px", flexWrap: "wrap" }}>
                  {[
                    { label: `${excl} exclamation${excl !== 1 ? "s" : ""}`, warn: excl > 3 },
                    { label: `${caps.toFixed(0)}% caps`, warn: caps > 15 },
                    { label: `${suspCount} suspicious word${suspCount !== 1 ? "s" : ""}`, warn: suspCount > 0 },
                  ].map(s => (
                    <span key={s.label} style={{ fontFamily: "'Figtree', sans-serif", fontSize: "0.7rem", fontWeight: 500,
                      padding: "4px 12px", borderRadius: "20px",
                      background: s.warn ? "#fffbea" : "#f0ede8",
                      border: `1px solid ${s.warn ? "#e8d080" : "#e5e2dc"}`,
                      color: s.warn ? C.amber : C.muted }}>
                      {s.warn ? "⚠ " : ""}{s.label}
                    </span>
                  ))}
                </div>
              )}

              {error && (
                <div style={{ fontFamily: "'Figtree', sans-serif", fontSize: "0.8rem", color: C.red,
                  borderLeft: `2px solid ${C.red}`, paddingLeft: "14px", lineHeight: 1.6 }}>
                  {error}
                </div>
              )}

              <div style={{ display: "flex", gap: "10px" }}>
                <button className="submit-btn" onClick={submit} disabled={!canSubmit}>
                  {loading
                    ? <span style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: "9px" }}>
                        <span style={{ width: "13px", height: "13px", border: "1.5px solid rgba(255,255,255,0.3)", borderTopColor: "#fff", borderRadius: "50%", display: "inline-block", animation: "spin 0.75s linear infinite" }} />
                        Analysing...
                      </span>
                    : "Verify article"}
                </button>
                <button className="reset-btn" onClick={reset}>Reset</button>
              </div>
            </div>

            {/* Session history */}
            {history.length > 0 && (
              <div style={{ marginTop: "56px", paddingTop: "28px", borderTop: `1px solid ${C.border}` }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "14px" }}>
                  <span className="label-tag" style={{ marginBottom: 0 }}>Session history</span>
                  <button onClick={() => setHistory([])} style={{ fontFamily: "'Figtree', sans-serif", fontSize: "0.68rem", color: "#ccc", background: "none", border: "none", cursor: "pointer" }}>Clear</button>
                </div>
                {history.map(h => (
                  <div key={h.id} className="hist-row" style={{ display: "flex", alignItems: "center", gap: "12px", padding: "10px 8px" }}>
                    <div style={{ width: "6px", height: "6px", borderRadius: "50%", flexShrink: 0, background: h.label === "REAL" ? C.green : C.red }} />
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontSize: "0.88rem", color: "#333", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{h.title}</div>
                      <div style={{ fontFamily: "'Figtree', sans-serif", fontSize: "0.65rem", color: "#bbb", marginTop: "2px" }}>
                        {h.label} · {(h.confidence * 100).toFixed(0)}% · {h.ts}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* RIGHT: RESULT */}
          <div className="result-col" ref={resultRef} style={{ position: "sticky", top: "90px" }}>

            {!result && !loading && (
              <div style={{ border: `1.5px dashed ${C.border}`, borderRadius: "4px", padding: "64px 40px",
                display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
                minHeight: "400px", gap: "14px", textAlign: "center" }}>
                <div style={{ width: "52px", height: "52px", borderRadius: "50%", border: `1.5px dashed ${C.border}`,
                  display: "flex", alignItems: "center", justifyContent: "center" }}>
                  <span style={{ fontStyle: "italic", fontSize: "1.4rem", color: "#ddd" }}>?</span>
                </div>
                <p style={{ fontFamily: "'Figtree', sans-serif", fontSize: "0.75rem", color: "#ccc", lineHeight: 1.8 }}>
                  Submit an article to see the verdict
                </p>
              </div>
            )}

            {loading && (
              <div style={{ border: `1.5px solid ${C.border}`, borderRadius: "4px", padding: "64px 40px",
                display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
                minHeight: "400px", gap: "18px" }}>
                <div style={{ width: "32px", height: "32px", border: `2px solid ${C.border}`,
                  borderTopColor: C.navy, borderRadius: "50%", animation: "spin 0.9s linear infinite" }} />
                <p style={{ fontFamily: "'Figtree', sans-serif", fontSize: "0.72rem", color: "#bbb",
                  letterSpacing: "0.1em", animation: "blink 2s ease infinite" }}>ANALYSING ARTICLE</p>
              </div>
            )}

            {result && (
              <div style={{ animation: "fadeUp 0.45s ease", background: "white", borderRadius: "4px",
                border: `1.5px solid ${C.border}`, overflow: "hidden" }}>

                {/* Verdict */}
                <div style={{ padding: "40px 36px 32px", borderBottom: `1.5px solid ${C.border}`,
                  background: isReal ? C.greenLight : C.redLight }}>
                  <span style={{ fontFamily: "'Figtree', sans-serif", fontSize: "0.62rem", fontWeight: 600,
                    letterSpacing: "0.16em", textTransform: "uppercase", color: isReal ? C.green : C.red,
                    display: "block", marginBottom: "12px" }}>Verdict</span>
                  <div style={{ fontFamily: "'Fraunces', Georgia, serif",
                    fontSize: "clamp(3.5rem, 8vw, 6rem)", fontWeight: 300, fontStyle: "italic",
                    lineHeight: 0.9, letterSpacing: "-2px", color: isReal ? C.green : C.red,
                    marginBottom: "20px", animation: "reveal 0.5s ease" }}>
                    {isReal ? "Credible." : "Suspicious."}
                  </div>
                  <div>
                    <div style={{ height: "4px", background: C.border, borderRadius: "2px", overflow: "hidden", marginBottom: "8px" }}>
                      <div style={{ height: "100%", width: `${result.real_probability * 100}%`,
                        background: isReal ? C.green : C.red, borderRadius: "2px",
                        animation: "growBar 1.2s cubic-bezier(0.16,1,0.3,1) forwards" }} />
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", fontFamily: "'Figtree', sans-serif", fontSize: "0.7rem" }}>
                      <span style={{ color: "#bbb" }}>Credibility score</span>
                      <span style={{ fontWeight: 600, color: isReal ? C.green : C.red }}>
                        <AnimatedNumber value={result.real_probability * 100} />
                      </span>
                    </div>
                  </div>
                </div>

                {/* Article signals */}
                <div style={{ padding: "8px 36px 16px" }}>
                  <span className="label-tag" style={{ marginTop: "12px", display: "block" }}>Article signals</span>
                  {[
                    { label: "Word count", value: wc, accent: C.mid },
                    { label: "Exclamation marks", value: excl, accent: excl > 3 ? C.amber : C.mid },
                    { label: "Caps ratio", value: `${caps.toFixed(1)}%`, accent: caps > 15 ? C.amber : C.mid },
                    { label: "Suspicious keywords", value: suspCount, accent: suspCount > 0 ? C.amber : C.mid },
                  ].map(s => (
                    <div key={s.label} className="stat-row">
                      <span style={{ fontFamily: "'Figtree', sans-serif", fontSize: "0.78rem", color: C.muted }}>{s.label}</span>
                      <span style={{ fontFamily: "'Figtree', sans-serif", fontSize: "0.82rem", fontWeight: 600, color: s.accent }}>{s.value}</span>
                    </div>
                  ))}
                </div>

                {/* LIME */}
                <div style={{ margin: "0 36px 28px", padding: "18px 20px", background: "#f7f6f3",
                  border: `1px solid ${C.border}`, borderRadius: "2px" }}>
                  <span style={{ fontFamily: "'Figtree', sans-serif", fontSize: "0.62rem", fontWeight: 600,
                    letterSpacing: "0.14em", textTransform: "uppercase", color: C.navy,
                    display: "block", marginBottom: "14px" }}>
                    LIME — Key words influencing prediction
                  </span>
                  {loadingExplain ? (
                    <div style={{ display: "flex", alignItems: "center", gap: "8px", color: "#bbb",
                      fontFamily: "'Figtree', sans-serif", fontSize: "0.75rem" }}>
                      <span style={{ width: "12px", height: "12px", border: `1.5px solid #ddd`,
                        borderTopColor: C.navy, borderRadius: "50%", display: "inline-block",
                        animation: "spin 0.9s linear infinite" }} />
                      Computing explanation…
                    </div>
                  ) : explanation ? (
                    <LimeChart features={explanation} isReal={isReal} />
                  ) : null}
                </div>

                {/* Flagged language */}
                {suspCount > 0 && (
                  <div style={{ margin: "0 36px 28px", padding: "18px 20px", background: "#fffbea",
                    border: `1px solid #e8d080`, borderLeft: `3px solid ${C.amber}`, borderRadius: "2px" }}>
                    <span style={{ fontFamily: "'Figtree', sans-serif", fontSize: "0.62rem", fontWeight: 600,
                      letterSpacing: "0.14em", textTransform: "uppercase", color: C.amber,
                      display: "block", marginBottom: "10px" }}>Flagged language</span>
                    <p style={{ fontSize: "0.85rem", color: "#7a5c00", lineHeight: 1.8, fontStyle: "italic", fontWeight: 300 }}
                      dangerouslySetInnerHTML={{ __html: markSuspicious(text.substring(0, 300) + (text.length > 300 ? "..." : "")) }} />
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </main>

      <div style={{ height: "3px", background: `linear-gradient(90deg, ${C.navy}, ${C.green})` }} />
      <footer style={{ borderTop: `1px solid ${C.border}`, padding: "22px 56px" }}>
        <div style={{ maxWidth: "1160px", margin: "0 auto", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <span style={{ fontSize: "1rem", fontWeight: 300, fontStyle: "italic", color: "#bbb" }}>TruthLens</span>
          <span style={{ fontFamily: "'Figtree', sans-serif", fontSize: "0.68rem", color: "#ccc", fontWeight: 300 }}>
            AI-powered news verification · LightGBM + LIME
          </span>
        </div>
      </footer>
    </div>
  );
}