import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, CartesianGrid, Legend, Cell, ScatterChart, Scatter,
} from "recharts";

// ═══════════════════════════════════════════════════════════
// CONFIG
// ═══════════════════════════════════════════════════════════

const API_BASE = "http://localhost:8000/api";
const DEFAULT_LEAGUE = "1311316930342195200";

const POS = {
  QB:  { color: "#ef4444", bg: "rgba(239,68,68,0.10)", glow: "rgba(239,68,68,0.25)" },
  RB:  { color: "#22c55e", bg: "rgba(34,197,94,0.10)",  glow: "rgba(34,197,94,0.25)" },
  WR:  { color: "#3b82f6", bg: "rgba(59,130,246,0.10)", glow: "rgba(59,130,246,0.25)" },
  TE:  { color: "#eab308", bg: "rgba(234,179,8,0.10)",  glow: "rgba(234,179,8,0.25)" },
  K:   { color: "#a855f7", bg: "rgba(168,85,247,0.10)", glow: "rgba(168,85,247,0.25)" },
  DEF: { color: "#6b7280", bg: "rgba(107,114,128,0.10)",glow: "rgba(107,114,128,0.25)" },
};

const posColor = (p) => POS[p]?.color || "#6b7280";
const posBg = (p) => POS[p]?.bg || "rgba(107,114,128,0.1)";

// ═══════════════════════════════════════════════════════════
// API CLIENT
// ═══════════════════════════════════════════════════════════

async function api(path, opts = {}) {
  const url = `${API_BASE}${path}`;
  const res = await fetch(url, opts);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `API ${res.status}`);
  }
  return res.json();
}

// ═══════════════════════════════════════════════════════════
// STYLES (CSS-in-JS theme object)
// ═══════════════════════════════════════════════════════════

const T = {
  bg0: "#06080d",
  bg1: "#0c1017",
  bg2: "#121820",
  bg3: "#1a2230",
  border: "#1e293b",
  borderLight: "#2a3548",
  text: "#e2e8f0",
  textDim: "#94a3b8",
  textMuted: "#64748b",
  textFaint: "#475569",
  accent: "#6366f1",
  accentGlow: "rgba(99,102,241,0.15)",
  accentText: "#a5b4fc",
  green: "#22c55e",
  red: "#ef4444",
  yellow: "#eab308",
  font: "'IBM Plex Mono', 'Fira Code', 'SF Mono', monospace",
  fontDisplay: "'IBM Plex Sans', 'Helvetica Neue', sans-serif",
  radius: 6,
};

// ═══════════════════════════════════════════════════════════
// MAIN APP
// ═══════════════════════════════════════════════════════════

export default function App() {
  const [page, setPage] = useState("draft");
  const [leagueId, setLeagueId] = useState(DEFAULT_LEAGUE);
  const [league, setLeague] = useState(null);
  const [error, setError] = useState(null);
  const [apiOk, setApiOk] = useState(null);

  // Health check
  useEffect(() => {
    api("/health").then(() => setApiOk(true)).catch(() => setApiOk(false));
  }, []);

  // Load league on ID change
  useEffect(() => {
    if (!leagueId) return;
    api(`/league/${leagueId}`)
      .then(setLeague)
      .catch((e) => setError(e.message));
  }, [leagueId]);

  const pages = [
    { id: "draft",    label: "LIVE DRAFT",   icon: "⚡" },
    { id: "rankings", label: "RANKINGS",     icon: "📊" },
    { id: "injuries", label: "INJURIES",     icon: "🏥" },
    { id: "aging",    label: "AGING CURVES", icon: "📈" },
    { id: "scarcity", label: "SCARCITY",     icon: "⚖️" },
    { id: "compare",  label: "COMPARE",      icon: "🔍" },
  ];

  return (
    <div style={{ fontFamily: T.font, background: T.bg0, color: T.text, minHeight: "100vh" }}>
      {/* ── HEADER ── */}
      <header style={{
        background: `linear-gradient(180deg, ${T.bg1} 0%, ${T.bg0} 100%)`,
        borderBottom: `1px solid ${T.border}`,
        padding: "14px 20px 0",
      }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 10 }}>
          <div style={{ display: "flex", alignItems: "baseline", gap: 12 }}>
            <h1 style={{
              fontFamily: T.fontDisplay, fontSize: 18, fontWeight: 800,
              margin: 0, letterSpacing: "-0.3px", color: T.text,
            }}>
              <span style={{ color: T.accent }}>▌</span>DYNASTY VORP
            </h1>
            {league && (
              <span style={{ fontSize: 11, color: T.textMuted }}>
                {league.name} · {league.season} · {league.total_rosters}T
              </span>
            )}
          </div>

          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <div style={{
              width: 7, height: 7, borderRadius: "50%",
              background: apiOk === true ? T.green : apiOk === false ? T.red : T.yellow,
              boxShadow: `0 0 6px ${apiOk === true ? T.green : apiOk === false ? T.red : T.yellow}`,
            }} />
            <span style={{ fontSize: 10, color: T.textMuted }}>
              {apiOk === true ? "API CONNECTED" : apiOk === false ? "API OFFLINE" : "CHECKING..."}
            </span>
            <input
              value={leagueId}
              onChange={(e) => setLeagueId(e.target.value)}
              placeholder="League ID"
              style={{
                background: T.bg2, border: `1px solid ${T.border}`, borderRadius: T.radius,
                padding: "6px 10px", color: T.text, fontSize: 11, width: 170, fontFamily: T.font,
              }}
            />
          </div>
        </div>

        {/* Nav */}
        <nav style={{ display: "flex", gap: 1, marginTop: 12 }}>
          {pages.map((p) => (
            <button
              key={p.id}
              onClick={() => setPage(p.id)}
              style={{
                background: page === p.id ? T.accentGlow : "transparent",
                border: `1px solid ${page === p.id ? T.accent + "44" : "transparent"}`,
                borderBottom: page === p.id ? `2px solid ${T.accent}` : "2px solid transparent",
                borderRadius: `${T.radius}px ${T.radius}px 0 0`,
                padding: "8px 14px", cursor: "pointer", fontFamily: T.font,
                color: page === p.id ? T.accentText : T.textMuted,
                fontSize: 11, fontWeight: page === p.id ? 700 : 500,
                letterSpacing: "0.4px", transition: "all 0.15s",
              }}
            >
              {p.icon} {p.label}
            </button>
          ))}
        </nav>
      </header>

      {error && (
        <div style={{
          background: "rgba(239,68,68,0.08)", border: `1px solid rgba(239,68,68,0.2)`,
          margin: "12px 20px 0", borderRadius: T.radius, padding: "8px 14px",
          fontSize: 12, color: "#fca5a5",
        }}>
          {error}
        </div>
      )}

      {/* ── CONTENT ── */}
      <main style={{ padding: "16px 20px", maxWidth: 1360, margin: "0 auto" }}>
        {apiOk === false ? (
          <OfflineMessage />
        ) : page === "draft" ? (
          <DraftPage leagueId={leagueId} />
        ) : page === "rankings" ? (
          <RankingsPage leagueId={leagueId} />
        ) : page === "injuries" ? (
          <InjuriesPage />
        ) : page === "aging" ? (
          <AgingPage />
        ) : page === "scarcity" ? (
          <ScarcityPage leagueId={leagueId} />
        ) : page === "compare" ? (
          <ComparePage leagueId={leagueId} />
        ) : null}
      </main>
    </div>
  );
}

function OfflineMessage() {
  return (
    <div style={{ textAlign: "center", padding: "80px 20px" }}>
      <div style={{ fontSize: 40, marginBottom: 16 }}>🔌</div>
      <h2 style={{ fontFamily: "'IBM Plex Sans', sans-serif", fontSize: 20, fontWeight: 700, color: T.text, margin: "0 0 8px" }}>
        API Not Connected
      </h2>
      <p style={{ color: T.textMuted, fontSize: 13, maxWidth: 440, margin: "0 auto 20px", lineHeight: 1.6 }}>
        Start the FastAPI backend to power this dashboard.
      </p>
      <code style={{
        background: T.bg2, border: `1px solid ${T.border}`, borderRadius: T.radius,
        padding: "10px 16px", fontSize: 12, color: T.accentText, display: "inline-block",
      }}>
        uvicorn src.api.main:app --reload --port 8000
      </code>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════
// SHARED COMPONENTS
// ═══════════════════════════════════════════════════════════

function Card({ children, style = {} }) {
  return (
    <div style={{
      background: T.bg1, border: `1px solid ${T.border}`,
      borderRadius: T.radius, ...style,
    }}>
      {children}
    </div>
  );
}

function PosBadge({ pos }) {
  return (
    <span style={{
      background: posBg(pos), color: posColor(pos),
      padding: "2px 8px", borderRadius: 3, fontSize: 10, fontWeight: 700,
      letterSpacing: "0.5px",
    }}>
      {pos}
    </span>
  );
}

function PosFilterBar({ value, onChange }) {
  return (
    <div style={{ display: "flex", gap: 4 }}>
      {["QB", "RB", "WR", "TE"].map((p) => (
        <button key={p} onClick={() => {
          onChange(value.includes(p) ? value.filter((x) => x !== p) : [...value, p]);
        }} style={{
          background: value.includes(p) ? posColor(p) : T.bg2,
          opacity: value.includes(p) ? 1 : 0.5,
          border: `1px solid ${posColor(p)}`, borderRadius: 3,
          padding: "3px 10px", color: "#fff", fontSize: 10, fontWeight: 700,
          cursor: "pointer", fontFamily: T.font, transition: "all 0.12s",
        }}>
          {p}
        </button>
      ))}
    </div>
  );
}

function SectionTitle({ children }) {
  return (
    <h2 style={{
      fontFamily: "'IBM Plex Sans', sans-serif", fontSize: 15, fontWeight: 700,
      margin: "0 0 12px", color: T.text, letterSpacing: "-0.2px",
    }}>
      {children}
    </h2>
  );
}

function Metric({ label, value, color }) {
  return (
    <div style={{ textAlign: "center" }}>
      <div style={{ fontSize: 20, fontWeight: 800, color: color || T.text, fontFamily: T.font }}>{value}</div>
      <div style={{ fontSize: 9, color: T.textMuted, letterSpacing: "0.5px", marginTop: 2 }}>{label}</div>
    </div>
  );
}

function DataTable({ columns, rows, maxHeight = 520 }) {
  return (
    <div style={{ overflowY: "auto", maxHeight, overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
        <thead>
          <tr style={{ position: "sticky", top: 0, background: T.bg0, zIndex: 2 }}>
            {columns.map((c) => (
              <th key={c.key} style={{
                padding: "8px 10px", textAlign: c.align || "left",
                color: T.textMuted, fontWeight: 600, fontSize: 9,
                letterSpacing: "0.6px", borderBottom: `1px solid ${T.border}`,
                whiteSpace: "nowrap",
              }}>
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} style={{
              background: i % 2 === 0 ? "transparent" : T.bg0 + "88",
              borderBottom: `1px solid ${T.border}22`,
              transition: "background 0.1s",
            }}
            onMouseEnter={(e) => e.currentTarget.style.background = T.accentGlow}
            onMouseLeave={(e) => e.currentTarget.style.background = i % 2 === 0 ? "transparent" : T.bg0 + "88"}
            >
              {columns.map((c) => (
                <td key={c.key} style={{
                  padding: "7px 10px", textAlign: c.align || "left",
                  color: c.color ? c.color(row) : T.text,
                  fontWeight: c.bold ? 600 : 400,
                  whiteSpace: "nowrap",
                }}>
                  {c.render ? c.render(row) : row[c.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════
// LIVE DRAFT PAGE
// ═══════════════════════════════════════════════════════════

function DraftPage({ leagueId }) {
  const [drafts, setDrafts] = useState([]);
  const [draftId, setDraftId] = useState("");
  const [draftState, setDraftState] = useState(null);
  const [available, setAvailable] = useState(null);
  const [mySlot, setMySlot] = useState(1);
  const [posFilter, setPosFilter] = useState(["QB", "RB", "WR", "TE"]);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [refreshInterval, setRefreshInterval] = useState(15);
  const [loading, setLoading] = useState(false);
  const intervalRef = useRef(null);

  // Load drafts
  useEffect(() => {
    if (!leagueId) return;
    api(`/draft/${leagueId}/drafts`).then((d) => {
      setDrafts(d || []);
      if (d?.length) setDraftId(d[0].draft_id);
    }).catch(() => {});
  }, [leagueId]);

  // Fetch draft state + available
  const fetchDraft = useCallback(async () => {
    if (!leagueId || !draftId) return;
    setLoading(true);
    try {
      const [st, av] = await Promise.all([
        api(`/draft/${leagueId}/state/${draftId}`),
        api(`/draft/${leagueId}/available/${draftId}?my_slot=${mySlot}&top_n=50&include_injuries=true`),
      ]);
      setDraftState(st);
      setAvailable(av);
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  }, [leagueId, draftId, mySlot]);

  useEffect(() => { fetchDraft(); }, [fetchDraft]);

  // Auto-refresh
  useEffect(() => {
    if (autoRefresh && draftState?.status === "drafting") {
      intervalRef.current = setInterval(fetchDraft, refreshInterval * 1000);
      return () => clearInterval(intervalRef.current);
    }
    return () => clearInterval(intervalRef.current);
  }, [autoRefresh, refreshInterval, fetchDraft, draftState?.status]);

  if (!drafts.length) {
    return (
      <Card style={{ padding: 40, textAlign: "center" }}>
        <div style={{ fontSize: 32, marginBottom: 12 }}>🏈</div>
        <p style={{ color: T.textMuted, fontSize: 13 }}>No drafts found for this league. Enter your Sleeper League ID above.</p>
      </Card>
    );
  }

  const ds = draftState;
  const av = available;
  const filteredAvailable = av?.available?.filter((p) => posFilter.includes(p.position)) || [];
  const need = av?.positional_need || {};

  return (
    <div>
      {/* Controls bar */}
      <Card style={{ padding: "10px 14px", marginBottom: 12, display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
        <select value={draftId} onChange={(e) => setDraftId(e.target.value)} style={{
          background: T.bg2, border: `1px solid ${T.border}`, borderRadius: T.radius,
          padding: "6px 10px", color: T.text, fontSize: 11, fontFamily: T.font,
        }}>
          {drafts.map((d) => (
            <option key={d.draft_id} value={d.draft_id}>
              {d.season} — {d.status} ({d.draft_id.slice(-6)})
            </option>
          ))}
        </select>
        <label style={{ fontSize: 11, color: T.textMuted, display: "flex", alignItems: "center", gap: 4 }}>
          My Slot:
          <input type="number" min={1} max={16} value={mySlot} onChange={(e) => setMySlot(+e.target.value)} style={{
            background: T.bg2, border: `1px solid ${T.border}`, borderRadius: T.radius,
            padding: "5px 8px", color: T.text, fontSize: 11, width: 44, fontFamily: T.font, textAlign: "center",
          }} />
        </label>
        <label style={{ fontSize: 11, color: T.textMuted, display: "flex", alignItems: "center", gap: 6, cursor: "pointer" }}>
          <input type="checkbox" checked={autoRefresh} onChange={(e) => setAutoRefresh(e.target.checked)} />
          Auto-refresh ({refreshInterval}s)
        </label>
        <button onClick={fetchDraft} style={{
          background: T.accent, border: "none", borderRadius: T.radius,
          padding: "6px 14px", color: "#fff", fontSize: 11, fontWeight: 700,
          cursor: "pointer", fontFamily: T.font, marginLeft: "auto",
        }}>
          {loading ? "⏳" : "🔄"} REFRESH
        </button>
      </Card>

      {/* Status bar */}
      {ds && (
        <Card style={{ padding: "12px 14px", marginBottom: 12, display: "flex", justifyContent: "space-around", flexWrap: "wrap", gap: 8 }}>
          <Metric label="STATUS" value={ds.status?.toUpperCase()} color={ds.status === "drafting" ? T.green : T.textMuted} />
          <Metric label="ROUND" value={`${ds.current_round}/${ds.total_rounds}`} />
          <Metric label="PICK" value={`${ds.current_pick_no}`} color={T.accentText} />
          <Metric label="MADE" value={`${ds.picks_made}/${ds.total_picks}`} />
        </Card>
      )}

      {/* Main content: Available + My Roster side by side */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 320px", gap: 12 }}>
        {/* Left: Best Available */}
        <Card style={{ overflow: "hidden" }}>
          <div style={{ padding: "10px 14px", borderBottom: `1px solid ${T.border}`, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <SectionTitle>⚡ Best Available</SectionTitle>
            <PosFilterBar value={posFilter} onChange={setPosFilter} />
          </div>
          <DataTable
            maxHeight={480}
            columns={[
              { key: "_rank", label: "#", align: "center", render: (_, i) => i + 1, color: () => T.textFaint },
              { key: "player_name", label: "PLAYER", bold: true },
              { key: "position", label: "POS", align: "center", render: (r) => <PosBadge pos={r.position} /> },
              { key: "age", label: "AGE", align: "center", color: () => T.textDim },
              { key: "ppg", label: "PPG", align: "center", bold: true, render: (r) => r.ppg?.toFixed(1) },
              { key: "dynasty_vorp", label: "DYN VORP", align: "center", color: () => T.accentText, bold: true, render: (r) => r.dynasty_vorp?.toFixed(1) },
              { key: "injury_adjusted_vorp", label: "INJ ADJ", align: "center", render: (r) => {
                if (!r.injury_adjusted_vorp || r.injury_status === "Healthy") return <span style={{ color: T.textFaint }}>—</span>;
                return <span style={{ color: T.red }}>{r.injury_adjusted_vorp?.toFixed(1)}</span>;
              }},
              { key: "injury_status", label: "INJURY", render: (r) => {
                if (r.injury_status === "Healthy") return <span style={{ color: T.textFaint }}>—</span>;
                return <span style={{ color: T.yellow, fontSize: 10 }}>{r.injury_status}</span>;
              }},
            ]}
            rows={filteredAvailable.map((r, i) => ({ ...r, _rank: i + 1 }))}
          />
        </Card>

        {/* Right: My Roster + Positional Need */}
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <Card style={{ overflow: "hidden" }}>
            <div style={{ padding: "10px 14px", borderBottom: `1px solid ${T.border}` }}>
              <SectionTitle>📋 My Roster (Slot {mySlot})</SectionTitle>
            </div>
            {av?.my_picks?.length ? (
              <DataTable
                maxHeight={240}
                columns={[
                  { key: "round", label: "RD", align: "center", color: () => T.textFaint },
                  { key: "player_name", label: "PLAYER", bold: true },
                  { key: "position", label: "POS", align: "center", render: (r) => <PosBadge pos={r.position} /> },
                ]}
                rows={av.my_picks}
              />
            ) : (
              <div style={{ padding: 20, textAlign: "center", color: T.textMuted, fontSize: 12 }}>
                No picks yet
              </div>
            )}
          </Card>

          <Card style={{ padding: 14 }}>
            <SectionTitle>📊 Positional Need</SectionTitle>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
              {["QB", "RB", "WR", "TE"].map((pos) => {
                const n = need[pos] ?? 0;
                const filled = n === 0;
                return (
                  <div key={pos} style={{
                    background: filled ? "rgba(34,197,94,0.06)" : "rgba(234,179,8,0.06)",
                    border: `1px solid ${filled ? "rgba(34,197,94,0.15)" : "rgba(234,179,8,0.15)"}`,
                    borderRadius: T.radius, padding: "8px 10px",
                    display: "flex", justifyContent: "space-between", alignItems: "center",
                  }}>
                    <span style={{ color: posColor(pos), fontWeight: 700, fontSize: 12 }}>{pos}</span>
                    <span style={{ fontSize: 11, color: filled ? T.green : T.yellow }}>
                      {filled ? "✓" : `Need ${n}`}
                    </span>
                  </div>
                );
              })}
            </div>
          </Card>

          {/* Recent picks */}
          {ds?.picks?.length > 0 && (
            <Card style={{ overflow: "hidden" }}>
              <div style={{ padding: "10px 14px", borderBottom: `1px solid ${T.border}` }}>
                <SectionTitle>🕐 Recent Picks</SectionTitle>
              </div>
              <DataTable
                maxHeight={180}
                columns={[
                  { key: "pick_no", label: "#", align: "center", color: () => T.textFaint },
                  { key: "player_name", label: "PLAYER", bold: true },
                  { key: "position", label: "POS", align: "center", render: (r) => <PosBadge pos={r.position} /> },
                  { key: "picked_by", label: "BY", color: () => T.textDim, render: (r) => r.picked_by?.slice(0, 10) },
                ]}
                rows={[...(ds.picks || [])].reverse().slice(0, 8)}
              />
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════
// RANKINGS PAGE
// ═══════════════════════════════════════════════════════════

function RankingsPage({ leagueId }) {
  const [players, setPlayers] = useState([]);
  const [posFilter, setPosFilter] = useState(["QB", "RB", "WR", "TE"]);
  const [showInjuries, setShowInjuries] = useState(true);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    const endpoint = showInjuries ? "/rankings/injured" : "/rankings";
    api(`${endpoint}?league_id=${leagueId}&top_n=150`)
      .then(setPlayers)
      .catch(() => setPlayers([]))
      .finally(() => setLoading(false));
  }, [leagueId, showInjuries]);

  const filtered = players.filter((p) => posFilter.includes(p.position));
  const rankCol = showInjuries ? "injury_adjusted_vorp" : "dynasty_vorp";

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12, flexWrap: "wrap", gap: 8 }}>
        <SectionTitle>Dynasty VORP Rankings {showInjuries ? "(Injury-Adjusted)" : ""}</SectionTitle>
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <label style={{ fontSize: 11, color: T.textMuted, display: "flex", alignItems: "center", gap: 4, cursor: "pointer" }}>
            <input type="checkbox" checked={showInjuries} onChange={(e) => setShowInjuries(e.target.checked)} />
            Injury adjustments
          </label>
          <PosFilterBar value={posFilter} onChange={setPosFilter} />
        </div>
      </div>

      {/* Scatter plot */}
      <Card style={{ padding: 16, marginBottom: 12 }}>
        <ResponsiveContainer width="100%" height={260}>
          <ScatterChart margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={T.border} />
            <XAxis dataKey="age" name="Age" type="number" domain={[20, 38]} stroke={T.textFaint} tick={{ fontSize: 10, fill: T.textMuted }} />
            <YAxis dataKey={rankCol} name="Dynasty VORP" stroke={T.textFaint} tick={{ fontSize: 10, fill: T.textMuted }} />
            <Tooltip content={({ payload }) => {
              if (!payload?.length) return null;
              const p = payload[0].payload;
              return (
                <div style={{ background: T.bg2, border: `1px solid ${T.border}`, borderRadius: T.radius, padding: "8px 12px", fontSize: 11 }}>
                  <div style={{ fontWeight: 700, color: posColor(p.position) }}>{p.player_name}</div>
                  <div style={{ color: T.textDim }}>{p.position} · Age {p.age}</div>
                  <div>PPG: {p.ppg?.toFixed(1)} · VORP: {p.vorp?.toFixed(1)}</div>
                  <div style={{ color: T.accentText, fontWeight: 600 }}>Dynasty: {p[rankCol]?.toFixed(1)}</div>
                  {p.injury_status && p.injury_status !== "Healthy" && (
                    <div style={{ color: T.yellow, fontSize: 10 }}>{p.injury_status} ({p.injury_discount_pct}% disc.)</div>
                  )}
                </div>
              );
            }} />
            {posFilter.map((pos) => (
              <Scatter key={pos} data={filtered.filter((p) => p.position === pos)} fill={posColor(pos)} fillOpacity={0.65} />
            ))}
          </ScatterChart>
        </ResponsiveContainer>
      </Card>

      {/* Table */}
      <Card style={{ overflow: "hidden" }}>
        {loading ? (
          <div style={{ padding: 40, textAlign: "center", color: T.textMuted }}>Loading rankings...</div>
        ) : (
          <DataTable
            maxHeight={500}
            columns={[
              { key: "_i", label: "#", align: "center", render: (_, i) => i + 1, color: () => T.textFaint },
              { key: "player_name", label: "PLAYER", bold: true },
              { key: "position", label: "POS", align: "center", render: (r) => <PosBadge pos={r.position} /> },
              { key: "age", label: "AGE", align: "center", color: () => T.textDim },
              { key: "ppg", label: "PPG", align: "center", render: (r) => r.ppg?.toFixed(1), bold: true },
              { key: "vorp", label: "VORP", align: "center", render: (r) => r.vorp?.toFixed(1), color: (r) => r.vorp > 0 ? T.green : T.red },
              { key: "dynasty_vorp", label: "DYN VORP", align: "center", render: (r) => r.dynasty_vorp?.toFixed(1), color: () => T.accentText, bold: true },
              ...(showInjuries ? [
                { key: "injury_adjusted_vorp", label: "INJ ADJ", align: "center", render: (r) => r.injury_adjusted_vorp?.toFixed(1), color: (r) => r.injury_status !== "Healthy" ? T.yellow : T.accentText, bold: true },
                { key: "injury_discount_pct", label: "DISC %", align: "center", render: (r) => r.injury_discount_pct > 0 ? `−${r.injury_discount_pct}%` : "—", color: (r) => r.injury_discount_pct > 0 ? T.red : T.textFaint },
              ] : []),
            ]}
            rows={filtered.slice(0, 80)}
          />
        )}
      </Card>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════
// INJURIES PAGE
// ═══════════════════════════════════════════════════════════

function InjuriesPage() {
  const [injuries, setInjuries] = useState([]);
  const [types, setTypes] = useState({});
  const [form, setForm] = useState({ player_name: "", position: "WR", injury_type: "acl", severity: "moderate", injury_date: "", surgery: false, notes: "" });

  useEffect(() => {
    api("/injuries").then(setInjuries).catch(() => {});
    api("/injuries/types").then(setTypes).catch(() => {});
  }, []);

  const addInjury = async () => {
    if (!form.player_name) return;
    await api("/injuries", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(form),
    });
    const updated = await api("/injuries");
    setInjuries(updated);
    setForm({ ...form, player_name: "", notes: "" });
  };

  const inputStyle = {
    background: T.bg2, border: `1px solid ${T.border}`, borderRadius: T.radius,
    padding: "7px 10px", color: T.text, fontSize: 11, fontFamily: T.font,
  };

  return (
    <div>
      <SectionTitle>🏥 Injury Tracker</SectionTitle>

      {/* Add injury form */}
      <Card style={{ padding: 14, marginBottom: 12 }}>
        <div style={{ fontSize: 11, color: T.textMuted, marginBottom: 10, fontWeight: 600 }}>ADD INJURY</div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "flex-end" }}>
          <input placeholder="Player name" value={form.player_name} onChange={(e) => setForm({ ...form, player_name: e.target.value })} style={{ ...inputStyle, flex: 1, minWidth: 160 }} />
          <select value={form.position} onChange={(e) => setForm({ ...form, position: e.target.value })} style={inputStyle}>
            {["QB", "RB", "WR", "TE"].map((p) => <option key={p}>{p}</option>)}
          </select>
          <select value={form.injury_type} onChange={(e) => setForm({ ...form, injury_type: e.target.value })} style={{ ...inputStyle, minWidth: 130 }}>
            {Object.entries(types).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}
          </select>
          <select value={form.severity} onChange={(e) => setForm({ ...form, severity: e.target.value })} style={inputStyle}>
            {["minor", "moderate", "major", "career_threatening"].map((s) => <option key={s}>{s}</option>)}
          </select>
          <input type="date" value={form.injury_date} onChange={(e) => setForm({ ...form, injury_date: e.target.value })} style={inputStyle} />
          <label style={{ fontSize: 11, color: T.textMuted, display: "flex", alignItems: "center", gap: 4 }}>
            <input type="checkbox" checked={form.surgery} onChange={(e) => setForm({ ...form, surgery: e.target.checked })} />
            Surgery
          </label>
          <button onClick={addInjury} style={{
            background: T.accent, border: "none", borderRadius: T.radius,
            padding: "7px 14px", color: "#fff", fontSize: 11, fontWeight: 700,
            cursor: "pointer", fontFamily: T.font,
          }}>
            ADD
          </button>
        </div>
      </Card>

      {/* Injury list */}
      <Card style={{ overflow: "hidden" }}>
        {injuries.length === 0 ? (
          <div style={{ padding: 30, textAlign: "center", color: T.textMuted, fontSize: 12 }}>No injuries tracked. Add one above or load from Sleeper.</div>
        ) : (
          <DataTable
            columns={[
              { key: "player_name", label: "PLAYER", bold: true },
              { key: "position", label: "POS", align: "center", render: (r) => <PosBadge pos={r.position} /> },
              { key: "injury", label: "INJURY", color: () => T.yellow },
              { key: "severity", label: "SEVERITY", align: "center", color: (r) => r.severity === "career_threatening" ? T.red : r.severity === "major" ? T.yellow : T.textDim },
              { key: "surgery", label: "SURG", align: "center", render: (r) => r.surgery ? "✓" : "—" },
              { key: "injury_date", label: "DATE", color: () => T.textDim },
              { key: "source", label: "SRC", color: () => T.textFaint },
            ]}
            rows={injuries}
          />
        )}
      </Card>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════
// AGING CURVES PAGE
// ═══════════════════════════════════════════════════════════

function AgingPage() {
  const [data, setData] = useState([]);

  useEffect(() => {
    api("/aging-curves?source=model").then(setData).catch(() => {});
  }, []);

  return (
    <div>
      <SectionTitle>📈 Positional Aging Curves</SectionTitle>
      <p style={{ color: T.textMuted, fontSize: 12, margin: "0 0 14px" }}>
        Year-over-year production retention factor. Above 1.0 = growth, below 1.0 = decline.
      </p>
      <Card style={{ padding: 16 }}>
        <ResponsiveContainer width="100%" height={340}>
          <LineChart data={data} margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={T.border} />
            <XAxis dataKey="age" stroke={T.textFaint} tick={{ fontSize: 10, fill: T.textMuted }} />
            <YAxis domain={[0.7, 1.12]} stroke={T.textFaint} tick={{ fontSize: 10, fill: T.textMuted }} />
            <Tooltip contentStyle={{ background: T.bg2, border: `1px solid ${T.border}`, borderRadius: T.radius, fontSize: 11 }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            {["QB", "RB", "WR", "TE"].map((pos) => (
              <Line key={pos} type="monotone" dataKey={`${pos}_retention`} name={pos} stroke={posColor(pos)} strokeWidth={2.5} dot={{ r: 2.5 }} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </Card>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 10, marginTop: 12 }}>
        {[
          { pos: "QB", note: "Long prime (26–32). Gradual decline after 34. Most durable position." },
          { pos: "RB", note: "Short peak (23–26). Sharp cliff at 27+. Highest bust risk in dynasty." },
          { pos: "WR", note: "Broad peak (24–29). Steady decline from 30. Best long-term value." },
          { pos: "TE", note: "Late bloomer (peak 25–29). Moderate decline. Often undervalued early." },
        ].map(({ pos, note }) => (
          <Card key={pos} style={{ padding: "10px 14px" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
              <div style={{ width: 8, height: 8, borderRadius: 2, background: posColor(pos) }} />
              <span style={{ fontWeight: 700, fontSize: 12, color: posColor(pos) }}>{pos}</span>
            </div>
            <p style={{ color: T.textDim, fontSize: 11, margin: 0, lineHeight: 1.5 }}>{note}</p>
          </Card>
        ))}
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════
// SCARCITY PAGE
// ═══════════════════════════════════════════════════════════

function ScarcityPage({ leagueId }) {
  const [data, setData] = useState([]);

  useEffect(() => {
    api(`/scarcity?league_id=${leagueId}`).then(setData).catch(() => {});
  }, [leagueId]);

  return (
    <div>
      <SectionTitle>⚖️ Positional Scarcity</SectionTitle>
      <p style={{ color: T.textMuted, fontSize: 12, margin: "0 0 14px" }}>
        Higher scarcity = steeper talent drop-off = draft that position earlier.
      </p>
      <Card style={{ padding: 16, marginBottom: 12 }}>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={data} margin={{ top: 10, right: 20, bottom: 10, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={T.border} />
            <XAxis dataKey="position" stroke={T.textFaint} tick={{ fontSize: 12, fontWeight: 700, fill: T.textMuted }} />
            <YAxis stroke={T.textFaint} tick={{ fontSize: 10, fill: T.textMuted }} />
            <Tooltip contentStyle={{ background: T.bg2, border: `1px solid ${T.border}`, borderRadius: T.radius, fontSize: 11 }} />
            <Bar dataKey="scarcity_score" name="Scarcity" radius={[4, 4, 0, 0]}>
              {data.map((d) => <Cell key={d.position} fill={posColor(d.position)} fillOpacity={0.8} />)}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </Card>
      <Card style={{ overflow: "hidden" }}>
        <DataTable
          columns={[
            { key: "position", label: "POS", align: "center", render: (r) => <span style={{ color: posColor(r.position), fontWeight: 700 }}>{r.position}</span> },
            { key: "top5_vorp_share", label: "TOP-5 SHARE", align: "center", render: (r) => `${(r.top5_vorp_share * 100).toFixed(1)}%` },
            { key: "max_vorp", label: "MAX VORP", align: "center", color: () => T.green, bold: true, render: (r) => r.max_vorp?.toFixed(1) },
            { key: "median_starter_vorp", label: "MEDIAN", align: "center", color: () => T.textDim, render: (r) => r.median_starter_vorp?.toFixed(1) },
            { key: "dropoff", label: "DROP-OFF", align: "center", color: () => T.yellow, bold: true, render: (r) => r.dropoff?.toFixed(1) },
            { key: "scarcity_score", label: "SCARCITY", align: "center", color: () => T.accentText, bold: true, render: (r) => r.scarcity_score?.toFixed(2) },
          ]}
          rows={data}
        />
      </Card>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════
// COMPARE PAGE
// ═══════════════════════════════════════════════════════════

function ComparePage({ leagueId }) {
  const [players, setPlayers] = useState([]);
  const [nameA, setNameA] = useState("");
  const [nameB, setNameB] = useState("");

  useEffect(() => {
    api(`/rankings/injured?league_id=${leagueId}&top_n=150`)
      .then(setPlayers).catch(() => {});
  }, [leagueId]);

  const pA = players.find((p) => p.player_name === nameA);
  const pB = players.find((p) => p.player_name === nameB);
  const names = players.map((p) => p.player_name);

  const selectStyle = {
    background: T.bg2, border: `1px solid ${T.border}`, borderRadius: T.radius,
    padding: "8px 12px", color: T.text, fontSize: 12, flex: 1, minWidth: 180, fontFamily: T.font,
  };

  return (
    <div>
      <SectionTitle>🔍 Player Comparison</SectionTitle>
      <div style={{ display: "flex", gap: 12, marginBottom: 16, alignItems: "center", flexWrap: "wrap" }}>
        <select value={nameA} onChange={(e) => setNameA(e.target.value)} style={selectStyle}>
          <option value="">Select Player A</option>
          {names.map((n) => <option key={n} value={n}>{n}</option>)}
        </select>
        <span style={{ color: T.textFaint, fontWeight: 700 }}>vs</span>
        <select value={nameB} onChange={(e) => setNameB(e.target.value)} style={selectStyle}>
          <option value="">Select Player B</option>
          {names.map((n) => <option key={n} value={n}>{n}</option>)}
        </select>
      </div>

      {pA && pB ? (
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          {[pA, pB].map((p) => (
            <Card key={p.player_name} style={{ padding: 16, borderColor: posColor(p.position) + "33" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 14 }}>
                <div>
                  <div style={{ fontSize: 16, fontWeight: 700, fontFamily: "'IBM Plex Sans', sans-serif" }}>{p.player_name}</div>
                  <PosBadge pos={p.position} />
                  {p.injury_status && p.injury_status !== "Healthy" && (
                    <span style={{ marginLeft: 8, fontSize: 10, color: T.yellow }}>⚠ {p.injury_status}</span>
                  )}
                </div>
                <div style={{ textAlign: "right" }}>
                  <div style={{ fontSize: 26, fontWeight: 800, color: T.accentText }}>
                    {(p.injury_adjusted_vorp || p.dynasty_vorp)?.toFixed(1)}
                  </div>
                  <div style={{ fontSize: 9, color: T.textMuted }}>DYNASTY VORP</div>
                </div>
              </div>
              {[
                ["Age", p.age],
                ["PPG", p.ppg?.toFixed(1)],
                ["Total Points", p.total_points?.toFixed(0)],
                ["Games", p.games_played],
                ["VORP", p.vorp?.toFixed(1)],
                ["Dynasty VORP", p.dynasty_vorp?.toFixed(1)],
                ...(p.injury_adjusted_vorp ? [["Injury-Adj VORP", p.injury_adjusted_vorp?.toFixed(1)]] : []),
                ...(p.injury_discount_pct > 0 ? [["Injury Discount", `−${p.injury_discount_pct}%`]] : []),
              ].map(([label, val]) => (
                <div key={label} style={{
                  display: "flex", justifyContent: "space-between",
                  padding: "5px 0", borderBottom: `1px solid ${T.border}22`, fontSize: 12,
                }}>
                  <span style={{ color: T.textMuted }}>{label}</span>
                  <span style={{ color: T.text, fontWeight: 600 }}>{val}</span>
                </div>
              ))}
            </Card>
          ))}
        </div>
      ) : (
        <Card style={{ padding: 40, textAlign: "center" }}>
          <p style={{ color: T.textMuted, fontSize: 13 }}>Select two players to compare side-by-side.</p>
        </Card>
      )}
    </div>
  );
}
