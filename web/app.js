// Cricket Analytics Hub: vanilla JS front end for the FastAPI service (src/api.py).

const COLOR_A = "#38ef7d";
const COLOR_B = "#00c6ff";
const COLOR_DANGER = "#ff4b4b";
const STYLE_COLORS = ["#38ef7d", "#00c6ff", "#ffaa00", "#c77dff", "#ff4b4b"];

const FEATURE_LABELS = {
  runs_needed: "Runs needed",
  balls_left: "Balls remaining",
  wickets_left: "Wickets in hand",
  req_rr: "Required run rate",
  curr_rr: "Current run rate",
  target_runs: "Target",
};

const BASE_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { color: "white" },
  margin: { t: 40, b: 40, l: 50, r: 20 },
};
const PLOT_CONFIG = { responsive: true, displayModeBar: false };

const $ = (id) => document.getElementById(id);

let meta = null;
let activeTab = "series";

// ---------------- Helpers ----------------

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}

function qs(params) {
  return new URLSearchParams(Object.entries(params).filter(([, v]) => v !== null && v !== undefined)).toString();
}

async function api(path, body) {
  const opts = body === undefined
    ? {}
    : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const res = await fetch(path, opts);
  if (!res.ok) {
    const detail = await res.json().catch(() => ({}));
    const msg = Array.isArray(detail.detail) ? detail.detail.map((d) => d.msg).join("; ") : detail.detail;
    throw new Error(msg || `${path} failed (${res.status})`);
  }
  return res.json();
}

function showError(err) {
  const box = $("error");
  box.textContent = err ? `⚠️ ${err.message || err}` : "";
  box.classList.toggle("hidden", !err);
}

function fillSelect(select, values, selected) {
  select.innerHTML = values.map((v) => `<option>${escapeHtml(v)}</option>`).join("");
  if (values.includes(selected)) select.value = selected;
}

function fmtDate(iso) {
  return new Date(`${iso}T00:00:00`).toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

function resultText(m) {
  if (m.result === "no result") return "No result";
  if (m.result === "tie") return m.winner ? `Tie (${m.winner} won super over)` : "Tie";
  const by = m.win_by_runs ? `${m.win_by_runs} runs` : `${m.win_by_wickets} wkts`;
  return `${m.winner} won by ${by}${m.method ? ` (${m.method})` : ""}`;
}

function table(el, columns, rows) {
  el.innerHTML = `
    <thead><tr>${columns.map((c) => `<th class="${c.num ? "num" : ""}">${escapeHtml(c.label)}</th>`).join("")}</tr></thead>
    <tbody>${rows.length ? rows.map((r) => `<tr>${columns.map((c) => {
      const v = c.render ? c.render(r) : escapeHtml(r[c.key] ?? "–");
      return `<td class="${c.num ? "num" : ""}">${v}</td>`;
    }).join("")}</tr>`).join("") : `<tr><td colspan="${columns.length}">No data</td></tr>`}</tbody>`;
}

// ---------------- Sidebar state ----------------

function state() {
  const fmt = $("format").value;
  const teamA = $("team-a").value;
  const teamB = $("team-b").value;
  const battingIsA = $("batting-side").value === "a";
  return {
    fmt, teamA, teamB,
    batting: battingIsA ? teamA : teamB,
    bowling: battingIsA ? teamB : teamA,
    chasing: $("innings").value === "2",
    target: Number($("target").value),
    current: Number($("current").value),
    overs: Number($("overs").value),
    wicketsLost: Number($("wickets").value),
  };
}

function oversToBalls(overs) {
  return Math.floor(overs) * 6 + Math.round((overs % 1) * 10);
}

function threeYearsBeforeLatest(fmt) {
  const latest = new Date(`${meta.coverage[fmt].to}T00:00:00`);
  latest.setFullYear(latest.getFullYear() - 3);
  return latest.toISOString().slice(0, 10);
}

// ---------------- Tab: series & head-to-head ----------------

async function renderSeries() {
  const s = state();
  const [series, h2h, phaseA, phaseB, batA, batB, bowlA, bowlB] = await Promise.all([
    api(`/api/v1/series/latest?${qs({ team_a: s.teamA, team_b: s.teamB })}`),
    api(`/api/v1/head-to-head?${qs({ team_a: s.teamA, team_b: s.teamB, format: s.fmt })}`),
    api(`/api/v1/phase-stats?${qs({ team: s.teamA, format: s.fmt, since: threeYearsBeforeLatest(s.fmt) })}`),
    api(`/api/v1/phase-stats?${qs({ team: s.teamB, format: s.fmt, since: threeYearsBeforeLatest(s.fmt) })}`),
    api(`/api/v1/players/batting?${qs({ format: s.fmt, team: s.teamA, opponent: s.teamB, limit: 6 })}`),
    api(`/api/v1/players/batting?${qs({ format: s.fmt, team: s.teamB, opponent: s.teamA, limit: 6 })}`),
    api(`/api/v1/players/bowling?${qs({ format: s.fmt, team: s.teamA, opponent: s.teamB, limit: 6 })}`),
    api(`/api/v1/players/bowling?${qs({ format: s.fmt, team: s.teamB, opponent: s.teamA, limit: 6 })}`),
  ]);

  if (series.event) {
    const tally = {};
    series.matches.forEach((m) => {
      tally[m.format] = tally[m.format] || { [s.teamA]: 0, [s.teamB]: 0 };
      if (m.winner in tally[m.format]) tally[m.format][m.winner] += 1;
    });
    const tallyText = Object.entries(tally)
      .map(([f, t]) => `${f}s: ${escapeHtml(s.teamA)} ${t[s.teamA]}–${t[s.teamB]} ${escapeHtml(s.teamB)}`).join(" • ");
    $("series-card").innerHTML = `
      <div class="headline">📅 Latest bilateral series: ${escapeHtml(series.event)}</div>
      <div class="detail">${tallyText}</div>
      <div class="detail">${series.matches.map((m) => `
        <span class="pill ${m.winner === s.teamA ? "win-a" : m.winner === s.teamB ? "win-b" : ""}">
          ${escapeHtml(m.format)} · ${fmtDate(m.start_date)} · ${escapeHtml(resultText(m))}
        </span>`).join(" ")}</div>
      <div class="detail hint">Matches from a series in progress appear once Cricsheet publishes them — run <code>python -m src.pipeline --refresh</code>.</div>`;
  } else {
    $("series-card").innerHTML = `<div class="headline neutral">No bilateral series between these teams in the data.</div>`;
  }

  const sum = h2h.summary;
  $("h2h-metrics").innerHTML = `
    <div class="metric card"><div class="metric-label">${s.fmt} matches</div><div class="metric-value">${sum.matches}</div>
      <div class="metric-sub">${sum.first_match ? `${fmtDate(sum.first_match)} – ${fmtDate(sum.last_match)}` : ""}</div></div>
    <div class="metric badge-team1"><div class="metric-label">${escapeHtml(s.teamA)} wins</div><div class="metric-value">${sum.team_a_wins}</div></div>
    <div class="metric badge-team2"><div class="metric-label">${escapeHtml(s.teamB)} wins</div><div class="metric-value">${sum.team_b_wins}</div></div>
    <div class="metric card"><div class="metric-label">Win rate (${escapeHtml(s.teamA)})</div>
      <div class="metric-value">${sum.team_a_wins + sum.team_b_wins ? Math.round(sum.team_a_wins / (sum.team_a_wins + sum.team_b_wins) * 100) : 0}%</div>
      <div class="metric-sub">${sum.no_result} no result</div></div>`;

  table($("results-table"), [
    { label: "Date", render: (m) => fmtDate(m.start_date) },
    { label: "Event", key: "event_name" },
    { label: "Result", render: (m) => escapeHtml(resultText(m)) },
    { label: "Player of match", key: "player_of_match" },
  ], h2h.matches.slice(0, 10));

  const phases = meta.formats[s.fmt].phases;
  Plotly.react("chart-phase", [
    { type: "bar", name: s.teamA, x: phases, y: phases.map((p) => phaseA.data[p].run_rate), marker: { color: COLOR_A } },
    { type: "bar", name: s.teamB, x: phases, y: phases.map((p) => phaseB.data[p].run_rate), marker: { color: COLOR_B } },
  ], {
    ...BASE_LAYOUT,
    title: { text: "Batting run rate by phase (last 3 years, all opponents)", font: { size: 15 } },
    yaxis: { title: "Runs per over", gridcolor: "#333" },
    legend: { orientation: "h", y: -0.15 },
  }, PLOT_CONFIG);

  const batCols = [
    { label: "Batter", key: "player" },
    { label: "Inns", key: "innings", num: true },
    { label: "Runs", key: "runs", num: true },
    { label: "Avg", key: "average", num: true },
    { label: "SR", key: "strike_rate", num: true },
  ];
  const bowlCols = [
    { label: "Bowler", key: "player" },
    { label: "Inns", key: "innings", num: true },
    { label: "Wkts", key: "wickets", num: true },
    { label: "Econ", key: "economy", num: true },
    { label: "Avg", key: "average", num: true },
  ];
  $("bat-a-title").textContent = `🏏 ${s.teamA} top batters vs ${s.teamB}`;
  $("bat-b-title").textContent = `🏏 ${s.teamB} top batters vs ${s.teamA}`;
  $("bowl-a-title").textContent = `🎯 ${s.teamA} top bowlers vs ${s.teamB}`;
  $("bowl-b-title").textContent = `🎯 ${s.teamB} top bowlers vs ${s.teamA}`;
  table($("bat-a"), batCols, batA.data);
  table($("bat-b"), batCols, batB.data);
  table($("bowl-a"), bowlCols, bowlA.data);
  table($("bowl-b"), bowlCols, bowlB.data);
}

// ---------------- Tab: live predictor ----------------

async function renderLive() {
  const s = state();
  $("chase-view").classList.toggle("hidden", !s.chasing);
  $("first-view").classList.toggle("hidden", s.chasing);

  if (!s.chasing) {
    const [proj, sim] = await Promise.all([
      api("/api/v1/project-score", { format: s.fmt, current_score: s.current, overs: s.overs, wickets_lost: s.wicketsLost }),
      api("/api/v1/simulate-match", {
        format: s.fmt, target_score: null, current_score: s.current, overs: s.overs,
        wickets_lost: s.wicketsLost, batting_team: s.batting, n_simulations: 5000,
      }),
    ]);
    const b = sim.baseline;
    $("verdict").innerHTML = `
      <div class="headline">📢 ${escapeHtml(s.batting)} are heading for about ${proj.projected_total}</div>
      <div class="detail">${s.current}/${s.wicketsLost} after ${s.overs} overs. The ML model projects ${proj.projected_total};
      a Monte Carlo simulation using ${escapeHtml(s.batting)}'s recent scoring puts the likely range at ${b.ci_10th}–${b.ci_90th}.</div>`;
    $("m-projection").textContent = proj.projected_total;
    $("m-projection-sub").textContent = `Regression model trained on past ${s.fmt} first innings`;
    $("m-sim-range").textContent = `${b.ci_10th} – ${b.ci_90th}`;
    $("m-first-crr").textContent = `${(s.current / Math.max(oversToBalls(s.overs), 1) * 6).toFixed(2)} RPO`;
    return;
  }

  const res = await api("/api/v1/predict-win-prob", {
    format: s.fmt, target: s.target, current_score: s.current, overs: s.overs, wickets_lost: s.wicketsLost,
  });
  const { state: st, prediction: p, pressure } = res;
  const favored = p.chasing_win_pct >= 50 ? s.batting : s.bowling;

  $("verdict").innerHTML = `
    <div class="headline">📢 ${escapeHtml(favored)} favoured (${Math.max(p.chasing_win_pct, p.defending_win_pct)}%)</div>
    <div class="detail">
      ${escapeHtml(s.batting)} need <strong>${st.runs_needed}</strong> off <strong>${st.balls_left}</strong> balls
      (required <strong>${st.req_rr} RPO</strong>) with <strong>${st.wickets_left}</strong> wickets in hand.
      Pressure: <strong>${pressure.emoji} ${escapeHtml(pressure.level)}</strong> — ${escapeHtml(pressure.desc)}
    </div>
    <div class="detail hint">The model reads the scoreboard only (runs, balls, wickets, target) — it does not rate the two teams.</div>`;
  $("m-bat-label").textContent = `${s.batting} (chasing)`;
  $("m-bat").textContent = `${p.chasing_win_pct}%`;
  $("m-bowl-label").textContent = `${s.bowling} (defending)`;
  $("m-bowl").textContent = `${p.defending_win_pct}%`;
  $("m-pressure").textContent = `${pressure.emoji} ${pressure.level}`;
  $("m-mpi").textContent = `MPI ${pressure.mpi}`;
  $("m-rrr").textContent = `${st.req_rr} RPO`;
  $("m-crr").textContent = `Current ${st.curr_rr} RPO`;

  Plotly.react("chart-donut", [{
    type: "pie",
    labels: [s.batting, s.bowling],
    values: [p.chasing_win_pct, p.defending_win_pct],
    hole: 0.65,
    sort: false,
    marker: { colors: [COLOR_A, COLOR_B], line: { color: "#0e1117", width: 3 } },
    textinfo: "label+percent",
  }], {
    ...BASE_LAYOUT,
    margin: { t: 20, b: 40, l: 20, r: 20 },
    legend: { orientation: "h", y: -0.2, x: 0.5, xanchor: "center" },
    annotations: [{ text: "<b>WIN PROBABILITY</b>", x: 0.5, y: 0.5, showarrow: false }],
  }, PLOT_CONFIG);

  const shap = p.shap_impacts;
  Plotly.react("chart-shap", [{
    type: "bar",
    orientation: "h",
    x: shap.map((d) => d.Impact),
    y: shap.map((d) => FEATURE_LABELS[d.Feature] || d.Feature),
    marker: { color: shap.map((d) => (d.Impact > 0 ? COLOR_A : COLOR_DANGER)) },
    text: shap.map((d) => (d.Impact >= 0 ? "+" : "") + d.Impact.toFixed(2)),
    textposition: "auto",
  }], {
    ...BASE_LAYOUT,
    title: { text: "Why: SHAP contributions vs an average chase", font: { size: 15 } },
    xaxis: { title: "Effect on chase win log-odds", gridcolor: "#333", zerolinecolor: "#666" },
    yaxis: { autorange: "reversed" },
    margin: { t: 40, b: 50, l: 130, r: 20 },
  }, PLOT_CONFIG);
}

// ---------------- Tab: matchups ----------------

async function renderMatchups() {
  const s = state();
  const res = await api("/api/v1/h2h-matrix", { format: s.fmt, team_a: s.batting, team_b: s.bowling });
  $("matchup-caption").textContent =
    `${s.batting} batters vs ${s.bowling} bowlers — XIs from their latest ${s.fmt} meeting, records across all ${s.fmt} internationals. Change "Batting side" to flip.`;

  const z = res.matrix.map((row) => res.bowlers.map((b) => (row[b].balls ? row[b].sr : null)));
  const text = res.matrix.map((row) => res.bowlers.map((b) => {
    const c = row[b];
    return c.balls ? `${c.runs}r/${c.balls}b<br>${c.dismissals}W` : "";
  }));
  Plotly.react("chart-heatmap", [{
    type: "heatmap",
    z, x: res.bowlers, y: res.batters, text,
    texttemplate: "%{text}",
    hovertemplate: "%{y} vs %{x}<br>SR %{z}<extra></extra>",
    colorscale: "Viridis",
    colorbar: { title: { text: "Strike rate" } },
    hoverongaps: false,
  }], {
    ...BASE_LAYOUT,
    xaxis: { title: `Bowlers (${res.team_b})`, tickangle: -45 },
    yaxis: { title: `Batters (${res.team_a})`, autorange: "reversed" },
    margin: { t: 20, b: 130, l: 140, r: 40 },
    height: 650,
  }, PLOT_CONFIG);

  $("alerts").innerHTML = res.alerts.length
    ? res.alerts.slice(0, 6).map((a) => `
        <div class="alert-card">
          <strong>${escapeHtml(a.bowler)}</strong> vs <strong>${escapeHtml(a.batter)}</strong><br>
          ${a.runs} runs off ${a.balls} balls • ${a.dismissals} dismissal${a.dismissals === 1 ? "" : "s"} • SR ${a.sr}
        </div>`).join("")
    : `<div class="info-card">✅ No bowler has a dominant record (12+ balls) against these batters.</div>`;
}

// ---------------- Tab: simulator ----------------

function statBlock(label, value, delta, unit = "") {
  const deltaHtml = delta === undefined || delta === null ? "" :
    `<div class="d ${delta >= 0 ? "up" : "down"}">${delta >= 0 ? "+" : ""}${delta.toFixed(1)}${unit} vs baseline</div>`;
  return `<div class="stat"><div class="k">${label}</div><div class="v">${value}</div>${deltaHtml}</div>`;
}

async function renderSim() {
  const s = state();
  const button = $("run-sim");
  button.disabled = true;
  button.textContent = "Simulating…";
  try {
    const res = await api("/api/v1/simulate-match", {
      format: s.fmt,
      target_score: s.chasing ? s.target : null,
      current_score: s.current,
      overs: s.overs,
      wickets_lost: s.wicketsLost,
      batting_team: $("use-team-profile").checked ? s.batting : null,
      tactical_choice: $("tactic").value,
      n_simulations: Number($("n-sims").value),
    });
    const { baseline, tactical, impact_summary } = res;
    const isBaseline = tactical === baseline || res.tactical_choice === meta.tactics[0];
    const delta = (a, b) => (isBaseline ? null : a - b);
    $("sim-results").innerHTML =
      (s.chasing ? statBlock(`${escapeHtml(s.batting)} chase win chance`, `${tactical.simulated_win_pct}%`,
        delta(tactical.simulated_win_pct, baseline.simulated_win_pct), " pts") : "") +
      statBlock("Expected final score", `${tactical.mean_score}`, delta(tactical.mean_score, baseline.mean_score)) +
      statBlock("Likely range (10th–90th pct)", `${tactical.ci_10th} – ${tactical.ci_90th}`) +
      `<div class="info-card">💡 ${escapeHtml(impact_summary)}</div>`;

    const vline = (x, color) => ({ type: "line", x0: x, x1: x, yref: "paper", y0: 0, y1: 1, line: { color, dash: "dash", width: 2 } });
    const shapes = [vline(tactical.mean_score, COLOR_A), vline(tactical.ci_10th, "#888"), vline(tactical.ci_90th, "#888")];
    if (s.chasing) shapes.push(vline(s.target, COLOR_DANGER));
    Plotly.react("chart-hist", [{
      type: "histogram", x: tactical.score_distribution, nbinsx: 30, marker: { color: COLOR_B },
    }], {
      ...BASE_LAYOUT,
      title: { text: `${tactical.n_simulations.toLocaleString()} simulated finishes for ${s.batting}`, font: { size: 15 } },
      xaxis: { title: "Final score", gridcolor: "#333" },
      yaxis: { title: "Simulations (sample of 1,000)", gridcolor: "#333" },
      shapes,
      annotations: s.chasing ? [{ x: s.target, y: 1, yref: "paper", text: "Target", showarrow: false, yanchor: "bottom", font: { color: COLOR_DANGER } }] : [],
    }, PLOT_CONFIG);
  } finally {
    button.disabled = false;
    button.textContent = "Run simulation";
  }
}

// ---------------- Tab: player styles (clustering) ----------------

async function renderStyles() {
  const s = state();
  const res = await api(`/api/v1/batter-clusters?${qs({ format: s.fmt })}`);
  const styles = [...new Set(res.players.map((p) => p.style))].sort();
  const focus = new Set([s.teamA, s.teamB]);

  $("styles-caption").textContent =
    `KMeans (k=${res.info.k}) on ${res.info.players} ${s.fmt} batters with ${res.info.min_balls}+ balls faced, using strike rate, average, boundary %, dot-ball % and the share of balls faced in the powerplay and death overs. Silhouette score ${res.info.silhouette}. ${s.teamA} and ${s.teamB} players are outlined.`;

  const traces = styles.map((style, i) => {
    const pts = res.players.filter((p) => p.style === style);
    return {
      type: "scatter",
      mode: "markers",
      name: style,
      x: pts.map((p) => p.strike_rate),
      y: pts.map((p) => p.average),
      text: pts.map((p) => `${p.player} (${p.team})`),
      hovertemplate: "%{text}<br>SR %{x:.1f} • Avg %{y:.1f}<extra>" + style + "</extra>",
      marker: {
        color: STYLE_COLORS[i % STYLE_COLORS.length],
        size: pts.map((p) => (focus.has(p.team) ? 12 : 6)),
        opacity: pts.map((p) => (focus.has(p.team) ? 1 : 0.35)),
        line: { color: "white", width: pts.map((p) => (focus.has(p.team) ? 1.5 : 0)) },
      },
    };
  });
  Plotly.react("chart-clusters", traces, {
    ...BASE_LAYOUT,
    xaxis: { title: "Strike rate", gridcolor: "#333" },
    yaxis: { title: "Batting average (capped at 80)", gridcolor: "#333" },
    legend: { orientation: "h", y: -0.12 },
    height: 600,
  }, PLOT_CONFIG);

  const rows = res.players.filter((p) => focus.has(p.team)).sort((a, b) => b.runs - a.runs).slice(0, 40);
  table($("styles-table"), [
    { label: "Player", key: "player" },
    { label: "Team", key: "team" },
    { label: "Style (cluster)", key: "style" },
    { label: "Runs", key: "runs", num: true },
    { label: "SR", render: (p) => p.strike_rate.toFixed(1), num: true },
    { label: "Avg", render: (p) => p.average.toFixed(1), num: true },
    { label: "Boundary %", render: (p) => p.boundary_pct.toFixed(1), num: true },
    { label: "Powerplay share", render: (p) => `${Math.round(p.powerplay_share * 100)}%`, num: true },
    { label: "Death share", render: (p) => `${Math.round(p.death_share * 100)}%`, num: true },
  ], rows);
}

// ---------------- Tab: model report ----------------

async function renderModels() {
  const s = state();
  const report = (await api("/api/v1/model-report"))[s.fmt];
  const win = report.win_probability;
  const score = report.score_projection;
  const clusters = report.batter_clusters;

  const metricTable = (models, cols, selected) => `
    <table><thead><tr><th>Model</th>${cols.map((c) => `<th class="num">${c.label}</th>`).join("")}</tr></thead>
    <tbody>${Object.entries(models).map(([name, m]) => `
      <tr><td class="${name === selected ? "selected-model" : ""}">${escapeHtml(name)}${name === selected ? " ✓ selected" : ""}</td>
      ${cols.map((c) => `<td class="num">${m[c.key]}</td>`).join("")}</tr>`).join("")}</tbody></table>`;

  $("model-report").innerHTML = `
    <div class="report-block">
      <h3>1. ${escapeHtml(win.task)}</h3>
      <p>Trained on ${win.train_matches.toLocaleString()} matches (${win.train_rows.toLocaleString()} ball states);
      tested on the ${win.test_matches} most recent matches from ${fmtDate(win.test_from)} onward. XGBoost replaces the logistic regression baseline only if its test log loss is lower by more than 0.005.</p>
      <div class="table-wrap">${metricTable(win.models, [
        { label: "Log loss ↓", key: "log_loss" }, { label: "Brier ↓", key: "brier" },
        { label: "Accuracy ↑", key: "accuracy" }, { label: "ROC AUC ↑", key: "roc_auc" },
      ], win.selected)}</div>
    </div>
    <div class="report-block">
      <h3>2. ${escapeHtml(score.task)}</h3>
      <p>Trained on ${score.train_matches.toLocaleString()} first innings; tested on ${score.test_matches} recent ones. Errors in runs.</p>
      <div class="table-wrap">${metricTable(score.models, [
        { label: "MAE ↓", key: "mae" }, { label: "RMSE ↓", key: "rmse" }, { label: "MAE at halfway ↓", key: "mae_at_halfway" },
      ], score.selected)}</div>
    </div>
    <div class="report-block">
      <h3>3. ${escapeHtml(clusters.task)}</h3>
      <p>${clusters.players} batters, k=${clusters.k}, silhouette ${clusters.silhouette}.</p>
      <div class="table-wrap"><table><thead><tr><th>Style</th><th class="num">Players</th><th class="num">SR</th><th class="num">Avg</th><th class="num">Boundary %</th><th class="num">Powerplay share</th><th class="num">Death share</th></tr></thead>
      <tbody>${clusters.clusters.map((c) => `<tr><td>${escapeHtml(c.style)}</td><td class="num">${c.players}</td><td class="num">${c.strike_rate}</td>
        <td class="num">${c.average}</td><td class="num">${c.boundary_pct}</td><td class="num">${Math.round(c.powerplay_share * 100)}%</td>
        <td class="num">${Math.round(c.death_share * 100)}%</td></tr>`).join("")}</tbody></table></div>
    </div>`;

  const cal = win.calibration;
  Plotly.react("chart-calibration", [
    { type: "scatter", mode: "lines", x: [0, 1], y: [0, 1], name: "Perfect calibration", line: { color: "#666", dash: "dash" } },
    {
      type: "scatter", mode: "lines+markers", name: win.selected,
      x: cal.map((c) => c.predicted), y: cal.map((c) => c.observed),
      text: cal.map((c) => `${c.n.toLocaleString()} ball states`),
      marker: { color: COLOR_A, size: 8 }, line: { color: COLOR_A },
    },
  ], {
    ...BASE_LAYOUT,
    title: { text: `Win-probability calibration on held-out ${s.fmt} matches`, font: { size: 15 } },
    xaxis: { title: "Predicted chase win probability", range: [0, 1], gridcolor: "#333" },
    yaxis: { title: "Observed win rate", range: [0, 1], gridcolor: "#333" },
    legend: { orientation: "h", y: -0.2 },
  }, PLOT_CONFIG);
}

// ---------------- Wiring ----------------

const RENDERERS = { series: renderSeries, live: renderLive, matchups: renderMatchups, sim: renderSim, styles: renderStyles, models: renderModels };

async function refresh() {
  try {
    await RENDERERS[activeTab]();
    showError(null);
  } catch (err) {
    showError(err);
  }
}

let debounceTimer = null;
function scheduleRefresh() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(refresh, 300);
}

function onFormatChange() {
  const fmt = $("format").value;
  const teams = meta.formats[fmt].teams;
  fillSelect($("team-a"), teams, $("team-a").value || meta.defaults.team_a);
  syncTeamB();
  $("overs").max = `${meta.formats[fmt].overs - 1}.5`;
}

function syncTeamB() {
  const teams = meta.formats[$("format").value].teams.filter((t) => t !== $("team-a").value);
  fillSelect($("team-b"), teams, $("team-b").value || meta.defaults.team_b);
  $("batting-side").options[0].textContent = $("team-a").value;
  $("batting-side").options[1].textContent = $("team-b").value;
}

function switchTab(name) {
  activeTab = name;
  history.replaceState(null, "", `#${name}`);
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("hidden", p.id !== `tab-${name}`));
  refresh();
}

async function init() {
  try {
    meta = await api("/api/v1/meta");
  } catch (err) {
    showError(err);
    return;
  }
  $("coverage").textContent = Object.entries(meta.coverage)
    .map(([f, c]) => `${c.matches.toLocaleString()} ${f}s (${c.from.slice(0, 4)}–${fmtDate(c.to)})`).join(" • ");
  fillSelect($("tactic"), meta.tactics, meta.tactics[0]);
  fillSelect($("team-a"), meta.formats.T20.teams, meta.defaults.team_a);
  onFormatChange();

  $("format").addEventListener("change", () => { onFormatChange(); scheduleRefresh(); });
  $("team-a").addEventListener("change", () => { syncTeamB(); scheduleRefresh(); });
  $("team-b").addEventListener("change", () => { syncTeamB(); scheduleRefresh(); });
  $("innings").addEventListener("change", () => {
    $("target-label").classList.toggle("hidden", $("innings").value !== "2");
    scheduleRefresh();
  });
  ["batting-side", "target", "current", "overs", "wickets"].forEach((id) => $(id).addEventListener("input", scheduleRefresh));
  ["tactic", "n-sims", "use-team-profile"].forEach((id) => $(id).addEventListener("change", scheduleRefresh));
  $("run-sim").addEventListener("click", refresh);
  document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => switchTab(b.dataset.tab)));

  // Deep links: /app/#live opens the Live Predictor tab
  const fromHash = location.hash.slice(1);
  switchTab(fromHash in RENDERERS ? fromHash : "series");
}

init();
