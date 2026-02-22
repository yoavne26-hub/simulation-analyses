const runStatus = document.getElementById("runStatus");
const analysisStatus = document.getElementById("analysisStatus");
const summary = document.getElementById("summary");
const plots = document.getElementById("plots");
const insightsBox = document.getElementById("insights");
const lastRun = document.getElementById("lastRun");

const overlay = document.getElementById("loadingOverlay");
const overlayStatus = document.getElementById("overlayStatus");
const overlayProgressBar = document.getElementById("overlayProgressBar");
const overlayProgressText = document.getElementById("overlayProgressText");
const overlayEta = document.getElementById("overlayEta");
const overlayElapsed = document.getElementById("overlayElapsed");
const overlayScenario = document.getElementById("overlayScenario");
const overlayRun = document.getElementById("overlayRun");

const scenarioChecklist = document.getElementById("scenarioChecklist");
const featureChecklist = document.getElementById("featureChecklist");

const ridgeAlphaInput = document.getElementById("ridgeAlpha");
const ridgeAutoInput = document.getElementById("ridgeAuto");
const boxCoxCheckInput = document.getElementById("boxCoxCheck");

const chart = document.getElementById("barChart");
const ctx = chart.getContext("2d");

const datasetPath = "data/simulation_results.csv";
let statusTimer = null;

const defaultScenarioOptions = [
  { value: "BASE", label: "BASE (No investments)" },
  { value: "ALT1", label: "Kitchen + Marketing + Website" },
  { value: "ALT2", label: "Website + Large Tube Capacity" },
  { value: "ALT3", label: "Marketing Only" },
  { value: "ALT4", label: "Kitchen Quality Only" },
  { value: "ALT5", label: "Website Only" },
  { value: "ALT6", label: "Kitchen + Website" },
  { value: "ALT7", label: "Capacity Expansion" },
  { value: "ALT8", label: "Premium Package" },
  { value: "ALT9", label: "Budget Marketing" },
  { value: "ALT10", label: "Kitchen + Capacity" },
];

const defaultFeatureOptions = [
  "avg_rating",
  "avg_food_income",
  "avg_food_income_per_visitor",
  "avg_reception_income",
  "avg_photo_income",
  "avg_abandonments_per_visitor",
  "total_customers",
  "customers_arriving",
  "customers_leaving",
  "total_people_ate",
  "food_income",
  "reception_income",
  "photo_income",
  "total_revenue",
  "abandonments",
  "drop_count",
];

function resizeCanvas() {
  const rect = chart.getBoundingClientRect();
  chart.width = Math.max(360, Math.floor(rect.width));
  chart.height = Math.max(180, Math.floor(rect.height));
}

function setStatus(el, text) {
  el.textContent = text;
}

function setOverlay(visible, text) {
  overlay.classList.toggle("hidden", !visible);
  if (text) {
    overlayStatus.textContent = text;
  }
}

function formatEta(seconds) {
  if (!Number.isFinite(seconds)) return "-";
  const s = Math.max(0, Math.floor(seconds));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return m > 0 ? `${m}m ${r}s` : `${r}s`;
}

function formatElapsed(seconds) {
  if (!Number.isFinite(seconds)) return "--";
  const s = Math.max(0, Math.floor(seconds));
  const m = Math.floor(s / 60);
  const r = s % 60;
  return m > 0 ? `${m}m ${r}s` : `${r}s`;
}

function renderChecklist(container, items, selectedSet) {
  if (!container) return;
  container.innerHTML = "";
  items.forEach((item) => {
    const value = item.value || item;
    const label = item.label || value;
    const wrapper = document.createElement("label");
    wrapper.className = "check-item";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = value;
    if (selectedSet && selectedSet.has(value)) {
      checkbox.checked = true;
    }

    const span = document.createElement("span");
    span.textContent = label;

    wrapper.appendChild(checkbox);
    wrapper.appendChild(span);
    container.appendChild(wrapper);
  });

  if (container.children.length === 0) {
    const empty = document.createElement("div");
    empty.className = "check-empty";
    empty.textContent = "No options available.";
    container.appendChild(empty);
  }
}

function getCheckedValues(container) {
  if (!container) return [];
  return Array.from(container.querySelectorAll('input[type="checkbox"]:checked')).map(
    (input) => input.value
  );
}

async function populateOptions() {
  const defaultSelected = new Set(["avg_rating", "avg_food_income", "total_customers"]);
  try {
    const scenarioRes = await fetch('/scenarios');
    if (scenarioRes.ok) {
      const data = await scenarioRes.json();
      const items = data.scenarios.map((scn) => ({ value: scn.key, label: scn.label }));
      renderChecklist(scenarioChecklist, items);
    } else {
      renderChecklist(scenarioChecklist, defaultScenarioOptions);
    }

    const featureRes = await fetch('/features');
    if (featureRes.ok) {
      const data = await featureRes.json();
      renderChecklist(featureChecklist, data.features, defaultSelected);
    } else {
      renderChecklist(featureChecklist, defaultFeatureOptions, defaultSelected);
    }
  } catch (err) {
    console.error(err);
    renderChecklist(scenarioChecklist, defaultScenarioOptions);
    renderChecklist(featureChecklist, defaultFeatureOptions, defaultSelected);
  }
}

async function postJson(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    throw new Error(await res.text());
  }
  return res.json();
}

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length <= 1) return [];
  const headers = lines[0].split(",").map((h) => h.trim());
  return lines.slice(1).map((line) => {
    const cols = line.split(",");
    const row = {};
    headers.forEach((h, i) => {
      row[h] = cols[i];
    });
    return row;
  });
}

function drawBarChart(rows) {
  resizeCanvas();
  const grouped = {};
  rows.forEach((row) => {
    const key = row.scenario_name || row.scenario_key;
    const value = Number(row.total_revenue || 0);
    if (!grouped[key]) {
      grouped[key] = { total: 0, count: 0 };
    }
    grouped[key].total += value;
    grouped[key].count += 1;
  });

  const labels = Object.keys(grouped);
  const values = labels.map((k) => grouped[k].total / grouped[k].count);

  ctx.clearRect(0, 0, chart.width, chart.height);

  if (labels.length === 0) {
    ctx.fillStyle = "#94a3b8";
    ctx.font = "14px 'Palatino Linotype', serif";
    ctx.fillText("No data yet.", 12, 24);
    return;
  }

  const maxLabelLen = Math.max(...labels.map((label) => label.length));
  const paddingLeft = 64;
  const paddingRight = 24;
  const paddingTop = 24;
  const paddingBottom = Math.min(220, 70 + maxLabelLen * 4);
  const maxVal = Math.max(...values) || 1;
  const plotWidth = chart.width - paddingLeft - paddingRight;
  const plotHeight = chart.height - paddingTop - paddingBottom;
  const barWidth = plotWidth / values.length;

  ctx.strokeStyle = "rgba(148, 163, 184, 0.25)";
  ctx.lineWidth = 1;
  ctx.font = "11px 'Palatino Linotype', serif";
  ctx.fillStyle = "#94a3b8";
  for (let i = 0; i <= 4; i += 1) {
    const y = paddingTop + (plotHeight / 4) * i;
    ctx.beginPath();
    ctx.moveTo(paddingLeft, y);
    ctx.lineTo(chart.width - paddingRight, y);
    ctx.stroke();

    const value = maxVal - (maxVal / 4) * i;
    ctx.textAlign = "right";
    ctx.fillText(value.toFixed(0), paddingLeft - 8, y + 4);
  }

  ctx.strokeStyle = "rgba(148, 163, 184, 0.5)";
  ctx.beginPath();
  ctx.moveTo(paddingLeft, paddingTop);
  ctx.lineTo(paddingLeft, chart.height - paddingBottom);
  ctx.lineTo(chart.width - paddingRight, chart.height - paddingBottom);
  ctx.stroke();

  values.forEach((value, i) => {
    const barHeight = (plotHeight * value) / maxVal;
    const x = paddingLeft + i * barWidth + barWidth * 0.2;
    const y = chart.height - paddingBottom - barHeight;
    const width = barWidth * 0.6;

    const gradient = ctx.createLinearGradient(0, y, 0, chart.height - paddingBottom);
    gradient.addColorStop(0, "#ffb44c");
    gradient.addColorStop(1, "#ff7f51");
    ctx.fillStyle = gradient;
    ctx.fillRect(x, y, width, barHeight);

    ctx.font = "11px 'Palatino Linotype', serif";
    ctx.textAlign = "center";
    ctx.fillStyle = "#f8fafc";
    ctx.fillText(value.toFixed(0), x + width / 2, y - 6);

    const label = labels[i];
    const maxLen = 18;
    const shortLabel = label.length > maxLen ? `${label.slice(0, maxLen - 3)}...` : label;

    ctx.textAlign = "center";
    ctx.fillStyle = "#e5e7eb";
    ctx.fillText(shortLabel, x + width / 2, chart.height - paddingBottom + 16);
  });
}

async function refreshChart() {
  try {
    const res = await fetch(`/${datasetPath}?t=${Date.now()}`);
    if (!res.ok) return;
    const text = await res.text();
    const rows = parseCsv(text);
    drawBarChart(rows);
  } catch (err) {
    console.error(err);
  }
}

async function pollStatus() {
  try {
    const res = await fetch(`/status?t=${Date.now()}`);
    if (!res.ok) return;
    const status = await res.json();

    if (status.running) {
      const eta = formatEta(status.eta_seconds);
      const elapsed = formatElapsed(status.elapsed_seconds);
      const progress = `${status.current}/${status.total}`;
      const percent = status.total > 0 ? Math.round((status.current / status.total) * 100) : 0;
      setOverlay(true, `${status.message} • ${progress} • ETA ${eta}`);
      setStatus(runStatus, `Running: ${status.message}`);
      overlayProgressBar.style.width = `${percent}%`;
      overlayProgressText.textContent = progress;
      overlayEta.textContent = `ETA ${eta}`;
      overlayElapsed.textContent = `Elapsed ${elapsed}`;
      overlayScenario.textContent = status.scenario || "-";
      overlayRun.textContent = status.run_id ? `${status.run_id}` : "-";
    } else {
      if (statusTimer) {
        clearInterval(statusTimer);
        statusTimer = null;
      }
      if (status.message && status.message.startsWith("error")) {
        setStatus(runStatus, status.message);
      } else if (status.message === "complete") {
        setStatus(runStatus, "Simulation complete.");
        lastRun.textContent = new Date().toLocaleTimeString();
        refreshChart();
      }
      setOverlay(false);
    }
  } catch (err) {
    console.error(err);
  }
}

async function runSimulation(mode) {
  setStatus(runStatus, "Starting simulation...");
  setOverlay(true, "Starting simulation...");
  try {
    const selectedScenarios = getCheckedValues(scenarioChecklist);
    const runs = Number(document.getElementById("runs").value || 0);
    const seed = Number(document.getElementById("seed").value || 102);

    if (mode === 'custom' && selectedScenarios.length === 0) {
      throw new Error('Select at least one scenario for custom mode.');
    }

    const payload = {
      mode,
      scenarios: selectedScenarios.join(','),
      runs,
      seed,
    };
    const result = await postJson("/run", payload);
    setStatus(runStatus, `Simulation started (${result.runs} runs)`);

    if (!statusTimer) {
      statusTimer = setInterval(pollStatus, 1000);
    }
  } catch (err) {
    setStatus(runStatus, `Error: ${err.message}`);
    setOverlay(false);
  }
}

function renderInsights(items) {
  if (!insightsBox) return;
  insightsBox.innerHTML = "";
  if (!items || items.length === 0) {
    insightsBox.textContent = "No insights available yet.";
    return;
  }
  items.forEach((text) => {
    const div = document.createElement("div");
    div.className = "insight-item";
    div.textContent = text;
    insightsBox.appendChild(div);
  });
}

async function runAnalysis() {
  setStatus(analysisStatus, "Running regression...");
  try {
    const target = document.getElementById("target").value.trim();
    const selectedFeatures = getCheckedValues(featureChecklist);
    if (selectedFeatures.length === 0) {
      throw new Error('Select at least one feature.');
    }
    const features = selectedFeatures.join(',');
    const robust = document.getElementById("robust").value.trim();
    const noPlots = document.getElementById("noPlots").checked;

    const ridgeAlphaRaw = ridgeAlphaInput ? ridgeAlphaInput.value.trim() : "";
    const ridgeAlpha = ridgeAlphaRaw.length ? Number(ridgeAlphaRaw) : null;
    const ridgeAuto = ridgeAutoInput ? ridgeAutoInput.checked : false;
    const boxCoxCheck = boxCoxCheckInput ? boxCoxCheckInput.checked : false;

    const payload = {
      input_path: datasetPath,
      outdir: "outputs",
      target,
      features,
      scenario_dummies: true,
      robust_se: robust || null,
      ridge_alpha: Number.isFinite(ridgeAlpha) ? ridgeAlpha : null,
      ridge_auto: ridgeAuto,
      boxcox_check: boxCoxCheck,
      no_plots: noPlots,
    };

    const result = await postJson("/analyze", payload);
    summary.textContent = result.summary;

    const insightsResult = await postJson("/insights", payload);
    if (insightsResult && insightsResult.insights) {
      renderInsights(insightsResult.insights.insights);
    }

    plots.innerHTML = "";
    if (!noPlots) {
      const descriptions = {
        "residuals_vs_fitted.png": "Residuals vs Fitted: checks linearity and equal error spread. Random scatter is good.",
        "qq_plot.png": "Q-Q Plot: checks if residuals follow a normal distribution.",
        "predicted_vs_actual.png": "Predicted vs Actual: compares model predictions to observed values.",
        "influence_plot.png": "Influence: leverage vs Cook's distance to spot influential points.",
      };
      const images = result.outputs.filter((p) => p.endsWith(".png"));
      images.forEach((src) => {
        const fileName = src.split(/[\\/]/).pop();
        const img = document.createElement("img");
        img.src = `/${src.replace(/\\/g, "/")}?t=${Date.now()}`;
        img.alt = fileName || "diagnostic plot";

        const card = document.createElement("div");
        card.className = "plot-card";
        card.appendChild(img);

        if (fileName && descriptions[fileName]) {
          const caption = document.createElement("p");
          caption.className = "plot-caption";
          caption.textContent = descriptions[fileName];
          card.appendChild(caption);
        }
        plots.appendChild(card);
      });
    }

    setStatus(analysisStatus, "Analysis complete.");
  } catch (err) {
    setStatus(analysisStatus, `Error: ${err.message}`);
  }
}

document.getElementById("runQuick").addEventListener("click", () => runSimulation("quick"));
document.getElementById("runFull").addEventListener("click", () => runSimulation("full"));
document.getElementById("runCustom").addEventListener("click", () => {
  const mode = document.getElementById("mode").value;
  runSimulation(mode);
});
document.getElementById("runAnalysis").addEventListener("click", runAnalysis);

populateOptions();
refreshChart();
window.addEventListener("resize", refreshChart);

