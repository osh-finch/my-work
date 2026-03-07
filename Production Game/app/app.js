const STORAGE_KEY = "production-game-cockpit-v1";

const PRICES = {
  ruler: 300,
  stencil5: 400,
  stencil10: 500,
  pen: 300,
  pencil: 100,
  rubber: 300,
  envelope: 10,
  white: 10,
  salmon: 20,
  pink: 20,
  blue: 20,
  green: 20,
  yellow: 20,
};
const ADMIN_LOT_SIZES = {
  ruler: 1,
  stencil5: 1,
  stencil10: 1,
  pen: 1,
  pencil: 1,
  rubber: 1,
  envelope: 10,
  white: 10,
  salmon: 10,
  pink: 10,
  blue: 10,
  green: 10,
  yellow: 10,
};

const CARDS_PER_SHEET = { A5: 2, A6: 4, A7: 8 };
const COLOUR_CHOICES = ["White", "Salmon", "Pink", "Blue", "Green", "Yellow"];
const SIZE_CHOICES = ["A5", "A6", "A7"];
const STARTER_STOCK = {
  ruler: 1, stencil5: 1, stencil10: 1, pen: 1, pencil: 1, rubber: 1,
  envelope: 10, white: 10, salmon: 10, pink: 10, blue: 10, green: 10, yellow: 10,
};
const APP_CONFIG = window.APP_CONFIG || {};
const DEFAULT_LLM = {
  apiKey: APP_CONFIG.OPENAI_API_KEY || "",
  model: APP_CONFIG.OPENAI_MODEL || "gpt-4.1-mini",
  endpoint: APP_CONFIG.OPENAI_ENDPOINT || "https://api.openai.com/v1/responses",
};

const state = loadState();
let timerIntervalId = null;
wireUI();
renderAll();

function loadState() {
  const raw = localStorage.getItem(STORAGE_KEY);
  const fallback = {
    settings: { gameMinute: 0, operatorCount: 6, parsedOrdersView: "hidden" },
    orders: [],
    procurementLog: [],
    stock: { ...STARTER_STOCK },
    ledger: [{ ts: nowTs(), type: "starter_pack", amount: -3100, note: "Starter pack issued" }],
    serial: 1,
    llm: { ...DEFAULT_LLM },
    photoParseLast: [],
    timer: { totalSeconds: 120 * 60, remainingSeconds: 120 * 60, running: false, lastTickMs: null },
    dashboardHistory: [],
  };
  if (!raw) return fallback;
  try {
    const parsed = JSON.parse(raw);
    return {
      ...fallback,
      ...parsed,
      settings: { ...fallback.settings, ...(parsed.settings || {}) },
      stock: { ...fallback.stock, ...(parsed.stock || {}) },
      llm: { ...DEFAULT_LLM, ...(parsed.llm || {}) },
      photoParseLast: Array.isArray(parsed.photoParseLast) ? parsed.photoParseLast : [],
      timer: { ...fallback.timer, ...(parsed.timer || {}) },
      dashboardHistory: Array.isArray(parsed.dashboardHistory) ? parsed.dashboardHistory : [],
    };
  } catch {
    return fallback;
  }
}

function saveState() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

function nowTs() {
  return new Date().toISOString();
}

function nextId() {
  return state.serial++;
}

function wireUI() {
  hydrateLiveTimestamps();
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      document.getElementById(btn.dataset.tab).classList.add("active");
      renderAll();
    });
  });

  document.getElementById("gameMinute").value = state.settings.gameMinute;
  document.getElementById("operatorCount").value = state.settings.operatorCount;
  document.getElementById("timerTotalMinutes").value = Math.max(1, Math.round(state.timer.totalSeconds / 60));
  document.getElementById("parsedOrdersView").value = state.settings.parsedOrdersView || "hidden";

  document.getElementById("saveSettings").addEventListener("click", () => {
    state.settings.gameMinute = Number(document.getElementById("gameMinute").value || 0);
    state.settings.operatorCount = Number(document.getElementById("operatorCount").value || 1);
    const totalMinutes = Math.max(1, Number(document.getElementById("timerTotalMinutes").value || 120));
    const nextTotalSeconds = Math.round(totalMinutes * 60);
    if (!state.timer.running) {
      state.timer.totalSeconds = nextTotalSeconds;
      state.timer.remainingSeconds = nextTotalSeconds;
      state.timer.lastTickMs = null;
    } else {
      state.timer.totalSeconds = nextTotalSeconds;
    }
    saveState();
    renderAll();
  });
  document.getElementById("parsedOrdersView").addEventListener("change", (e) => {
    state.settings.parsedOrdersView = String(e.target.value || "hidden");
    saveState();
    renderPhotoParseResults();
  });
  document.getElementById("llmSettingsForm").addEventListener("submit", (e) => {
    e.preventDefault();
    state.llm.apiKey = String(document.getElementById("llmApiKey").value || "").trim();
    state.llm.model = String(document.getElementById("llmModel").value || DEFAULT_LLM.model).trim();
    state.llm.endpoint = String(document.getElementById("llmEndpoint").value || DEFAULT_LLM.endpoint).trim();
    saveState();
    alert("Settings saved.");
    renderAll();
  });

  document.getElementById("resetData").addEventListener("click", () => {
    if (!confirm("Reset all cockpit data?")) return;
    localStorage.removeItem(STORAGE_KEY);
    location.reload();
  });
  document.getElementById("timerStart").addEventListener("click", startTimer);
  document.getElementById("timerPause").addEventListener("click", pauseTimer);
  document.getElementById("timerReset").addEventListener("click", resetTimer);

  document.getElementById("orderForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const form = e.target;
    const data = new FormData(form);
    const order = {
      rowId: nextId(),
      id: String(data.get("id")).trim(),
      occasion: String(data.get("occasion")).trim(),
      qty: Number(data.get("qty")),
      colour: String(data.get("colour")),
      size: String(data.get("size")),
      lines: Number(data.get("lines")),
      price: Number(data.get("price")),
      leadTime: Number(data.get("leadTime") || 0),
      openDelivery: !!data.get("openDelivery"),
      createdMinute: state.settings.gameMinute,
      status: "board",
      notes: "",
    };
    if (!order.id) return;
    state.orders.push(order);
    saveState();
    form.reset();
    renderAll();
  });

  document.getElementById("receiveProcurement").addEventListener("click", receiveDueProcurement);
  document.getElementById("photoParseForm").addEventListener("submit", onPhotoParseSubmit);

  document.getElementById("adjustmentForm").addEventListener("submit", (e) => {
    e.preventDefault();
    const data = new FormData(e.target);
    const type = String(data.get("type"));
    const amount = Number(data.get("amount"));
    const note = String(data.get("note"));
    const sign = type === "penalty" || type === "intercompany_out" ? -1 : 1;
    state.ledger.push({ ts: nowTs(), type, amount: sign * amount, note });
    saveState();
    e.target.reset();
    renderAll();
  });

  document.getElementById("llmApiKey").value = state.llm.apiKey || DEFAULT_LLM.apiKey || "";
  document.getElementById("llmModel").value = state.llm.model || DEFAULT_LLM.model;
  document.getElementById("llmEndpoint").value = state.llm.endpoint || DEFAULT_LLM.endpoint;

  buildProcurementForm();
  ensureTimerLoop();
  renderTimer();
  window.addEventListener("resize", () => {
    const dashboardActive = document.getElementById("dashboard")?.classList.contains("active");
    if (dashboardActive) renderDashboard();
  });
}

function hydrateLiveTimestamps() {
  let changed = false;
  const nowMs = Date.now();
  state.orders.forEach((o) => {
    const liveStatus = o.status === "accepted" || o.status === "in_progress";
    if (!liveStatus) return;
    if (o.openDelivery) return;
    if (o.takenTs) return;
    if (Number.isFinite(o.takenMinute)) {
      const elapsedMin = Math.max(0, state.settings.gameMinute - o.takenMinute);
      o.takenTs = new Date(nowMs - elapsedMin * 60 * 1000).toISOString();
      changed = true;
      return;
    }
    o.takenMinute = state.settings.gameMinute;
    o.takenTs = new Date(nowMs).toISOString();
    changed = true;
  });
  if (changed) saveState();
}

function baseEffectiveMinutes(order, operatorCount = state.settings.operatorCount) {
  const sheetNeed = Math.ceil(order.qty / CARDS_PER_SHEET[order.size]);
  const materialCost = sheetNeed * PRICES[order.colour.toLowerCase()] + PRICES.envelope;

  const cutFold = { A5: 0.55, A6: 0.45, A7: 0.4 }[order.size] * order.qty;
  const front = { A5: 0.6, A6: 0.48, A7: 0.4 }[order.size] * order.qty;
  const verse = (0.12 * order.lines + 0.15) * order.qty;
  const back = 0.18 * order.qty;
  const packDeliver = 2.8;
  const rawMinutes = cutFold + front + verse + back + packDeliver;
  const effectiveMinutes = rawMinutes / Math.max(operatorCount * 0.82, 1);
  return { sheetNeed, materialCost, rawMinutes, effectiveMinutes };
}

function getAdaptiveModel() {
  const completed = state.orders.filter((o) =>
    (o.status === "delivered" || o.status === "rejected") &&
    Number(o.actualDurationMin) > 0 &&
    Number(o.predictedMinutesAtStart) > 0
  );

  if (completed.length < 2) {
    return { sampleCount: completed.length, globalFactor: 1, sizeFactor: { A5: 1, A6: 1, A7: 1 }, confidence: "Low" };
  }

  const ratios = completed
    .map((o) => Number(o.actualDurationMin) / Number(o.predictedMinutesAtStart))
    .filter((x) => Number.isFinite(x) && x > 0);
  if (ratios.length === 0) {
    return { sampleCount: 0, globalFactor: 1, sizeFactor: { A5: 1, A6: 1, A7: 1 }, confidence: "Low" };
  }

  const sorted = ratios.slice().sort((a, b) => a - b);
  const trim = Math.floor(sorted.length * 0.15);
  const trimmed = sorted.slice(trim, sorted.length - trim || sorted.length);
  const globalFactor = clamp(mean(trimmed), 0.5, 2.0);

  const sizeFactor = { A5: 1, A6: 1, A7: 1 };
  for (const size of SIZE_CHOICES) {
    const rows = completed
      .filter((o) => o.size === size)
      .map((o) => Number(o.actualDurationMin) / Number(o.predictedMinutesAtStart))
      .filter((x) => Number.isFinite(x) && x > 0);
    if (rows.length >= 2) {
      sizeFactor[size] = clamp(mean(rows) / globalFactor, 0.75, 1.35);
    }
  }

  const confidence = completed.length >= 10 ? "High" : completed.length >= 5 ? "Medium" : "Low";
  return { sampleCount: completed.length, globalFactor, sizeFactor, confidence };
}

function estimateOrder(order, opts = {}) {
  const qty = Math.max(1, Number(order.qty || 0));
  const lines = Math.max(1, Number(order.lines || 0));
  const workers = Math.max(1, Number(state.settings.operatorCount || 1));

  const paperCostPerCard = String(order.colour || "").toLowerCase() === "white" ? 10 : 20;
  const cogs = qty * paperCostPerCard + 10; // includes one envelope at £10
  const profit = Number(order.price || 0) - cogs;

  const timeBySize = {
    A5: { fold: 0.5, stencil: 1.5, writePerLine: 0.5, id: 0.25, qc: 0.5 },
    A6: { fold: 0.75, stencil: 1.5, writePerLine: 0.5, id: 0.25, qc: 0.5 },
    A7: { fold: 1.0, stencil: 1.0, writePerLine: 0.4, id: 0.25, qc: 0.5 },
  };
  const t = timeBySize[order.size] || timeBySize.A6;
  const timePerCard = t.fold + t.stencil + (lines * t.writePerLine) + t.id + t.qc;
  const verseComposeTime = lines <= 4 ? 2 : 3;
  const fixedSetup = 1 + verseComposeTime + 2; // select order + compose + SODN/pack/deliver
  const totalPersonMins = fixedSetup + (qty * timePerCard);
  const wallClockMins = totalPersonMins / workers;
  const effectiveMinutes = wallClockMins;

  const includeQueue = opts.includeQueue !== false;
  const queueMinutes = includeQueue ? queuedMinutes(order.rowId) : 0;
  const estFinishMinute = state.settings.gameMinute + queueMinutes + wallClockMins;

  let dueMinute = Infinity;
  if (!order.openDelivery) {
    const referenceMinute = Number.isFinite(order.takenMinute) ? order.takenMinute : state.settings.gameMinute;
    dueMinute = referenceMinute + Number(order.leadTime || 0);
  }

  const feasibleByOrder = order.openDelivery || wallClockMins <= Number(order.leadTime || 0);
  const feasible = order.openDelivery || estFinishMinute <= dueMinute;

  let riskMultiplier = 1.2;
  if (!order.openDelivery) {
    const lead = Math.max(0, Number(order.leadTime || 0));
    if (wallClockMins <= 0.7 * lead) riskMultiplier = 1.0;
    else if (wallClockMins <= lead) riskMultiplier = 0.7;
    else riskMultiplier = 0.3;
  }

  const score = (profit / Math.max(totalPersonMins, 0.1)) * riskMultiplier;
  return {
    sheetNeed: Math.ceil(qty / CARDS_PER_SHEET[order.size]),
    materialCost: cogs,
    cogs,
    adaptiveFactor: 1,
    baseEffectiveMinutes: wallClockMins,
    effectiveMinutes,
    estFinishMinute,
    dueMinute,
    pRejectOrLate: 0,
    expectedPenalty: 0,
    expectedProfit: profit,
    profit,
    paperCostPerCard,
    timePerCard,
    totalPersonMins,
    wallClockMins,
    fixedSetup,
    verseComposeTime,
    riskMultiplier,
    formulaFeasible: feasibleByOrder,
    score,
    feasible,
    confidence: "Formula",
    sampleCount: 0,
  };
}

function paperListPrice(order) {
  return String(order.colour || "").toLowerCase() === "white" ? 10 : 20;
}

function computeRiskFactor(order, totalPersonMins, delayMins = 0) {
  if (order.openDelivery) return { riskFactor: 1.2, slackRatio: Infinity, feasibleTimed: true };
  const lead = Math.max(0, Number(order.leadTime || 0) - delayMins);
  const workers = Math.max(1, Number(state.settings.operatorCount || 1));
  const slackRatio = (lead * workers) / Math.max(totalPersonMins, 0.1);
  const riskFactor = clamp(Math.min(slackRatio, 1.5) / 1.5, 0, 1);
  return { riskFactor, slackRatio, feasibleTimed: lead > 0 && slackRatio > 0 };
}

function computeAdjustedScenario(order, actualPaperCost, opts = {}) {
  const qty = Math.max(1, Number(order.qty || 0));
  const lines = Math.max(1, Number(order.lines || 0));
  const size = order.size;
  const t = {
    A5: { fold: 0.5, stencil: 1.5, writePerLine: 0.5, id: 0.25, qc: 0.5 },
    A6: { fold: 0.75, stencil: 1.5, writePerLine: 0.5, id: 0.25, qc: 0.5 },
    A7: { fold: 1.0, stencil: 1.0, writePerLine: 0.4, id: 0.25, qc: 0.5 },
  }[size] || { fold: 0.75, stencil: 1.5, writePerLine: 0.5, id: 0.25, qc: 0.5 };
  const timePerCard = t.fold + t.stencil + (lines * t.writePerLine) + t.id + t.qc;
  const verseComposeTime = lines <= 4 ? 2 : 3;
  const fixedSetup = 1 + verseComposeTime + 2;
  const totalPersonMins = fixedSetup + (qty * timePerCard);

  const delayMins = Number(opts.delayMins || 0);
  const resourceFactor = Number(opts.resourceFactor ?? 1.0);
  const envelopeCost = 10;
  const netProfit = Number(order.price || 0) - (qty * actualPaperCost) - envelopeCost;
  const { riskFactor, slackRatio, feasibleTimed } = computeRiskFactor(order, totalPersonMins, delayMins);
  const adjustedScore = (netProfit / Math.max(totalPersonMins, 0.1)) * riskFactor * resourceFactor;
  const feasible = order.openDelivery || feasibleTimed;
  return {
    netProfit,
    totalPersonMins,
    timePerCard,
    fixedSetup,
    verseComposeTime,
    riskFactor,
    slackRatio,
    resourceFactor,
    delayMins,
    adjustedScore,
    feasible,
  };
}

function getHurdleRate() {
  const minute = Number(state.settings.gameMinute || 0);
  const phase = minute < 40 ? "early" : minute < 80 ? "mid" : "late";
  if (phase === "late") return 0.01;

  const sampleScores = state.orders
    .filter((o) => o.status === "board")
    .map((o) => {
      const est = estimateOrder(o);
      const deficit = materialDeficitForOrder(o, est);
      if (deficit.totalUnitsMissing > 0) return null;
      const base = computeAdjustedScenario(o, paperListPrice(o), { delayMins: 0, resourceFactor: 1.0 });
      if (!base.feasible || base.netProfit <= 0) return null;
      return base.adjustedScore;
    })
    .filter((x) => Number.isFinite(x));

  if (!sampleScores.length) return 0.01;
  const sorted = sampleScores.slice().sort((a, b) => a - b);
  const idx = phase === "early"
    ? Math.max(0, Math.floor(0.75 * (sorted.length - 1)))
    : Math.max(0, Math.floor(0.5 * (sorted.length - 1)));
  return Math.max(0.01, sorted[idx]);
}

function queuedMinutes(excludeRowId = null) {
  return state.orders
    .filter((o) => (o.status === "accepted" || o.status === "in_progress") && o.rowId !== excludeRowId)
    .map((o) => estimateOrder(o, { includeQueue: false }).effectiveMinutes)
    .reduce((a, b) => a + b, 0);
}

function mean(arr) {
  if (!arr.length) return 1;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function actionAccept(rowId) {
  const order = state.orders.find((o) => o.rowId === rowId);
  if (!order) return;
  if (!Number.isFinite(order.takenMinute)) {
    order.takenMinute = state.settings.gameMinute;
    order.takenTs = nowTs();
  }
  order.status = "accepted";
  saveState();
  renderAll();
}

function actionStart(rowId) {
  const order = state.orders.find((o) => o.rowId === rowId);
  if (!order) return;
  if (!allocateMaterialsForOrder(order)) return;
  if (!Number.isFinite(order.takenMinute)) {
    order.takenMinute = state.settings.gameMinute;
    order.takenTs = nowTs();
  }
  if (order.status === "board" || order.status === "accepted") {
    const startEstimate = estimateOrder(order, { includeQueue: false });
    order.predictedMinutesAtStart = Number(startEstimate.effectiveMinutes.toFixed(2));
    order.startedMinute = state.settings.gameMinute;
    order.startedTs = nowTs();
  }
  order.status = "in_progress";
  saveState();
  renderAll();
}

function consumeMaterials(order) {
  const est = estimateOrder(order);
  const colourKey = order.colour.toLowerCase();
  state.stock[colourKey] = Math.max((state.stock[colourKey] || 0) - est.sheetNeed, 0);
  state.stock.envelope = Math.max((state.stock.envelope || 0) - 1, 0);
}

function actionDeliver(rowId) {
  const order = state.orders.find((o) => o.rowId === rowId);
  if (!order) return;
  if (order.status !== "accepted" && order.status !== "in_progress") return;
  completeOrderTracking(order);
  order.status = "delivered";
  state.ledger.push({ ts: nowTs(), type: "sale", amount: order.price, note: `Order ${order.id} delivered` });
  saveState();
  renderAll();
}

function actionReject(rowId) {
  const order = state.orders.find((o) => o.rowId === rowId);
  if (!order) return;
  if (order.status !== "accepted" && order.status !== "in_progress") return;
  completeOrderTracking(order);
  order.status = "rejected";
  const penalty = -0.2 * order.price;
  state.ledger.push({ ts: nowTs(), type: "penalty", amount: penalty, note: `Order ${order.id} rejected/missed` });
  saveState();
  renderAll();
}

function completeOrderTracking(order) {
  order.completedMinute = state.settings.gameMinute;
  order.completedTs = nowTs();
  let actualDurationMin = null;
  if (Number.isFinite(order.startedMinute)) {
    actualDurationMin = order.completedMinute - order.startedMinute;
  }
  if (!Number.isFinite(actualDurationMin) || actualDurationMin <= 0) {
    const entered = prompt(
      `Enter actual production minutes for order ${order.id} (used for learning predictions):`,
      order.predictedMinutesAtStart ? String(Math.round(order.predictedMinutesAtStart)) : ""
    );
    if (entered !== null && entered.trim() !== "") {
      const parsed = Number(entered);
      if (Number.isFinite(parsed) && parsed > 0) actualDurationMin = parsed;
    }
  }
  if (Number.isFinite(actualDurationMin) && actualDurationMin > 0) {
    order.actualDurationMin = Number(actualDurationMin.toFixed(2));
  }
}

function actionRemove(rowId) {
  const order = state.orders.find((o) => o.rowId === rowId);
  if (order && (order.status === "board" || order.status === "accepted" || order.status === "in_progress")) {
    restoreAllocatedMaterials(order);
  }
  state.orders = state.orders.filter((o) => o.rowId !== rowId);
  saveState();
  renderAll();
}

function createPurchaseOrderForOrder(rowId, source = "team") {
  const order = state.orders.find((o) => o.rowId === rowId);
  if (!order) return;
  const est = estimateOrder(order, { includeQueue: false });
  const deficit = materialDeficitForOrder(order, est);
  if (deficit.totalUnitsMissing <= 0) {
    alert(`No missing stock for order ${order.id}.`);
    return;
  }

  const form = document.getElementById("procurementForm");
  if (!form) return;

  const sourcePartyEl = document.getElementById("procurementSourceParty");
  if (sourcePartyEl) sourcePartyEl.value = source === "admin" ? "Controllers" : "Team A";
  const placedEl = document.getElementById("procurementPlacedTime");
  if (placedEl) placedEl.value = formatClock(Date.now());

  // Reset all quantities, then fill missing requirements for this SO.
  form.querySelectorAll("input[name]").forEach((input) => {
    if (input.name === "procurementPlacedTime") return;
    input.value = "0";
  });
  missingItemsList(deficit).forEach((m) => {
    const input = form.querySelector(`input[name="${m.key}"]`);
    if (input) input.value = String(m.qty);
  });

  // Bring user to procurement tab to review and submit manually.
  document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
  document.querySelectorAll(".tab-panel").forEach((p) => p.classList.remove("active"));
  const targetTab = document.querySelector('.tab[data-tab="procurement"]');
  const targetPanel = document.getElementById("procurement");
  if (targetTab) targetTab.classList.add("active");
  if (targetPanel) targetPanel.classList.add("active");

  updateProcurementDraftSummary();
  alert(`PO form prefilled for order ${order.id}. Review and click Place Order to confirm.`);
}

function allocateMaterialsForOrder(order) {
  if (order.materialsAllocated) return true;
  const est = estimateOrder(order, { includeQueue: false });
  const colourKey = order.colour.toLowerCase();
  const haveSheets = Number(state.stock[colourKey] || 0);
  const haveEnvelopes = Number(state.stock.envelope || 0);
  if (haveSheets < est.sheetNeed || haveEnvelopes < 1) {
    const missingSheets = Math.max(0, est.sheetNeed - haveSheets);
    const missingEnvelope = Math.max(0, 1 - haveEnvelopes);
    const qty = Math.max(1, Number(order.qty || 1));
    const personMins = Number(est.totalPersonMins || 1);
    const hurdleRate = getHurdleRate();
    const maxMarkupPrice = (Number(order.price || 0) - hurdleRate * personMins - 10) / qty;
    const maxMarkupText = maxMarkupPrice > 0
      ? `Max team markup to pay: £${maxMarkupPrice.toFixed(2)} per sheet`
      : "No positive team markup is justified (skip unless near list price).";
    const deficit = { colourKey, missingSheets, missingEnvelope };
    const missingText = missingItemsList(deficit).map((x) => `${x.qty} ${x.key}(s)`).join(", ");
    alert(
      `Insufficient stock for ${order.id}.\n` +
      `Missing: ${missingText || "none"}.\n` +
      `${maxMarkupText}`
    );
    return false;
  }
  state.stock[colourKey] = haveSheets - est.sheetNeed;
  state.stock.envelope = haveEnvelopes - 1;
  order.materialsAllocated = true;
  order.allocatedMaterials = { colourKey, sheetNeed: est.sheetNeed, envelope: 1 };
  return true;
}

function restoreAllocatedMaterials(order) {
  if (!order.materialsAllocated || !order.allocatedMaterials) return;
  const { colourKey, sheetNeed, envelope } = order.allocatedMaterials;
  state.stock[colourKey] = Number(state.stock[colourKey] || 0) + Number(sheetNeed || 0);
  state.stock.envelope = Number(state.stock.envelope || 0) + Number(envelope || 0);
  order.materialsAllocated = false;
}

function buildProcurementForm() {
  const wrap = document.getElementById("procurementForm");
  const fields = ["ruler", "stencil5", "stencil10", "pen", "pencil", "rubber", "envelope", "white", "salmon", "pink", "blue", "green", "yellow"];
  wrap.innerHTML = `
    <label class="wide">Source
      <select id="procurementSourceParty" name="procurementSourceParty">
        <option value="Controllers">Controllers</option>
        <option value="Team A">Team A</option>
        <option value="Team C">Team C</option>
        <option value="Team D">Team D</option>
      </select>
    </label>
    <label>Placed Time (actual)
      <input id="procurementPlacedTime" name="procurementPlacedTime" type="text" value="${formatClock(Date.now())}" readonly />
    </label>
  ` + fields
    .map((k) => `<label>${k} (£${PRICES[k]}, admin lot ${ADMIN_LOT_SIZES[k]}) <input type="number" min="0" value="0" name="${k}" /></label>`)
    .join("") + "<button type='submit'>Place Order</button>";
  wrap.querySelectorAll("input").forEach((input) => input.addEventListener("input", updateProcurementDraftSummary));
  const sourceEl = wrap.querySelector("#procurementSourceParty");
  if (sourceEl) sourceEl.addEventListener("change", updateProcurementDraftSummary);
  updateProcurementDraftSummary();

  wrap.addEventListener("submit", (e) => {
    e.preventDefault();
    const data = new FormData(e.target);
    const sourceParty = String(data.get("procurementSourceParty") || "Controllers");
    const source = sourceParty === "Controllers" ? "admin" : "team";
    const counterparty = sourceParty;
    const nowMs = Date.now();
    const placedMinute = Number(state.settings.gameMinute || 0);
    const items = {};
    let total = 0;
    for (const [k, v] of data.entries()) {
      if (k === "procurementSourceParty" || k === "procurementPlacedTime") continue;
      const q = Number(v || 0);
      if (q > 0) {
        items[k] = q;
        total += q * PRICES[k];
      }
    }

    if (total === 0) return;
    if (source === "admin") {
      const badLots = Object.entries(items).filter(([k, q]) => q % (ADMIN_LOT_SIZES[k] || 1) !== 0);
      if (badLots.length) {
        const msg = badLots
          .map(([k, q]) => `${k}: ${q} (must be multiples of ${ADMIN_LOT_SIZES[k] || 1})`)
          .join("\n");
        alert(`Admin orders must follow designated quantities:\n${msg}`);
        return;
      }
    }
    if (source === "admin" && total > 1000) {
      alert("Procurement order exceeds £1000 limit.");
      return;
    }

    const minute = state.settings.gameMinute;
    const lastAdmin = state.procurementLog.filter((p) => p.type === "order" && (p.source || "admin") === "admin").slice(-1)[0];
    if (source === "admin" && lastAdmin && Number.isFinite(lastAdmin.placedAtMs) && nowMs - lastAdmin.placedAtMs < 5 * 60 * 1000) {
      const remaining = (5 * 60 * 1000) - (nowMs - lastAdmin.placedAtMs);
      alert(`Must leave 5 minutes between controller orders. Cooldown remaining: ${formatMsCountdown(remaining)}.`);
      return;
    }

    if (source === "team") {
      // Team purchases are logged per item (not as a single batch order).
      Object.entries(items).forEach(([itemKey, qty]) => {
        const lineTotal = qty * PRICES[itemKey];
        const dueMinute = placedMinute;
        const receivedNow = dueMinute <= minute;
        state.procurementLog.push({
          rowId: nextId(),
          type: "order",
          source,
          counterparty,
          items: { [itemKey]: qty },
          total: lineTotal,
          placedMinute,
          placedAtMs: nowMs,
          dueMinute,
          dueAtMs: nowMs,
          received: receivedNow,
        });
        if (receivedNow) {
          state.stock[itemKey] = (state.stock[itemKey] || 0) + qty;
        }
        state.ledger.push({ ts: nowTs(), type: "purchase", amount: -lineTotal, note: `Team purchase ${itemKey} x${qty}` });
      });
    } else {
      const dueMinute = placedMinute + 10;
      const dueAtMs = nowMs + 10 * 60 * 1000;
      const receivedNow = false;
      state.procurementLog.push({
        rowId: nextId(),
        type: "order",
        source,
        counterparty,
        items,
        total,
        placedMinute,
        placedAtMs: nowMs,
        dueMinute,
        dueAtMs,
        received: receivedNow,
        notifiedReady: false,
      });
      if (receivedNow) {
        Object.entries(items).forEach(([k, q]) => {
          state.stock[k] = (state.stock[k] || 0) + q;
        });
      }
      state.ledger.push({ ts: nowTs(), type: "purchase", amount: -total, note: `Procurement order #${state.serial - 1}` });
    }
    saveState();
    e.target.reset();
    const placedInput = document.getElementById("procurementPlacedTime");
    if (placedInput) placedInput.value = formatClock(Date.now());
    updateProcurementDraftSummary();
    renderAll();
  });
}

function updateProcurementDraftSummary() {
  const form = document.getElementById("procurementForm");
  if (!form) return;
  const data = new FormData(form);
  const sourceParty = String(data.get("procurementSourceParty") || "Controllers");
  const source = sourceParty === "Controllers" ? "admin" : "team";
  const placedMinute = Number(state.settings.gameMinute || 0);
  const lineItems = [];
  let total = 0;
  for (const [k, v] of data.entries()) {
    if (k === "procurementSourceParty" || k === "procurementPlacedTime") continue;
    const q = Number(v || 0);
    if (q > 0) {
      const unit = PRICES[k];
      const line = q * unit;
      total += line;
      lineItems.push({ item: k, qty: q, unit, line });
    }
  }
  const cooldownText = controllerCooldownText();
  if (lineItems.length === 0) {
    document.getElementById("procurementSummary").innerHTML =
      source === "admin"
        ? `<p>No items selected yet.</p><p>Controllers rules: designated lot sizes required, max per order £1000, 10-minute lead. Draft due minute: ${placedMinute + 10}. ${cooldownText}</p>`
        : `<p>No items selected yet.</p><p>${sourceParty} rules: any quantity allowed, no max value, no lead time.</p>`;
    return;
  }
  const lotWarnings = source === "admin"
    ? lineItems
        .filter((x) => x.qty % (ADMIN_LOT_SIZES[x.item] || 1) !== 0)
        .map((x) => `${x.item} must be in multiples of ${ADMIN_LOT_SIZES[x.item] || 1}`)
    : [];
  document.getElementById("procurementSummary").innerHTML = `
    <table>
      <thead><tr><th>Item</th><th>Qty</th><th>Unit £</th><th>Line £</th></tr></thead>
      <tbody>${lineItems.map((x) => `<tr><td>${x.item}</td><td>${x.qty}</td><td>${x.unit}</td><td>${x.line}</td></tr>`).join("")}</tbody>
    </table>
    <p><strong>Draft goods cost:</strong> £${total.toFixed(2)}</p>
    ${lotWarnings.length ? `<p style="color: var(--bad);"><strong>Designated quantity warnings:</strong><br>${lotWarnings.join("<br>")}</p>` : ""}
    <p>${
      source === "admin"
        ? `Controllers rules: designated lot sizes required, max per order £1000, 10-minute lead. Draft due minute: ${placedMinute + 10}. ${cooldownText}`
        : `${sourceParty} rules: entries logged per item (not batched), any quantity, no max value, no lead time.`
    }</p>
  `;
}

function controllerCooldownText() {
  const last = state.procurementLog.filter((p) => p.type === "order" && (p.source || "admin") === "admin").slice(-1)[0];
  if (!last || !Number.isFinite(last.placedAtMs)) return "Controller cooldown: ready now.";
  const remaining = (last.placedAtMs + 5 * 60 * 1000) - Date.now();
  if (remaining <= 0) return "Controller cooldown: ready now.";
  return `Controller cooldown remaining: ${formatMsCountdown(remaining)}.`;
}

function controllerCooldownData() {
  const last = state.procurementLog.filter((p) => p.type === "order" && (p.source || "admin") === "admin").slice(-1)[0];
  if (!last || !Number.isFinite(last.placedAtMs)) {
    return { label: "Controllers cooldown: ready now", cls: "time-good", remainingMs: 0 };
  }
  const remainingMs = (last.placedAtMs + 5 * 60 * 1000) - Date.now();
  if (remainingMs <= 0) return { label: "Controllers cooldown: ready now", cls: "time-good", remainingMs: 0 };
  const totalMs = 5 * 60 * 1000;
  const ratio = remainingMs / totalMs;
  const cls = ratio > 0.66 ? "time-bad" : ratio > 0.33 ? "time-warn" : "time-good";
  return { label: `Controllers cooldown: ${formatMsCountdown(remainingMs)} remaining`, cls, remainingMs };
}

function renderControllerCooldownBanner() {
  const el = document.getElementById("controllerCooldownBanner");
  if (!el) return;
  const d = controllerCooldownData();
  el.innerHTML = `<span class="tag ${d.cls}">${d.label}</span>`;
}

function receiveDueProcurement() {
  const received = processProcurementArrivals();
  if (received === 0) alert("No due procurement deliveries yet.");
  renderAll();
}

function processProcurementArrivals() {
  const nowMs = Date.now();
  let received = 0;
  state.procurementLog.forEach((p) => {
    if (p.received) return;
    const dueMs = Number.isFinite(p.dueAtMs) ? p.dueAtMs : nowMs;
    if (nowMs >= dueMs) {
      Object.entries(p.items).forEach(([k, q]) => {
        state.stock[k] = (state.stock[k] || 0) + q;
      });
      p.received = true;
      received += 1;
      if (p.source === "admin" && !p.notifiedReady) {
        p.notifiedReady = true;
        alert(`Controller order #${p.rowId} is ready for collection.`);
      }
    }
  });
  if (received > 0) saveState();
  return received;
}

async function onPhotoParseSubmit(e) {
  e.preventDefault();
  const fileInput = document.getElementById("ordersPhoto");
  const file = fileInput.files && fileInput.files[0];
  if (!file) {
    setPhotoStatus("Choose an image first.", true);
    return;
  }

  const keyFromInput = String(document.getElementById("llmApiKey").value || "").trim();
  const fallbackKey = String(state.llm.apiKey || DEFAULT_LLM.apiKey || APP_CONFIG.OPENAI_API_KEY || "").trim();
  state.llm.apiKey = keyFromInput || fallbackKey;
  state.llm.model = String(document.getElementById("llmModel").value || state.llm.model || DEFAULT_LLM.model || APP_CONFIG.OPENAI_MODEL || "gpt-4.1-mini").trim();
  state.llm.endpoint = String(document.getElementById("llmEndpoint").value || state.llm.endpoint || DEFAULT_LLM.endpoint || APP_CONFIG.OPENAI_ENDPOINT || "https://api.openai.com/v1/responses").trim();
  saveState();
  if (!state.llm.apiKey) {
    setPhotoStatus("API key required to call the LLM endpoint.", true);
    return;
  }

  setPhotoStatus("Parsing image with LLM vision...", false);
  try {
    const parsed = await parseOrdersFromPhoto(file);
    if (parsed.length === 0) {
      setPhotoStatus("No valid orders found in image.", true);
      state.photoParseLast = [];
      renderPhotoParseResults();
      return;
    }

    const autoAdd = document.getElementById("autoAddParsed").checked;
    const resultRows = [];
    parsed.forEach((o, idx) => {
      const order = {
        rowId: nextId(),
        id: o.id || `IMG-${state.settings.gameMinute}-${idx + 1}`,
        occasion: o.occasion || "Unknown",
        qty: Number(o.quantity),
        colour: normalizeColour(o.colour),
        size: normalizeSize(o.size),
        lines: Math.max(1, Number(o.lines)),
        price: Number(o.price),
        leadTime: Number(o.lead_time_minutes || 0),
        openDelivery: !!o.open_delivery,
        createdMinute: state.settings.gameMinute,
        status: "board",
        notes: "Parsed from order-board photo",
      };
      if (!isValidParsedOrder(order)) return;
      const est = estimateOrder(order);
      const rec = !est.feasible ? "Skip" : est.score > 5 ? "Take Now" : est.score > 2 ? "Take If Idle" : "Skip";
      resultRows.push({
        id: order.id,
        occasion: order.occasion,
        qty: order.qty,
        colour: order.colour,
        size: order.size,
        price: order.price,
        lead: order.openDelivery ? "Open" : `${order.leadTime}m`,
        score: est.score,
        expectedProfit: est.expectedProfit,
        recommendation: rec,
      });
      if (autoAdd) state.orders.push(order);
    });

    state.photoParseLast = resultRows;
    saveState();
    renderAll();
    setPhotoStatus(
      autoAdd
        ? `Parsed ${resultRows.length} orders and added to board.`
        : `Parsed ${resultRows.length} orders. Review below (not added).`,
      false
    );
  } catch (err) {
    setPhotoStatus(`Parse failed: ${err.message}`, true);
  }
}

async function parseOrdersFromPhoto(file) {
  const imageDataUrl = await fileToDataUrl(file);
  const prompt = [
    "Extract all greeting-card orders visible in this image.",
    "Return strict JSON only with this shape:",
    "{\"orders\":[{\"id\":\"string\",\"occasion\":\"string\",\"quantity\":number,\"colour\":\"White|Salmon|Pink|Blue|Green|Yellow\",\"size\":\"A5|A6|A7\",\"lines\":number,\"price\":number,\"lead_time_minutes\":number|null,\"open_delivery\":boolean}]}",
    "Rules:",
    "- If delivery says Open, set open_delivery=true and lead_time_minutes=null.",
    "- If lead time exists, set open_delivery=false and lead_time_minutes to minutes as number.",
    "- Use only listed colours/sizes; best guess if unclear.",
    "- Output JSON only, no markdown."
  ].join("\n");

  const body = {
    model: state.llm.model,
    input: [
      {
        role: "user",
        content: [
          { type: "input_text", text: prompt },
          { type: "input_image", image_url: imageDataUrl },
        ],
      },
    ],
    temperature: 0,
  };

  const res = await fetch(state.llm.endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${state.llm.apiKey}`,
    },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`API ${res.status}: ${text.slice(0, 180)}`);
  }
  const json = await res.json();
  const text = extractResponseText(json);
  const parsed = extractJsonObject(text);
  if (!parsed || !Array.isArray(parsed.orders)) return [];
  return parsed.orders;
}

function extractResponseText(resp) {
  if (typeof resp.output_text === "string" && resp.output_text.trim()) return resp.output_text;
  if (Array.isArray(resp.output)) {
    const parts = [];
    resp.output.forEach((item) => {
      if (Array.isArray(item.content)) {
        item.content.forEach((c) => {
          if (typeof c.text === "string") parts.push(c.text);
        });
      }
    });
    if (parts.length) return parts.join("\n");
  }
  return JSON.stringify(resp);
}

function extractJsonObject(text) {
  const cleaned = String(text || "").trim();
  if (!cleaned) return null;
  try {
    return JSON.parse(cleaned);
  } catch {
    const start = cleaned.indexOf("{");
    const end = cleaned.lastIndexOf("}");
    if (start >= 0 && end > start) {
      try {
        return JSON.parse(cleaned.slice(start, end + 1));
      } catch {
        return null;
      }
    }
    return null;
  }
}

function fileToDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error("Could not read image file"));
    reader.readAsDataURL(file);
  });
}

function normalizeColour(input) {
  const value = String(input || "").trim().toLowerCase();
  const found = COLOUR_CHOICES.find((c) => c.toLowerCase() === value);
  return found || "White";
}

function normalizeSize(input) {
  const value = String(input || "").trim().toUpperCase();
  return SIZE_CHOICES.includes(value) ? value : "A6";
}

function isValidParsedOrder(order) {
  return (
    order.id &&
    order.qty > 0 &&
    COLOUR_CHOICES.includes(order.colour) &&
    SIZE_CHOICES.includes(order.size) &&
    order.lines > 0 &&
    order.price > 0 &&
    (order.openDelivery || order.leadTime >= 0)
  );
}

function setPhotoStatus(msg, isError) {
  const el = document.getElementById("photoParseStatus");
  el.textContent = msg;
  el.style.color = isError ? "var(--bad)" : "var(--muted)";
}

function startTimer() {
  if (state.timer.remainingSeconds <= 0) {
    state.timer.remainingSeconds = state.timer.totalSeconds;
  }
  state.timer.running = true;
  state.timer.lastTickMs = Date.now();
  ensureTimerLoop();
  saveState();
  renderTimer();
}

function pauseTimer() {
  syncTimer();
  state.timer.running = false;
  state.timer.lastTickMs = null;
  saveState();
  renderTimer();
}

function resetTimer() {
  const totalMinutes = Math.max(1, Number(document.getElementById("timerTotalMinutes").value || 120));
  state.timer.totalSeconds = Math.round(totalMinutes * 60);
  state.timer.remainingSeconds = state.timer.totalSeconds;
  state.timer.running = false;
  state.timer.lastTickMs = null;
  saveState();
  renderTimer();
}

function syncTimer() {
  if (!state.timer.running || !state.timer.lastTickMs) return;
  const now = Date.now();
  const elapsedSeconds = Math.floor((now - state.timer.lastTickMs) / 1000);
  if (elapsedSeconds <= 0) return;
  state.timer.remainingSeconds = Math.max(0, state.timer.remainingSeconds - elapsedSeconds);
  state.timer.lastTickMs += elapsedSeconds * 1000;
  if (state.timer.remainingSeconds <= 0) {
    state.timer.running = false;
    state.timer.lastTickMs = null;
  }
}

function ensureTimerLoop() {
  if (timerIntervalId) return;
  timerIntervalId = setInterval(() => {
    if (state.timer.running) {
      syncTimer();
      renderTimer();
      saveState();
    }
    processProcurementArrivals();
    renderOrders();
    renderProcurement();
  }, 1000);
}

function renderTimer() {
  syncTimer();
  const el = document.getElementById("timerDisplay");
  if (!el) return;
  el.textContent = formatSeconds(state.timer.remainingSeconds);
  el.style.color = state.timer.remainingSeconds === 0 ? "var(--bad)" : "#0c3e68";
  el.style.borderColor = state.timer.remainingSeconds === 0 ? "#dca6a6" : "#9ec0df";
}

function formatSeconds(totalSeconds) {
  const sec = Math.max(0, Math.floor(totalSeconds));
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = sec % 60;
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function renderAll() {
  renderTimer();
  renderDashboard();
  renderOrders();
  renderPhotoParseResults();
  renderProduction();
  renderProcurement();
  renderAccounts();
}

function renderDashboard() {
  const sales = state.ledger.filter((l) => l.type === "sale").reduce((a, l) => a + l.amount, 0);
  const penalties = state.ledger.filter((l) => l.type === "penalty").reduce((a, l) => a + Math.abs(l.amount), 0);
  const purchases = state.ledger.filter((l) => l.type === "purchase").reduce((a, l) => a + Math.abs(l.amount), 0);
  const cashNet = state.ledger.reduce((a, l) => a + l.amount, 0);
  const rejectedOrderValue = state.orders
    .filter((o) => o.status === "rejected")
    .reduce((a, o) => a + Number(o.price || 0), 0);
  const lostRevenue = rejectedOrderValue + penalties;
  const elapsedMinutes = Math.max(1, state.settings.gameMinute);
  const earningsPer10 = (sales / elapsedMinutes) * 10;
  const acceptedCount = state.orders.filter((o) => o.status === "accepted").length;
  const ongoing = state.orders.filter((o) => o.status === "in_progress");
  const ongoingCount = ongoing.length;
  const rejectedOnly = Math.max(0, rejectedOrderValue);

  recordDashboardSnapshot({
    minute: Number(state.settings.gameMinute || 0),
    revenue: Number(sales.toFixed(2)),
    earningsPer10: Number(earningsPer10.toFixed(2)),
    acceptedCount,
    ongoingCount,
    lostRevenue: Number(lostRevenue.toFixed(2)),
    penalties: Number(penalties.toFixed(2)),
    rejectedValue: Number(rejectedOnly.toFixed(2)),
  });

  document.getElementById("dashboardKpis").innerHTML = `
    <article class="card"><div class="kpi-label">Cumulative Revenue</div><div class="kpi-value">£${sales.toFixed(2)}</div></article>
    <article class="card"><div class="kpi-label">Net Cash</div><div class="kpi-value">£${cashNet.toFixed(2)}</div></article>
    <article class="card"><div class="kpi-label">Earnings / 10 min</div><div class="kpi-value">£${earningsPer10.toFixed(2)}</div></article>
    <article class="card"><div class="kpi-label">Accepted Orders</div><div class="kpi-value">${acceptedCount}</div></article>
    <article class="card"><div class="kpi-label">Ongoing Orders</div><div class="kpi-value">${ongoingCount}</div></article>
    <article class="card"><div class="kpi-label">Lost Revenue</div><div class="kpi-value">£${lostRevenue.toFixed(2)}</div></article>
    <article class="card"><div class="kpi-label">Purchase Spend</div><div class="kpi-value">£${purchases.toFixed(2)}</div></article>
  `;

  renderDashboardCharts();

  const acceptedRows = state.orders
    .filter((o) => o.status === "accepted")
    .map((o) => {
      const e = estimateOrder(o);
      return `<tr><td>${o.id}</td><td>${o.occasion}</td><td>${getOrderTimeLeftData(o, e).label}</td><td>£${o.price.toFixed(0)}</td></tr>`;
    })
    .join("");
  document.getElementById("dashboardAccepted").innerHTML = `
    <table>
      <thead><tr><th>Order</th><th>Occasion</th><th>Time Left</th><th>Value</th></tr></thead>
      <tbody>${acceptedRows || "<tr><td colspan='4'>No accepted orders.</td></tr>"}</tbody>
    </table>
  `;

  const ongoingRows = ongoing
    .map((o) => {
      const e = estimateOrder(o);
      const duration = Number.isFinite(o.startedMinute) ? Math.max(0, state.settings.gameMinute - o.startedMinute).toFixed(1) : "-";
      return `<tr><td>${o.id}</td><td>${o.occasion}</td><td>${duration}m</td><td>${getOrderTimeLeftData(o, e).label}</td></tr>`;
    })
    .join("");
  document.getElementById("dashboardOngoing").innerHTML = `
    <table>
      <thead><tr><th>Order</th><th>Occasion</th><th>Running For</th><th>Time Left</th></tr></thead>
      <tbody>${ongoingRows || "<tr><td colspan='4'>No ongoing orders.</td></tr>"}</tbody>
    </table>
  `;
}

function recordDashboardSnapshot(snap) {
  const hist = state.dashboardHistory || (state.dashboardHistory = []);
  const last = hist[hist.length - 1];
  let changed = false;
  if (last && last.minute === snap.minute) {
    const same = Object.keys(snap).every((k) => snap[k] === last[k]);
    if (!same) {
      hist[hist.length - 1] = snap;
      changed = true;
    }
  } else {
    hist.push(snap);
    changed = true;
  }
  if (hist.length > 300) {
    hist.splice(0, hist.length - 300);
    changed = true;
  }
  if (changed) saveState();
}

function renderDashboardCharts() {
  const hist = state.dashboardHistory || [];
  renderSvgLineChart("chartRevenue", hist.map((d) => ({ x: d.minute, y: d.revenue })), {
    lineColor: "#0f5e9c",
    fillColor: "rgba(15,94,156,0.10)",
    emptyLabel: "No revenue history yet",
    yPrefix: "£",
  });
  renderSvgLineChart("chartProductivity", hist.map((d) => ({ x: d.minute, y: d.earningsPer10 })), {
    lineColor: "#1f8a4c",
    fillColor: "rgba(31,138,76,0.10)",
    emptyLabel: "No productivity history yet",
    yPrefix: "£",
  });
  renderSvgMultiLineChart(
    "chartOrderFlow",
    [
      { name: "Accepted", color: "#b06a00", points: hist.map((d) => ({ x: d.minute, y: d.acceptedCount })) },
      { name: "Ongoing", color: "#0f5e9c", points: hist.map((d) => ({ x: d.minute, y: d.ongoingCount })) },
    ],
    { emptyLabel: "No order-flow history yet" }
  );

  const last = hist[hist.length - 1];
  const penalties = last ? last.penalties : 0;
  const rejectedValue = last ? last.rejectedValue : 0;
  renderSvgDonutChart("chartLoss", [
    { label: "Penalties", value: penalties, color: "#b02727" },
    { label: "Rejected Value", value: rejectedValue, color: "#d97a7a" },
  ]);
}

function chartDims(containerId) {
  const el = document.getElementById(containerId);
  if (!el) return null;
  // Use stable logical coordinates; CSS handles responsive scaling.
  const width = 600;
  const height = 150;
  return { el, width, height };
}

function renderSvgLineChart(containerId, points, opts = {}) {
  const dims = chartDims(containerId);
  if (!dims) return;
  const { el, width, height } = dims;
  if (!points.length) {
    el.innerHTML = `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none"><text x="12" y="${Math.floor(height / 2)}" fill="#6b7785" font-size="12">${opts.emptyLabel || "No data"}</text></svg>`;
    return;
  }
  const pad = { l: 36, r: 10, t: 10, b: 20 };
  const plotW = width - pad.l - pad.r;
  const plotH = height - pad.t - pad.b;
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMax = Math.max(1, ...ys);
  const toX = (x) => pad.l + (xMax === xMin ? 0 : ((x - xMin) / (xMax - xMin)) * plotW);
  const toY = (y) => pad.t + (1 - y / yMax) * plotH;
  const linePath = points.map((p, i) => `${i === 0 ? "M" : "L"}${toX(p.x)},${toY(p.y)}`).join(" ");
  const areaPath = `${linePath} L ${toX(points[points.length - 1].x)},${pad.t + plotH} L ${toX(points[0].x)},${pad.t + plotH} Z`;
  el.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
      <line x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${pad.t + plotH}" stroke="#d9e2ea"/>
      <line x1="${pad.l}" y1="${pad.t + plotH}" x2="${pad.l + plotW}" y2="${pad.t + plotH}" stroke="#d9e2ea"/>
      <path d="${areaPath}" fill="${opts.fillColor || "rgba(15,94,156,0.10)"}"></path>
      <path d="${linePath}" fill="none" stroke="${opts.lineColor || "#0f5e9c"}" stroke-width="2"></path>
      <text x="4" y="${pad.t + 8}" fill="#5c6b7a" font-size="10">${opts.yPrefix || ""}${yMax.toFixed(0)}</text>
      <text x="8" y="${pad.t + plotH}" fill="#5c6b7a" font-size="10">${opts.yPrefix || ""}0</text>
    </svg>
  `;
}

function renderSvgMultiLineChart(containerId, series, opts = {}) {
  const dims = chartDims(containerId);
  if (!dims) return;
  const { el, width, height } = dims;
  const allPoints = series.flatMap((s) => s.points);
  if (!allPoints.length) {
    el.innerHTML = `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none"><text x="12" y="${Math.floor(height / 2)}" fill="#6b7785" font-size="12">${opts.emptyLabel || "No data"}</text></svg>`;
    return;
  }
  const pad = { l: 36, r: 10, t: 10, b: 20 };
  const plotW = width - pad.l - pad.r;
  const plotH = height - pad.t - pad.b;
  const xs = allPoints.map((p) => p.x);
  const ys = allPoints.map((p) => p.y);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMax = Math.max(1, ...ys);
  const toX = (x) => pad.l + (xMax === xMin ? 0 : ((x - xMin) / (xMax - xMin)) * plotW);
  const toY = (y) => pad.t + (1 - y / yMax) * plotH;
  const paths = series.map((s) => ({
    name: s.name,
    color: s.color,
    d: s.points.map((p, i) => `${i === 0 ? "M" : "L"}${toX(p.x)},${toY(p.y)}`).join(" "),
  }));
  const legends = paths.map((p, i) => `
    <rect x="${width - 106}" y="${12 + i * 14}" width="10" height="10" fill="${p.color}"></rect>
    <text x="${width - 92}" y="${21 + i * 14}" fill="#2f3a46" font-size="11">${p.name}</text>
  `).join("");
  const pathNodes = paths.map((p) => `<path d="${p.d}" fill="none" stroke="${p.color}" stroke-width="2"></path>`).join("");
  el.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
      <line x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${pad.t + plotH}" stroke="#d9e2ea"/>
      <line x1="${pad.l}" y1="${pad.t + plotH}" x2="${pad.l + plotW}" y2="${pad.t + plotH}" stroke="#d9e2ea"/>
      ${pathNodes}
      ${legends}
    </svg>
  `;
}

function renderSvgDonutChart(containerId, parts) {
  const dims = chartDims(containerId);
  if (!dims) return;
  const { el, width, height } = dims;
  const total = parts.reduce((a, p) => a + p.value, 0);
  if (total <= 0) {
    el.innerHTML = `<svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none"><text x="12" y="${Math.floor(height / 2)}" fill="#6b7785" font-size="12">No loss recorded</text></svg>`;
    return;
  }
  const cx = width * 0.30;
  const cy = height * 0.5;
  const r = Math.min(width, height) * 0.28;
  const strokeW = Math.max(12, Math.floor(r * 0.5));
  let acc = -Math.PI / 2;
  const circles = parts.map((p) => {
    if (p.value <= 0) return "";
    const frac = p.value / total;
    const dash = frac * 2 * Math.PI * r;
    const gap = 2 * Math.PI * r - dash;
    const start = acc;
    acc += frac * 2 * Math.PI;
    const rot = (start * 180) / Math.PI;
    return `<circle cx="${cx}" cy="${cy}" r="${r}" fill="none" stroke="${p.color}" stroke-width="${strokeW}" stroke-dasharray="${dash} ${gap}" transform="rotate(${rot} ${cx} ${cy})"></circle>`;
  }).join("");
  const legend = parts.map((p, i) => `
    <rect x="${width * 0.58}" y="${20 + i * 18}" width="10" height="10" fill="${p.color}"></rect>
    <text x="${width * 0.58 + 16}" y="${29 + i * 18}" fill="#2f3a46" font-size="11">${p.label}: £${p.value.toFixed(0)}</text>
  `).join("");
  el.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
      ${circles}
      <text x="${width * 0.58}" y="12" fill="#2f3a46" font-size="11">Total £${total.toFixed(0)}</text>
      ${legend}
    </svg>
  `;
}

function setupCanvas(canvasId) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return null;
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const logicalWidth = Math.max(260, Math.floor(rect.width || canvas.clientWidth || 480));

  // Persist the intended logical canvas size once, then reuse it on every redraw.
  if (!canvas.dataset.baseHeight) {
    const initialHeight = Number(canvas.getAttribute("height"));
    canvas.dataset.baseHeight = String(Number.isFinite(initialHeight) && initialHeight > 0 ? initialHeight : 220);
  }
  const logicalHeight = Number(canvas.dataset.baseHeight);

  const pixelWidth = Math.floor(logicalWidth * dpr);
  const pixelHeight = Math.floor(logicalHeight * dpr);
  if (canvas.width !== pixelWidth) canvas.width = pixelWidth;
  if (canvas.height !== pixelHeight) canvas.height = pixelHeight;

  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, logicalWidth, logicalHeight);
  return { ctx, width: logicalWidth, height: logicalHeight };
}

function drawLineChart(canvasId, points, opts = {}) {
  const setup = setupCanvas(canvasId);
  if (!setup) return;
  const { ctx, width, height } = setup;
  const pad = { l: 40, r: 12, t: 12, b: 24 };
  const plotW = width - pad.l - pad.r;
  const plotH = height - pad.t - pad.b;
  if (!points.length) {
    drawEmpty(ctx, width, height, opts.emptyLabel || "No data");
    return;
  }
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = 0, yMax = Math.max(1, ...ys);
  drawAxes(ctx, pad, plotW, plotH, width, height, yMax, opts.yPrefix || "");
  const toX = (x) => pad.l + (xMax === xMin ? 0 : ((x - xMin) / (xMax - xMin)) * plotW);
  const toY = (y) => pad.t + (1 - (y - yMin) / (yMax - yMin)) * plotH;

  ctx.beginPath();
  points.forEach((p, i) => {
    const x = toX(p.x), y = toY(p.y);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.strokeStyle = opts.lineColor || "#0f5e9c";
  ctx.lineWidth = 2;
  ctx.stroke();

  if (opts.fillColor) {
    ctx.lineTo(toX(points[points.length - 1].x), pad.t + plotH);
    ctx.lineTo(toX(points[0].x), pad.t + plotH);
    ctx.closePath();
    ctx.fillStyle = opts.fillColor;
    ctx.fill();
  }
}

function drawMultiLineChart(canvasId, series, opts = {}) {
  const setup = setupCanvas(canvasId);
  if (!setup) return;
  const { ctx, width, height } = setup;
  const pad = { l: 40, r: 12, t: 12, b: 24 };
  const plotW = width - pad.l - pad.r;
  const plotH = height - pad.t - pad.b;
  const allPoints = series.flatMap((s) => s.points);
  if (!allPoints.length) {
    drawEmpty(ctx, width, height, opts.emptyLabel || "No data");
    return;
  }
  const xs = allPoints.map((p) => p.x);
  const ys = allPoints.map((p) => p.y);
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMax = Math.max(1, ...ys);
  drawAxes(ctx, pad, plotW, plotH, width, height, yMax, "");
  const toX = (x) => pad.l + (xMax === xMin ? 0 : ((x - xMin) / (xMax - xMin)) * plotW);
  const toY = (y) => pad.t + (1 - y / yMax) * plotH;

  series.forEach((s, idx) => {
    ctx.beginPath();
    s.points.forEach((p, i) => {
      const x = toX(p.x), y = toY(p.y);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = s.color;
    ctx.font = "12px sans-serif";
    ctx.fillText(s.name, width - 110, 18 + idx * 14);
  });
}

function drawDonutChart(canvasId, parts) {
  const setup = setupCanvas(canvasId);
  if (!setup) return;
  const { ctx, width, height } = setup;
  const total = parts.reduce((a, p) => a + p.value, 0);
  if (total <= 0) {
    drawEmpty(ctx, width, height, "No loss recorded");
    return;
  }
  const cx = width * 0.33;
  const cy = height * 0.5;
  const rOuter = Math.min(width, height) * 0.32;
  const rInner = rOuter * 0.55;
  let start = -Math.PI / 2;
  parts.forEach((p, i) => {
    if (p.value <= 0) return;
    const angle = (p.value / total) * Math.PI * 2;
    ctx.beginPath();
    ctx.arc(cx, cy, rOuter, start, start + angle);
    ctx.arc(cx, cy, rInner, start + angle, start, true);
    ctx.closePath();
    ctx.fillStyle = p.color;
    ctx.fill();
    start += angle;
  });
  ctx.fillStyle = "#1f2a36";
  ctx.font = "12px sans-serif";
  ctx.fillText(`Total £${total.toFixed(0)}`, width * 0.62, 24);
  parts.forEach((p, idx) => {
    ctx.fillStyle = p.color;
    ctx.fillRect(width * 0.62, 42 + idx * 18, 10, 10);
    ctx.fillStyle = "#1f2a36";
    ctx.fillText(`${p.label}: £${p.value.toFixed(0)}`, width * 0.62 + 16, 51 + idx * 18);
  });
}

function drawAxes(ctx, pad, plotW, plotH, width, height, yMax, yPrefix) {
  ctx.strokeStyle = "#d9e2ea";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.l, pad.t);
  ctx.lineTo(pad.l, pad.t + plotH);
  ctx.lineTo(pad.l + plotW, pad.t + plotH);
  ctx.stroke();
  ctx.fillStyle = "#5c6b7a";
  ctx.font = "11px sans-serif";
  ctx.fillText(`${yPrefix}${yMax.toFixed(0)}`, 4, pad.t + 8);
  ctx.fillText(`${yPrefix}0`, 10, pad.t + plotH);
}

function drawEmpty(ctx, width, height, label) {
  ctx.fillStyle = "#6b7785";
  ctx.font = "13px sans-serif";
  ctx.fillText(label, 14, Math.floor(height / 2));
}

function renderOrders() {
  const orderRecs = buildOrderRecommendations();
  const marketWrap = document.getElementById("orderTableWrap");
  const activeWrap = document.getElementById("activeOrdersWrap");

  const marketRows = state.orders
    .filter((o) => o.status === "board")
    .map((o) => {
      const e = estimateOrder(o);
      const deficit = materialDeficitForOrder(o, e);
      const recInfo = orderRecs.byRowId.get(o.rowId) || { label: "Skip", cls: "bad", stockLabel: "-" };
      const timeLeft = getOrderTimeLeftData(o, e);
      const poButtons = deficit.totalUnitsMissing > 0
        ? `<button onclick="createPurchaseOrderForOrder(${o.rowId}, 'team')">PO Team</button>
           <button onclick="createPurchaseOrderForOrder(${o.rowId}, 'admin')">PO Admin</button>`
        : "";
      return `
      <tr>
        <td>${o.id}</td>
        <td>${o.occasion}</td>
        <td>${o.qty}</td>
        <td>${o.lines}</td>
        <td>${o.colour}</td>
        <td>${o.size}</td>
        <td>£${o.price.toFixed(0)}</td>
        <td>${o.openDelivery ? "Open" : o.leadTime + "m"}</td>
        <td><span class="tag ${timeLeft.cls}">${timeLeft.label}</span></td>
        <td>${e.effectiveMinutes.toFixed(1)}m</td>
        <td>${Number.isFinite(e.estFinishMinute) ? e.estFinishMinute.toFixed(1) : "-"}</td>
        <td>£${e.cogs.toFixed(1)}</td>
        <td>£${e.expectedProfit.toFixed(1)}</td>
        <td>${Number(recInfo.score ?? e.score).toFixed(2)}</td>
        <td>${recInfo.stockLabel}</td>
        <td><span class="tag ${recInfo.cls}">${recInfo.label}</span></td>
        <td class="actions">
          <button onclick="actionAccept(${o.rowId})">Accept</button>
          <button onclick="actionStart(${o.rowId})">Start</button>
          ${poButtons}
          <button onclick="actionRemove(${o.rowId})">X</button>
        </td>
      </tr>
    `;
    })
    .join("");

  const activeRows = state.orders
    .filter((o) => o.status === "accepted" || o.status === "in_progress")
    .map((o) => {
    const e = estimateOrder(o);
    const recInfo = orderRecs.byRowId.get(o.rowId) || { label: "In Progress Flow", cls: "warn", stockLabel: "-" };
    const timeLeft = getOrderTimeLeftData(o, e);
    return `
      <tr>
        <td>${o.id}</td>
        <td>${o.occasion}</td>
        <td>${o.qty}</td>
        <td>${o.lines}</td>
        <td>${o.colour}</td>
        <td>${o.size}</td>
        <td>£${o.price.toFixed(0)}</td>
        <td>${o.openDelivery ? "Open" : o.leadTime + "m"}</td>
        <td><span class="tag ${timeLeft.cls}">${timeLeft.label}</span></td>
        <td>${e.effectiveMinutes.toFixed(1)}m</td>
        <td>${Number.isFinite(e.estFinishMinute) ? e.estFinishMinute.toFixed(1) : "-"}</td>
        <td>£${e.cogs.toFixed(1)}</td>
        <td>£${e.expectedProfit.toFixed(1)}</td>
        <td>${Number(recInfo.score ?? e.score).toFixed(2)}</td>
        <td>${recInfo.stockLabel}</td>
        <td><span class="tag ${recInfo.cls}">${recInfo.label}</span></td>
        <td><span class="tag ${statusTag(o.status)}">${o.status}</span></td>
        <td class="actions">
          <button onclick="actionStart(${o.rowId})">Start</button>
          <button onclick="actionDeliver(${o.rowId})">Deliver</button>
          <button onclick="actionReject(${o.rowId})">Reject</button>
          <button onclick="actionRemove(${o.rowId})">X</button>
        </td>
      </tr>
    `;
  }).join("");

  marketWrap.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>ID</th><th>Occasion</th><th>Qty</th><th>Lines</th><th>Colour</th><th>Size</th><th>Price</th><th>Due</th>
          <th>Time Left</th><th>Est Min</th><th>Est Finish</th><th>COGS</th><th>Exp Profit</th><th>Score</th><th>Stock</th><th>Rec</th><th>Actions</th>
        </tr>
      </thead>
      <tbody>${marketRows || "<tr><td colspan='17'>No marketplace orders available.</td></tr>"}</tbody>
    </table>
  `;

  activeWrap.innerHTML = `
    <table>
      <thead>
        <tr>
          <th>ID</th><th>Occasion</th><th>Qty</th><th>Lines</th><th>Colour</th><th>Size</th><th>Price</th><th>Due</th>
          <th>Time Left</th><th>Est Min</th><th>Est Finish</th><th>COGS</th><th>Exp Profit</th><th>Score</th><th>Stock</th><th>Rec</th><th>Status</th><th>Actions</th>
        </tr>
      </thead>
      <tbody>${activeRows || "<tr><td colspan='18'>No accepted or ongoing orders.</td></tr>"}</tbody>
    </table>
  `;
}

function buildOrderRecommendations() {
  const byRowId = new Map();
  const hurdleRate = getHurdleRate();
  const evaluations = [];

  state.orders.forEach((o) => {
    const est = estimateOrder(o);
    const deficit = materialDeficitForOrder(o, est);
    const stockLabel = formatStockLabel(deficit);
    const listPrice = paperListPrice(o);
    const stockScenario = computeAdjustedScenario(o, listPrice, { delayMins: 0, resourceFactor: 1.0 });
    const controllerScenario = computeAdjustedScenario(o, listPrice, { delayMins: 10, resourceFactor: 0.85 });
    const qty = Math.max(1, Number(o.qty || 1));
    const maxMarkupPrice = (Number(o.price || 0) - hurdleRate * stockScenario.totalPersonMins - 10) / qty;
    const teamScenarioAtList = computeAdjustedScenario(o, listPrice, { delayMins: 0, resourceFactor: 1.0 });

    if (o.status !== "board") {
      byRowId.set(o.rowId, {
        label: "In Progress Flow",
        cls: "warn",
        stockLabel,
        score: stockScenario.adjustedScore,
        hurdleRate,
        maxMarkupPrice,
      });
      return;
    }

    let recLabel = "Skip";
    let recClass = "bad";
    let scenarioScore = stockScenario.adjustedScore;
    let reason = "";

    if (deficit.totalUnitsMissing === 0) {
      if (stockScenario.feasible && stockScenario.netProfit > 0 && stockScenario.adjustedScore >= hurdleRate) {
        recLabel = "Take Now";
        recClass = "ok";
      } else {
        reason = stockScenario.adjustedScore < hurdleRate ? "Below Hurdle" : "Infeasible/Low Profit";
      }
      scenarioScore = stockScenario.adjustedScore;
    } else {
      const controllerViable =
        deficit.missingCost <= 1000 &&
        controllerScenario.feasible &&
        controllerScenario.netProfit > 0 &&
        controllerScenario.adjustedScore >= hurdleRate;
      const teamViablePotential =
        teamScenarioAtList.feasible &&
        teamScenarioAtList.netProfit > 0 &&
        maxMarkupPrice > 0 &&
        teamScenarioAtList.adjustedScore >= hurdleRate;

      if (teamViablePotential && (!controllerViable || teamScenarioAtList.adjustedScore >= controllerScenario.adjustedScore)) {
        recLabel = `Try Buy Team <=£${maxMarkupPrice.toFixed(2)}`;
        recClass = "warn";
        scenarioScore = teamScenarioAtList.adjustedScore;
      } else if (controllerViable) {
        recLabel = "Buy Admin + Take";
        recClass = "warn";
        scenarioScore = controllerScenario.adjustedScore;
      } else {
        reason = "No Viable Procurement";
        scenarioScore = Math.max(teamScenarioAtList.adjustedScore, controllerScenario.adjustedScore);
      }
    }

    const rec = {
      rowId: o.rowId,
      label: recLabel,
      cls: recClass,
      stockLabel,
      score: scenarioScore,
      hurdleRate,
      maxMarkupPrice,
      reason,
    };
    evaluations.push(rec);
    byRowId.set(o.rowId, rec);
  });

  const actionable = evaluations.filter((r) => r.label !== "Skip");
  const maxScore = actionable.length ? Math.max(...actionable.map((r) => r.score)) : null;
  const topRowIds = new Set(
    actionable
      .filter((r) => maxScore !== null && Math.abs(r.score - maxScore) <= 0.0001)
      .map((r) => r.rowId)
  );

  evaluations.forEach((r) => {
    if (r.label === "Skip") return;
    if (!topRowIds.has(r.rowId)) {
      byRowId.set(r.rowId, {
        ...r,
        label: "Skip (Not Top Score)",
        cls: "bad",
      });
    }
  });

  return { byRowId, topRowIds, maxScore, hurdleRate };
}

function materialDeficitForOrder(order, est = estimateOrder(order)) {
  const colourKey = order.colour.toLowerCase();
  const need = { [colourKey]: est.sheetNeed, envelope: 1 };
  const have = { [colourKey]: Number(state.stock[colourKey] || 0), envelope: Number(state.stock.envelope || 0) };
  const missingSheets = Math.max(0, need[colourKey] - have[colourKey]);
  const missingEnvelope = Math.max(0, need.envelope - have.envelope);
  const totalUnitsMissing = missingSheets + missingEnvelope;
  const missingCost = missingSheets * PRICES[colourKey] + missingEnvelope * PRICES.envelope;
  return { colourKey, missingSheets, missingEnvelope, totalUnitsMissing, missingCost, need, have };
}

function missingItemsList(deficit) {
  const list = [];
  if (!deficit) return list;
  if (Number(deficit.missingSheets || 0) > 0) {
    list.push({ key: deficit.colourKey, qty: Number(deficit.missingSheets) });
  }
  if (Number(deficit.missingEnvelope || 0) > 0) {
    list.push({ key: "envelope", qty: Number(deficit.missingEnvelope) });
  }
  return list;
}

function formatStockLabel(deficit) {
  if (!deficit || deficit.totalUnitsMissing === 0) return "Yes";
  const bits = missingItemsList(deficit).map((x) => `${x.qty} ${x.key}`);
  return `No (need ${bits.join(", ")})`;
}

function getOrderTimeLeftData(order, est) {
  if (order.openDelivery) return { label: "Open", cls: "time-good" };
  if (!Number.isFinite(order.takenMinute) && !order.takenTs) return { label: "-", cls: "time-good" };

  let remainSec = null;
  const totalSec = Math.max(1, Math.round(order.leadTime * 60));

  if (order.takenTs) {
    const elapsedSec = Math.max(0, Math.floor((Date.now() - Date.parse(order.takenTs)) / 1000));
    remainSec = totalSec - elapsedSec;
  } else {
    const remainMin = est.dueMinute - state.settings.gameMinute;
    remainSec = Math.round(remainMin * 60);
  }

  const ratio = remainSec / totalSec;
  let cls = "time-bad";
  if (remainSec < 0) cls = "time-bad";
  else if (ratio > 0.66) cls = "time-good";
  else if (ratio > 0.33) cls = "time-warn";
  else cls = "time-bad";

  if (remainSec >= 0) return { label: `${formatSecondCountdown(remainSec)} left`, cls };
  return { label: `${formatSecondCountdown(-remainSec)} overdue`, cls: "time-bad" };
}

function formatMinuteCountdown(minutesFloat) {
  const totalSec = Math.max(0, Math.round(minutesFloat * 60));
  const m = Math.floor(totalSec / 60);
  const s = totalSec % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function formatSecondCountdown(totalSecRaw) {
  const totalSec = Math.max(0, Math.round(totalSecRaw));
  const m = Math.floor(totalSec / 60);
  const s = totalSec % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function statusTag(status) {
  if (status === "delivered") return "ok";
  if (status === "rejected") return "bad";
  if (status === "accepted" || status === "in_progress") return "warn";
  return "";
}

function renderProduction() {
  const queue = state.orders.filter((o) => o.status === "accepted" || o.status === "in_progress");
  let cursor = state.settings.gameMinute;
  const qRows = queue.map((o) => {
    const e = estimateOrder(o);
    const start = cursor;
    const end = start + e.effectiveMinutes;
    cursor = end;
    return `<tr><td>${o.id}</td><td>${o.occasion}</td><td>${start.toFixed(1)}</td><td>${end.toFixed(1)}</td><td>${e.effectiveMinutes.toFixed(1)}m</td></tr>`;
  }).join("");

  document.getElementById("productionQueue").innerHTML = `
    <table>
      <thead><tr><th>Order</th><th>Occasion</th><th>Start Minute</th><th>End Minute</th><th>Duration</th></tr></thead>
      <tbody>${qRows || "<tr><td colspan='5'>No accepted/in-progress orders.</td></tr>"}</tbody>
    </table>
  `;

  const model = getAdaptiveModel();
  document.getElementById("learningStats").innerHTML = `
    <table>
      <tbody>
        <tr><th>Training Samples</th><td>${model.sampleCount}</td></tr>
        <tr><th>Confidence</th><td>${model.confidence}</td></tr>
        <tr><th>Global Time Factor</th><td>${model.globalFactor.toFixed(2)}x</td></tr>
        <tr><th>A5 Factor</th><td>${model.sizeFactor.A5.toFixed(2)}x</td></tr>
        <tr><th>A6 Factor</th><td>${model.sizeFactor.A6.toFixed(2)}x</td></tr>
        <tr><th>A7 Factor</th><td>${model.sizeFactor.A7.toFixed(2)}x</td></tr>
      </tbody>
    </table>
  `;

  const completed = state.orders
    .filter((o) => (o.status === "delivered" || o.status === "rejected"))
    .slice()
    .reverse();
  const completedRows = completed.map((o) => {
    const pred = Number(o.predictedMinutesAtStart || 0);
    const actual = Number(o.actualDurationMin || 0);
    const err = pred > 0 && actual > 0 ? ((actual - pred) / pred) * 100 : null;
    return `
      <tr>
        <td>${o.id}</td>
        <td>${o.status}</td>
        <td>${pred > 0 ? pred.toFixed(1) : "-"}</td>
        <td>${actual > 0 ? actual.toFixed(1) : "-"}</td>
        <td>${err === null ? "-" : `${err.toFixed(1)}%`}</td>
        <td>${Number.isFinite(o.startedMinute) ? o.startedMinute : "-"}</td>
        <td>${Number.isFinite(o.completedMinute) ? o.completedMinute : "-"}</td>
      </tr>
    `;
  }).join("");
  document.getElementById("completedOrders").innerHTML = `
    <table>
      <thead><tr><th>Order</th><th>Status</th><th>Pred Min (Start)</th><th>Actual Min</th><th>Error</th><th>Start Min</th><th>End Min</th></tr></thead>
      <tbody>${completedRows || "<tr><td colspan='7'>No completed orders yet.</td></tr>"}</tbody>
    </table>
  `;

  document.getElementById("taskBreakdown").innerHTML = `
    <table>
      <thead><tr><th>Stage</th><th>Description</th></tr></thead>
      <tbody>
        <tr><td>1. Cut/Fold</td><td>Paper cutting and fold alignment by card size (A5/A6/A7).</td></tr>
        <tr><td>2. Front Stencil</td><td>Stencil occasion title with required size stencil.</td></tr>
        <tr><td>3. Verse</td><td>Copy consistent rhyming verse in pen.</td></tr>
        <tr><td>4. Back ID</td><td>Write manufacturer ID + order number on rear corner.</td></tr>
        <tr><td>5. Pack + SODN</td><td>Single-order envelope with SODN and external labels.</td></tr>
        <tr><td>6. Deliver</td><td>Submit to controller before due time.</td></tr>
      </tbody>
    </table>
  `;
}

function renderProcurement() {
  const placedInput = document.getElementById("procurementPlacedTime");
  if (placedInput) placedInput.value = formatClock(Date.now());
  renderControllerCooldownBanner();

  document.getElementById("stockView").innerHTML = `
    <table>
      <thead><tr><th>Item</th><th>Qty</th></tr></thead>
      <tbody>${Object.entries(state.stock).map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("")}</tbody>
    </table>
  `;

  const rows = state.procurementLog.map((p) => {
    const countdown = procurementCountdownData(p);
    return `
    <tr>
      <td>#${p.rowId}</td>
      <td>${p.counterparty || (p.source === "admin" ? "Controllers" : "Team")}</td>
      <td>${Number.isFinite(p.placedAtMs) ? formatClock(p.placedAtMs) : p.placedMinute}</td>
      <td>${Number.isFinite(p.dueAtMs) ? formatClock(p.dueAtMs) : p.dueMinute}</td>
      <td><span class="tag ${countdown.cls}">${countdown.label}</span></td>
      <td>£${p.total}</td>
      <td>${p.received ? "Received" : "Pending"}</td>
      <td>${Object.entries(p.items).map(([k, q]) => `${k}:${q}`).join(", ")}</td>
    </tr>`;
  }).join("");

  document.getElementById("procurementLog").innerHTML = `
    <table>
      <thead><tr><th>ID</th><th>Source</th><th>Placed</th><th>Due</th><th>Arrival Countdown</th><th>Total</th><th>Status</th><th>Items</th></tr></thead>
      <tbody>${rows || "<tr><td colspan='8'>No procurement orders placed.</td></tr>"}</tbody>
    </table>
  `;

  updateProcurementDraftSummary();
}

function procurementCountdownData(po) {
  const nowMs = Date.now();
  const dueMs = Number.isFinite(po.dueAtMs) ? po.dueAtMs : nowMs;
  const remainMs = dueMs - nowMs;
  if (po.received) return { label: "Arrived", cls: "time-good" };
  const totalMsRaw = Number.isFinite(po.dueAtMs) && Number.isFinite(po.placedAtMs)
    ? Math.max(1, po.dueAtMs - po.placedAtMs)
    : 10 * 60 * 1000;
  const ratio = remainMs / totalMsRaw;
  let cls = "time-bad";
  if (remainMs < 0) cls = "time-bad";
  else if (ratio > 0.66) cls = "time-good";
  else if (ratio > 0.33) cls = "time-warn";
  else cls = "time-bad";
  if (remainMs >= 0) return { label: `${formatMsCountdown(remainMs)} left`, cls };
  return { label: `${formatMsCountdown(-remainMs)} overdue`, cls: "time-bad" };
}

function formatClock(ms) {
  const d = new Date(ms);
  const hh = String(d.getHours()).padStart(2, "0");
  const mm = String(d.getMinutes()).padStart(2, "0");
  const ss = String(d.getSeconds()).padStart(2, "0");
  return `${hh}:${mm}:${ss}`;
}

function formatMsCountdown(ms) {
  const sec = Math.max(0, Math.floor(ms / 1000));
  const mm = Math.floor(sec / 60);
  const ss = sec % 60;
  return `${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")}`;
}

function renderAccounts() {
  const assetsNow = Object.entries(state.stock).reduce((sum, [k, q]) => sum + (PRICES[k] || 0) * q, 0);
  const assetRecovery = 0.3 * assetsNow;
  const cashNet = state.ledger.reduce((a, x) => a + x.amount, 0);
  const projectedProfit = cashNet + assetRecovery;

  document.getElementById("pnlView").innerHTML = `
    <table>
      <tbody>
        <tr><th>Net Cashflow</th><td>£${cashNet.toFixed(2)}</td></tr>
        <tr><th>Current Asset Value</th><td>£${assetsNow.toFixed(2)}</td></tr>
        <tr><th>End Game Asset Recovery (30%)</th><td>£${assetRecovery.toFixed(2)}</td></tr>
        <tr><th>Projected Profit/Loss</th><td><strong>£${projectedProfit.toFixed(2)}</strong></td></tr>
      </tbody>
    </table>
  `;

  const rows = state.ledger.slice().reverse().map((l) => `
    <tr><td>${l.ts.replace("T", " ").slice(0, 19)}</td><td>${l.type}</td><td>£${l.amount.toFixed(2)}</td><td>${l.note}</td></tr>
  `).join("");

  document.getElementById("ledgerView").innerHTML = `
    <table>
      <thead><tr><th>Time</th><th>Type</th><th>Amount</th><th>Note</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function renderPhotoParseResults() {
  const view = state.settings.parsedOrdersView || "hidden";
  if (view === "hidden") {
    document.getElementById("photoParseResults").innerHTML = "<p class='hint'>Parsed orders table is hidden.</p>";
    return;
  }
  const rows = (state.photoParseLast || []).map((r) => `
    <tr>
      <td>${r.id}</td><td>${r.occasion}</td><td>${r.qty}</td><td>${r.colour}</td><td>${r.size}</td>
      <td>£${Number(r.price).toFixed(0)}</td><td>${r.lead}</td><td>£${Number(r.expectedProfit).toFixed(1)}</td>
      <td>${Number(r.score).toFixed(2)}</td><td>${r.recommendation}</td>
    </tr>
  `).join("");
  document.getElementById("photoParseResults").innerHTML = `
    <table>
      <thead>
        <tr><th>ID</th><th>Occasion</th><th>Qty</th><th>Colour</th><th>Size</th><th>Price</th><th>Due</th><th>Exp Profit</th><th>Score</th><th>Recommendation</th></tr>
      </thead>
      <tbody>${rows || "<tr><td colspan='10'>No parsed orders yet.</td></tr>"}</tbody>
    </table>
  `;
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

window.actionAccept = actionAccept;
window.actionStart = actionStart;
window.actionDeliver = actionDeliver;
window.actionReject = actionReject;
window.actionRemove = actionRemove;
window.createPurchaseOrderForOrder = createPurchaseOrderForOrder;
