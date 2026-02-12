/* ================================================================
   Education Intelligence Dashboard — Frontend Logic
   Ministry of Education, Government of India
   v2.0 — Structured responses, calendar-aware, conference-ready
   ================================================================ */

const API_BASE = window.location.origin;
const FETCH_TIMEOUT = 15000;
const MAX_RETRIES = 2;

/* ---- State ---- */
let currentSessionId = null;
let currentMonth = null;

/* ---- Helpers ---- */

function sanitise(text) {
    const div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
}

async function apiFetch(path, options = {}, retries = MAX_RETRIES) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT);

    try {
        const resp = await fetch(`${API_BASE}${path}`, {
            ...options,
            signal: controller.signal,
            headers: { "Content-Type": "application/json", ...(options.headers || {}) },
        });
        clearTimeout(timeout);

        if (resp.status === 429) {
            showToast("Rate limit reached. Please wait a moment.");
            return null;
        }
        if (!resp.ok) {
            const body = await resp.json().catch(() => ({}));
            throw new Error(body.detail || `HTTP ${resp.status}`);
        }
        return await resp.json();
    } catch (err) {
        clearTimeout(timeout);
        if (retries > 0 && (err.name === "AbortError" || err.message.includes("fetch"))) {
            await new Promise(r => setTimeout(r, 1000));
            return apiFetch(path, options, retries - 1);
        }
        console.error(`API error (${path}):`, err);
        return null;
    }
}

/* ---- Toast notifications ---- */

function showToast(message, duration = 4000) {
    const existing = document.querySelector(".error-toast");
    if (existing) existing.remove();

    const toast = document.createElement("div");
    toast.className = "error-toast";
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), duration);
}

/* ---- Fullscreen ---- */

function toggleFullscreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch(() => {});
    } else {
        document.exitFullscreen();
    }
}

/* ================================================================
   SYSTEM HEALTH
   ================================================================ */

async function checkSystemHealth() {
    const data = await apiFetch("/api/health");
    const llmEl = document.getElementById("status-llm");
    const vdbEl = document.getElementById("status-vdb");
    const apiEl = document.getElementById("status-api");
    const badge = document.getElementById("health-badge");

    if (!data) {
        setStatusDot(llmEl, "red", "LLM: Offline");
        setStatusDot(vdbEl, "red", "Vector DB: Offline");
        setStatusDot(apiEl, "red", "API: Offline");
        if (badge) { badge.textContent = "Offline"; badge.className = "health-badge unhealthy"; }
        return;
    }

    const svc = data.services || {};

    if (svc.llm && svc.llm.connected) {
        setStatusDot(llmEl, "green", "LLM: Connected");
    } else {
        setStatusDot(llmEl, "amber", "LLM: Unavailable");
    }

    if (svc.vector_db && svc.vector_db.connected) {
        setStatusDot(vdbEl, "green", "RAG: " + (svc.vector_db.detail || "Ready"));
    } else {
        setStatusDot(vdbEl, "red", "RAG: Offline");
    }

    setStatusDot(apiEl, "green", "API: Healthy");

    if (badge) {
        badge.textContent = "Healthy";
        badge.className = "health-badge healthy";
    }
}

function setStatusDot(el, colour, text) {
    if (!el) return;
    el.innerHTML = "";
    const dotEl = document.createElement("span");
    dotEl.className = "status-dot " + colour;
    el.appendChild(dotEl);
    el.appendChild(document.createTextNode(" " + text));
}

setInterval(checkSystemHealth, 30000);

/* ================================================================
   CHAT — Structured Responses
   ================================================================ */

let chatBusy = false;

function handleChatKeydown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

function autoGrow(el) {
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 120) + "px";
}

function sendDemo(btn) {
    const input = document.getElementById("chat-input");
    if (input) {
        input.value = btn.textContent;
        sendMessage();
    }
}

async function sendMessage() {
    if (chatBusy) return;

    const input = document.getElementById("chat-input");
    const raw = (input.value || "").trim();
    if (!raw) return;

    const query = raw.substring(0, 2000);
    input.value = "";
    input.style.height = "auto";

    appendMessage("user", query);
    showTyping();
    chatBusy = true;
    document.getElementById("send-btn").disabled = true;

    const monthSel = document.getElementById("chat-month-selector");
    const selectedMonth = monthSel ? monthSel.value : currentMonth;

    const data = await apiFetch("/api/chat", {
        method: "POST",
        body: JSON.stringify({
            query: query,
            current_month: selectedMonth || null,
            include_visualization: true,
            session_id: currentSessionId,
        }),
    });

    removeTyping();
    chatBusy = false;
    document.getElementById("send-btn").disabled = false;

    if (!data) {
        appendMessage("ai", "I apologise — the service is currently unreachable. Please check the system status and try again.");
        return;
    }

    if (data.session_id) {
        currentSessionId = data.session_id;
    }

    appendStructuredMessage(data);
}

function appendStructuredMessage(data) {
    const container = document.getElementById("chat-messages");
    if (!container) return;

    const msg = document.createElement("div");
    msg.className = "message ai-message";

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = "AI";

    const content = document.createElement("div");
    content.className = "message-content structured-response";

    // Intent badge + confidence
    const meta = document.createElement("div");
    meta.className = "response-meta";
    if (data.intent && data.intent !== "general" && data.intent !== "error") {
        const badge = document.createElement("span");
        badge.className = "intent-badge intent-" + data.intent;
        badge.textContent = data.intent.replace("_", " ");
        meta.appendChild(badge);
    }
    if (data.confidence) {
        const conf = document.createElement("span");
        conf.className = "confidence-indicator confidence-" + data.confidence;
        conf.textContent = data.confidence + " confidence";
        meta.appendChild(conf);
    }
    if (data.data_backed) {
        const dbBadge = document.createElement("span");
        dbBadge.className = "data-backed-badge";
        dbBadge.textContent = "data-verified";
        meta.appendChild(dbBadge);
    }
    if (meta.children.length > 0) {
        content.appendChild(meta);
    }

    // Summary
    const summary = document.createElement("div");
    summary.className = "response-summary";
    summary.innerHTML = formatMarkdown(sanitise(data.summary || data.answer || ""));
    content.appendChild(summary);

    // Key points
    if (data.key_points && data.key_points.length > 0) {
        const kpDiv = document.createElement("div");
        kpDiv.className = "key-points";
        const kpHeader = document.createElement("div");
        kpHeader.className = "key-points-header";
        kpHeader.textContent = "Key Data Points";
        kpDiv.appendChild(kpHeader);
        const ul = document.createElement("ul");
        data.key_points.forEach(kp => {
            const li = document.createElement("li");
            li.innerHTML = formatMarkdown(sanitise(kp));
            ul.appendChild(li);
        });
        kpDiv.appendChild(ul);
        content.appendChild(kpDiv);
    }

    // Sources
    const showSources = document.getElementById("show-sources");
    if (data.sources && data.sources.length > 0 && showSources && showSources.checked) {
        const srcDiv = document.createElement("div");
        srcDiv.className = "message-sources";
        srcDiv.innerHTML = "<strong>Sources:</strong> " +
            data.sources.map(s => {
                const label = s.month ? `${s.month} (${s.section || s.type})` : (s.type || "data");
                return `<span class="source-tag">${sanitise(label)}</span>`;
            }).join(" ");
        content.appendChild(srcDiv);
    }

    // Visualization hint
    if (data.visualization && data.visualization.type) {
        const vizHint = document.createElement("div");
        vizHint.className = "viz-hint";
        vizHint.textContent = "Chart available: " + data.visualization.type.replace("_", " ");
        vizHint.onclick = () => showVisualization(data.visualization);
        content.appendChild(vizHint);
    }

    msg.appendChild(avatar);
    msg.appendChild(content);
    container.appendChild(msg);
    container.scrollTop = container.scrollHeight;
}

function appendMessage(role, text, sources) {
    const container = document.getElementById("chat-messages");
    if (!container) return;

    const msg = document.createElement("div");
    msg.className = "message " + (role === "user" ? "user-message" : "ai-message");

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = role === "user" ? "You" : "AI";

    const content = document.createElement("div");
    content.className = "message-content";
    content.innerHTML = formatMarkdown(sanitise(text));

    msg.appendChild(avatar);
    msg.appendChild(content);
    container.appendChild(msg);
    container.scrollTop = container.scrollHeight;
}

function showVisualization(vizData) {
    // Placeholder — dispatch event for dashboard chart highlighting
    console.log("Visualization hint:", vizData);
    showToast("Chart data available — view on Dashboard");
}

function showTyping() {
    const container = document.getElementById("chat-messages");
    if (!container) return;

    const msg = document.createElement("div");
    msg.className = "message ai-message";
    msg.id = "typing-indicator";

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = "AI";

    const dots = document.createElement("div");
    dots.className = "typing-indicator";
    dots.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';

    msg.appendChild(avatar);
    msg.appendChild(dots);
    container.appendChild(msg);
    container.scrollTop = container.scrollHeight;
}

function removeTyping() {
    const el = document.getElementById("typing-indicator");
    if (el) el.remove();
}

function clearChat() {
    const container = document.getElementById("chat-messages");
    if (!container) return;
    container.innerHTML = "";
    currentSessionId = null;
    appendMessage("ai", "Session cleared. How can I help you?");
}

function formatMarkdown(text) {
    return text
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(/^- (.+)/gm, "&bull; $1")
        .replace(/\n/g, "<br>");
}

/* ================================================================
   DASHBOARD — Month Selector (calendar-driven)
   ================================================================ */

async function loadMonths() {
    const sel = document.getElementById("month-selector");
    if (!sel) return;

    const data = await apiFetch("/api/newsletter/months");
    if (!data || !data.months) {
        sel.innerHTML = '<option value="">Unavailable</option>';
        return;
    }

    sel.innerHTML = "";
    data.months.forEach(m => {
        const opt = document.createElement("option");
        opt.value = m;
        opt.textContent = m;
        if (m === data.default) opt.selected = true;
        sel.appendChild(opt);
    });

    currentMonth = data.default || data.months[data.months.length - 1];
    onMonthChange();

    // Also populate chat month selector if present
    const chatSel = document.getElementById("chat-month-selector");
    if (chatSel) {
        chatSel.innerHTML = '<option value="">Auto-detect</option>';
        data.months.forEach(m => {
            const opt = document.createElement("option");
            opt.value = m;
            opt.textContent = m;
            chatSel.appendChild(opt);
        });
    }
}

async function onMonthChange() {
    const sel = document.getElementById("month-selector");
    if (!sel) return;
    currentMonth = sel.value;
    await loadMonthData(currentMonth);
}

async function loadMonthData(month) {
    if (!month) return;
    const encoded = month.replace(/ /g, "-");
    const data = await apiFetch(`/api/newsletter/${encoded}`);
    if (!data) return;

    // Update KPI cards
    const kpis = data.kpis || {};
    updateKPI("kpi-ger-primary", kpis.ger_primary, "%");
    updateKPI("kpi-ger-he", kpis.ger_higher_education, "%");
    updateKPI("kpi-ptr", kpis.pupil_teacher_ratio_primary, ":1");
    updateKPI("kpi-literacy", kpis.literacy_rate, "%");

    // Update theme
    const themeEl = document.getElementById("month-theme");
    if (themeEl) themeEl.textContent = data.theme || "";

    // Update highlights
    const hlEl = document.getElementById("month-highlights");
    if (hlEl && data.highlights) {
        hlEl.innerHTML = data.highlights.map(h =>
            `<li>${sanitise(h)}</li>`
        ).join("");
    }
}

function updateKPI(elementId, value, suffix) {
    const el = document.getElementById(elementId);
    if (!el || value === undefined) return;
    const valEl = el.querySelector(".kpi-value");
    if (valEl) valEl.textContent = value + (suffix || "");
}

/* ================================================================
   DASHBOARD — Search (debounced)
   ================================================================ */

let searchTimer = null;

function debouncedSearch() {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(executeSearch, 400);
}

async function executeSearch() {
    const input = document.getElementById("search-input");
    const spinner = document.getElementById("search-spinner");
    const results = document.getElementById("search-results");
    if (!input || !results) return;

    const query = input.value.trim();
    if (!query) { results.innerHTML = ""; return; }

    if (spinner) spinner.classList.remove("hidden");

    const data = await apiFetch("/api/search", {
        method: "POST",
        body: JSON.stringify({ query: query, top_k: 5 }),
    });

    if (spinner) spinner.classList.add("hidden");

    if (!data || !data.results) {
        results.innerHTML = '<p style="color: var(--grey-500); font-size: 0.85rem;">No results found.</p>';
        return;
    }

    results.innerHTML = data.results.map(r => `
        <div class="search-result-item">
            <div class="search-result-title">${sanitise(r.title)}</div>
            <div class="search-result-snippet">${sanitise(r.snippet)}</div>
            <div class="search-result-score">Relevance: ${(r.relevance * 100).toFixed(0)}%${r.month ? ' | ' + sanitise(r.month) : ''}</div>
        </div>
    `).join("");
}

/* ================================================================
   DASHBOARD — Comparison
   ================================================================ */

async function loadCompareStates() {
    const selA = document.getElementById("compare-a-sel");
    const selB = document.getElementById("compare-b-sel");
    if (!selA || !selB) return;

    const data = await apiFetch("/api/compare/states");
    if (!data || !data.states) return;

    [selA, selB].forEach(sel => {
        sel.innerHTML = '<option value="">Select state...</option>';
        data.states.forEach(s => {
            const opt = document.createElement("option");
            opt.value = s;
            opt.textContent = s;
            sel.appendChild(opt);
        });
    });
}

async function runComparison() {
    const a = document.getElementById("compare-a-sel") || document.getElementById("compare-a");
    const b = document.getElementById("compare-b-sel") || document.getElementById("compare-b");
    const metric = document.getElementById("compare-metric");
    const resultsEl = document.getElementById("compare-results");
    if (!a || !b || !resultsEl) return;

    const entityA = a.value.trim();
    const entityB = b.value.trim();
    if (!entityA || !entityB) {
        showToast("Please select both states to compare.");
        return;
    }

    resultsEl.innerHTML = '<div class="loading-text">Generating comparison...</div>';

    const data = await apiFetch("/api/compare", {
        method: "POST",
        body: JSON.stringify({
            entities: [entityA, entityB],
            metric: metric ? metric.value.trim() || null : null,
        }),
    });

    if (!data) {
        resultsEl.textContent = "Comparison service unavailable. Please try again.";
        return;
    }

    if (typeof data.comparison === "object") {
        resultsEl.innerHTML = renderComparisonTable(data.comparison, data.month);
    } else {
        resultsEl.textContent = data.comparison || "No comparison data available.";
    }
}

function renderComparisonTable(states, month) {
    const stateNames = Object.keys(states);
    if (stateNames.length === 0) return "<p>No data available for these states.</p>";

    const allMetrics = new Set();
    stateNames.forEach(s => Object.keys(states[s]).forEach(k => allMetrics.add(k)));

    let html = `<div class="compare-month-label">Data for: ${sanitise(month || "Latest")}</div>`;
    html += '<table class="compare-table"><thead><tr><th>Metric</th>';
    stateNames.forEach(s => { html += `<th>${sanitise(s)}</th>`; });
    html += '</tr></thead><tbody>';

    allMetrics.forEach(metric => {
        html += `<tr><td>${sanitise(metric.replace(/_/g, " "))}</td>`;
        stateNames.forEach(s => {
            const val = states[s][metric];
            html += `<td>${val !== undefined ? val : "—"}</td>`;
        });
        html += '</tr>';
    });

    html += '</tbody></table>';
    return html;
}

/* ================================================================
   INIT
   ================================================================ */

document.addEventListener("DOMContentLoaded", () => {
    checkSystemHealth();
    if (document.getElementById("month-selector")) {
        loadMonths();
        loadCompareStates();
    }
    if (document.getElementById("chat-month-selector")) {
        loadChatMonths();
    }
});

async function loadChatMonths() {
    const sel = document.getElementById("chat-month-selector");
    if (!sel) return;
    const data = await apiFetch("/api/newsletter/months");
    if (!data || !data.months) return;
    sel.innerHTML = '<option value="">Auto-detect</option>';
    data.months.forEach(m => {
        const opt = document.createElement("option");
        opt.value = m;
        opt.textContent = m;
        sel.appendChild(opt);
    });
}
