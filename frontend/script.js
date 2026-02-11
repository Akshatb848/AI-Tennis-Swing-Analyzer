/* ================================================================
   Education Intelligence Dashboard — Frontend Logic
   Ministry of Education, Government of India
   ================================================================ */

const API_BASE = window.location.origin;
const FETCH_TIMEOUT = 15000; // 15s
const MAX_RETRIES = 2;

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
        setStatusDot(vdbEl, "green", "Vector DB: Ready");
    } else {
        setStatusDot(vdbEl, "red", "Vector DB: Offline");
    }

    setStatusDot(apiEl, "green", "API: Healthy");

    if (badge) {
        badge.textContent = "Healthy";
        badge.className = "health-badge healthy";
    }
}

function setStatusDot(el, colour, text) {
    if (!el) return;
    const dot = el.querySelector(".status-dot");
    if (dot) { dot.className = "status-dot " + colour; }
    el.innerHTML = "";
    const dotEl = document.createElement("span");
    dotEl.className = "status-dot " + colour;
    el.appendChild(dotEl);
    el.appendChild(document.createTextNode(" " + text));
}

// Refresh health every 30s
setInterval(checkSystemHealth, 30000);

/* ================================================================
   CHAT
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

    const data = await apiFetch("/api/chat", {
        method: "POST",
        body: JSON.stringify({ query: query, current_month: null, include_visualization: false }),
    });

    removeTyping();
    chatBusy = false;
    document.getElementById("send-btn").disabled = false;

    if (!data) {
        appendMessage("ai", "I apologise — the service is currently unreachable. Please check the system status and try again.");
        return;
    }

    appendMessage("ai", data.answer, data.sources);
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

    // Sources
    const showSources = document.getElementById("show-sources");
    if (sources && sources.length > 0 && showSources && showSources.checked) {
        const srcDiv = document.createElement("div");
        srcDiv.className = "message-sources";
        srcDiv.textContent = "Sources: " + sources.join(", ");
        content.appendChild(srcDiv);
    }

    msg.appendChild(avatar);
    msg.appendChild(content);
    container.appendChild(msg);
    container.scrollTop = container.scrollHeight;
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
    appendMessage("ai", "Session cleared. How can I help you?");
}

function formatMarkdown(text) {
    // Minimal markdown: bold, bullet points, line breaks
    return text
        .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
        .replace(/^- (.+)/gm, "&bull; $1")
        .replace(/\n/g, "<br>");
}

/* ================================================================
   DASHBOARD — Month Selector
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
}

function onMonthChange() {
    // Could trigger data refresh for the selected period
    const sel = document.getElementById("month-selector");
    if (sel) console.log("Period changed:", sel.value);
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
            <div class="search-result-score">Relevance: ${(r.relevance * 100).toFixed(0)}%</div>
        </div>
    `).join("");
}

/* ================================================================
   DASHBOARD — Comparison
   ================================================================ */

async function runComparison() {
    const a = document.getElementById("compare-a");
    const b = document.getElementById("compare-b");
    const metric = document.getElementById("compare-metric");
    const resultsEl = document.getElementById("compare-results");
    if (!a || !b || !resultsEl) return;

    const entityA = a.value.trim();
    const entityB = b.value.trim();
    if (!entityA || !entityB) {
        showToast("Please enter both entities to compare.");
        return;
    }

    resultsEl.textContent = "Generating comparison...";

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

    resultsEl.textContent = data.comparison || "No comparison data available.";
}
