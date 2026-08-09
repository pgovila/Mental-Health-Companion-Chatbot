// chat.js — Mental Health Companion Web Frontend

// ── State ─────────────────────────────────────────────────────────────────
const moodCounts    = { positive: 0, negative: 0, neutral: 0 };
let   polaritySum   = 0;
let   turnCount     = 0;
let   isWaiting     = false;

// ── DOM refs ──────────────────────────────────────────────────────────────
const messagesEl  = document.getElementById("messages");
const inputEl     = document.getElementById("user-input");
const sendBtn     = document.getElementById("send-btn");
const typingEl    = document.getElementById("typing-indicator");

// ── Init ──────────────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
  appendBotBubble(
    "Hello! I'm your Mental Health Companion. 💙\n\n" +
    "I'm here to listen — share how you're feeling, and I'll respond with " +
    "empathy and support.\n\n" +
    "You can also use the quick-action buttons on the left, or just type naturally.",
    false
  );
  inputEl.focus();
});

// ── Input handling ────────────────────────────────────────────────────────
inputEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

inputEl.addEventListener("input", () => {
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + "px";
});

// ── Send a normal chat message ────────────────────────────────────────────
function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || isWaiting) return;

  appendUserBubble(text);
  inputEl.value = "";
  inputEl.style.height = "auto";
  postToBot(text);
}

// ── Send a sidebar command ────────────────────────────────────────────────
function sendCommand(cmd) {
  if (isWaiting) return;
  appendUserBubble(cmd);
  postToBot(cmd);
}

// ── API call ──────────────────────────────────────────────────────────────
async function postToBot(text) {
  setWaiting(true);

  try {
    const res  = await fetch("/api/chat", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ message: text }),
    });
    const data = await res.json();
    handleResponse(data);
  } catch (err) {
    appendBotBubble("Sorry, something went wrong. Please refresh the page and try again.", false);
  } finally {
    setWaiting(false);
  }
}

// ── Handle API response ───────────────────────────────────────────────────
function handleResponse(data) {
  // Mood badge for the user's last bubble
  if (data.analysis) {
    attachMoodBadge(data.analysis);
    updateSidebar(data.analysis);
  }

  // Main bot reply
  appendBotBubble(data.text, data.is_crisis);

  // Relaxation tip card
  if (data.tip) {
    appendTipCard(data.tip);
  }

  // Session summary on farewell
  if (data.is_farewell && data.summary) {
    appendSummaryCard(data.summary);
  }
}

// ── DOM helpers ───────────────────────────────────────────────────────────
function appendUserBubble(text) {
  const wrap = document.createElement("div");
  wrap.className = "bubble-wrap user";
  wrap.dataset.last = "true";   // used by attachMoodBadge

  const label = document.createElement("div");
  label.className = "bubble-label";
  label.textContent = "You";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  wrap.appendChild(label);
  wrap.appendChild(bubble);
  messagesEl.appendChild(wrap);
  scrollBottom();
}

function appendBotBubble(text, isCrisis) {
  const wrap = document.createElement("div");
  wrap.className = "bubble-wrap bot" + (isCrisis ? " crisis" : "");

  const label = document.createElement("div");
  label.className = "bubble-label";
  label.textContent = isCrisis ? "⚠ Crisis Support" : "Companion";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;

  wrap.appendChild(label);
  wrap.appendChild(bubble);
  messagesEl.appendChild(wrap);
  scrollBottom();
}

function appendTipCard(tipText) {
  const card = document.createElement("div");
  card.className = "tip-card";

  const title = document.createElement("div");
  title.className = "tip-card-title";
  title.textContent = "🌿 Relaxation Exercise";

  const body = document.createElement("div");
  body.textContent = tipText;

  card.appendChild(title);
  card.appendChild(body);
  messagesEl.appendChild(card);
  scrollBottom();
}

function appendSummaryCard(summaryText) {
  const card = document.createElement("div");
  card.className = "summary-card";
  card.textContent = summaryText;
  messagesEl.appendChild(card);
  scrollBottom();
}

// Attach inline mood badge beneath the most-recent user bubble
function attachMoodBadge(analysis) {
  // Find last user bubble-wrap
  const wraps = messagesEl.querySelectorAll(".bubble-wrap.user");
  if (!wraps.length) return;
  const lastWrap = wraps[wraps.length - 1];

  const badge = document.createElement("div");
  badge.className = "mood-badge-inline";

  const tag = document.createElement("span");
  tag.className = "mood-tag " + analysis.sentiment;
  tag.textContent = analysis.emotion.toUpperCase();

  const detail = document.createElement("span");
  detail.textContent = `${analysis.sentiment} · ${analysis.intensity}% intensity`;

  badge.appendChild(tag);
  badge.appendChild(detail);
  lastWrap.appendChild(badge);
}

// ── Sidebar updates ───────────────────────────────────────────────────────
function updateSidebar(analysis) {
  // Mood counts
  moodCounts[analysis.sentiment] = (moodCounts[analysis.sentiment] || 0) + 1;
  document.getElementById("cnt-positive").textContent = moodCounts.positive;
  document.getElementById("cnt-negative").textContent = moodCounts.negative;
  document.getElementById("cnt-neutral").textContent  = moodCounts.neutral;

  // Running average polarity fill (polarity is -1..1, map to 0..100%)
  polaritySum += analysis.polarity;
  turnCount++;
  const avgPolarity  = polaritySum / turnCount;            // -1..1
  const fillPercent  = ((avgPolarity + 1) / 2) * 100;     // 0..100
  document.getElementById("polarity-fill").style.width = fillPercent.toFixed(1) + "%";

  // Emotion badge
  const badgeEl = document.getElementById("emotion-badge");
  badgeEl.className = "emotion-badge " + analysis.emotion;
  badgeEl.textContent = analysis.emotion.toUpperCase();
  document.getElementById("last-emotion-section").style.display = "block";
}

// ── Waiting state ─────────────────────────────────────────────────────────
function setWaiting(on) {
  isWaiting       = on;
  sendBtn.disabled = on;
  typingEl.classList.toggle("hidden", !on);
  if (on) scrollBottom();
}

// ── Reset session ─────────────────────────────────────────────────────────
async function resetSession() {
  await fetch("/api/reset", { method: "POST" });

  // Clear UI
  messagesEl.innerHTML = "";
  moodCounts.positive = moodCounts.negative = moodCounts.neutral = 0;
  polaritySum = turnCount = 0;
  document.getElementById("cnt-positive").textContent = "0";
  document.getElementById("cnt-negative").textContent = "0";
  document.getElementById("cnt-neutral").textContent  = "0";
  document.getElementById("polarity-fill").style.width = "50%";
  document.getElementById("last-emotion-section").style.display = "none";
  document.getElementById("emotion-badge").className = "emotion-badge";
  document.getElementById("emotion-badge").textContent = "—";

  appendBotBubble(
    "Session reset! 🌱 Ready to start fresh. Share how you're feeling whenever you're ready.",
    false
  );
  inputEl.focus();
}

// ── Utilities ─────────────────────────────────────────────────────────────
function scrollBottom() {
  setTimeout(() => {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }, 50);
}
